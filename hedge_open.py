"""Fair probability while a window is still open, against the last trade.

The oracle level at clock t is the fitted exchange average for the stamp
t+4, which is the latest stamp Coinbase's lag lets us price. The distance
to the price to beat is divided by the move still plausible over the
remaining time. A logistic curve, fit on 8-9 October only, turns that
z-score into P(Up).

On 10 October, once a minute, if that probability beats the last traded
price by more than the taker fee plus 3 cents, buy one share of the cheap
side and hold it to the end. One trade per window, the first signal.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.fit_oracle import TRAIN_END, load_labels
from utils.gamma import iter_events
from utils.http import session
from utils.preprocess_fit import build_features, dense, load_prices, load_usdt_usd

ROOT = Path(__file__).resolve().parent
TEST_DAY = datetime(2026, 10, 10, tzinfo=timezone.utc)
TEST_DAY_END = datetime(2026, 10, 11, tzinfo=timezone.utc)
FEE_RATE = 0.07
EDGE_BUFFER = 0.03
MIN_ENTRY = 0.05
MAX_ENTRY = 0.95
MAX_QUOTE_AGE = 30
DURATIONS = {"5m": 300, "15m": 900}


def fee(price: float) -> float:
    raw = FEE_RATE * price * (1.0 - price)
    rounded = round(raw + 1e-15, 5)
    return 0.0 if rounded < 1e-5 else rounded


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_logistic(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def nll(theta: np.ndarray) -> float:
        p = sigmoid(theta[0] + theta[1] * z)
        return float(-np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12)))

    result = minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


def prediction_by_stamp(features: dict[str, pd.Series], model: dict) -> pd.Series:
    anchor = features["coinbase"].index.to_numpy()
    pred = np.full(len(anchor), float(model["base_usd"]))
    ok = np.ones(len(anchor), dtype=bool)
    for name, weight in model["alpha"].items():
        looked = features[name].reindex(anchor + int(model["deltas"][name])).to_numpy(dtype="float64")
        ok &= np.isfinite(looked)
        pred = pred + float(weight) * np.where(np.isfinite(looked), looked, 0.0)
    pred[~ok] = np.nan
    return pd.Series(pred, index=anchor)


def markets_for_day(client) -> pd.DataFrame:
    rows = []
    end_max = TEST_DAY_END - pd.Timedelta(seconds=1)
    for timeframe in ("5m", "15m"):
        for event in iter_events(client, timeframe, TEST_DAY, end_max):
            market = (event.get("markets") or [{}])[0]
            token_ids = json.loads(market.get("clobTokenIds") or "[]")
            outcomes = json.loads(market.get("outcomes") or "[]")
            if len(token_ids) < 2 or outcomes[:2] != ["Up", "Down"]:
                continue
            end = pd.Timestamp(event["endDate"])
            rows.append(
                {
                    "timeframe": timeframe,
                    "slug": event["slug"],
                    "end_sec": int(end.timestamp()),
                    "up_token": token_ids[0],
                    "down_token": token_ids[1],
                }
            )
    return pd.DataFrame(rows).drop_duplicates("slug")


def fetch_quotes(jobs: list[tuple]) -> pd.DataFrame:
    def one(job):
        slug, side, token, decision = job
        response = session().get(
            "https://data-api.polymarket.com/v2/prices-history",
            params={"token_id": token, "as_of": int(decision)},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        if not data:
            return {"slug": slug, "side": side, "decision": decision, "quote_ts": np.nan, "quote": np.nan}
        point = data[-1]
        return {
            "slug": slug,
            "side": side,
            "decision": decision,
            "quote_ts": int(point["timestamp"]),
            "quote": float(point["price"]),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"quotes {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def checkpoints(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in frame.itertuples(index=False):
        duration = DURATIONS[record.timeframe]
        start = int(record.end_sec) - duration
        for offset in range(60, duration - 30, 60):
            rows.append(
                {
                    "slug": record.slug,
                    "timeframe": record.timeframe,
                    "end_sec": int(record.end_sec),
                    "strike": float(record.strike),
                    "settle": float(record.settle),
                    "up": int(record.settle >= record.strike),
                    "decision_stamp": start + offset,
                    "tau": int(record.end_sec) - (start + offset),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    model = json.loads((ROOT / "data" / "fit" / "preprocess_summary.json").read_text(encoding="utf-8"))["best_on_train"]
    rmse = float(model["train"]["rmse"])
    features = build_features(
        load_prices(),
        load_usdt_usd(),
        model["ffill_seconds"],
        model["window_seconds"],
        model["aggregator"],
    )
    pred = prediction_by_stamp(features, model)
    train_mask = (pred.index >= int(datetime(2026, 10, 8, tzinfo=timezone.utc).timestamp())) & (
        pred.index < int(TRAIN_END.timestamp())
    )
    changes = pred.loc[train_mask].diff()
    sigma = float(changes.std(skipna=True))
    print(f"sigma_per_second {sigma:.4f} train_rmse {rmse:.4f}", flush=True)

    opening, close = load_labels(ROOT)
    opening["oracle_ts"] = pd.to_datetime(opening["oracle_ts"], utc=True)
    close["oracle_ts"] = pd.to_datetime(close["oracle_ts"], utc=True)
    windows = opening[["slug", "timeframe", "oracle_ts", "price"]].rename(columns={"price": "strike", "oracle_ts": "start"})
    windows = windows.merge(close[["slug", "price"]].rename(columns={"price": "settle"}), on="slug")
    windows["end_sec"] = (windows["start"].astype("int64") // 10**9) + windows["timeframe"].map(DURATIONS)
    windows = windows[(windows["start"] >= "2026-10-08") & (windows["start"] < "2026-10-11")]

    samples = checkpoints(windows.rename(columns={"start": "window_start"}))
    samples["level"] = pred.reindex(samples["decision_stamp"].to_numpy()).to_numpy()
    scale = np.sqrt(sigma**2 * samples["tau"].to_numpy() + rmse**2)
    samples["z"] = (samples["level"] - samples["strike"]) / scale
    usable = samples["z"].notna() & samples["level"].notna()
    train = samples[usable & (samples["decision_stamp"] < int(TRAIN_END.timestamp()))]
    intercept, slope = fit_logistic(train["z"].to_numpy(), train["up"].to_numpy().astype(float))
    print(f"logistic intercept {intercept:.3f} slope {slope:.3f} train_n {len(train)}", flush=True)
    samples["fair_up"] = sigmoid(intercept + slope * samples["z"].to_numpy())

    test = samples[usable & (samples["decision_stamp"] >= int(TEST_DAY.timestamp())) & (samples["decision_stamp"] < int(TEST_DAY_END.timestamp()))].copy()
    print(f"test checkpoints {len(test)}", flush=True)

    client = session()
    listed = markets_for_day(client)
    test = test.merge(listed[["slug", "up_token", "down_token"]], on="slug", how="inner", suffixes=("", "_book"))
    jobs = [(row.slug, "up", row.up_token, row.decision_stamp) for row in test.itertuples(index=False)]
    quotes = fetch_quotes(jobs)
    quotes = quotes.rename(columns={"quote": "up_quote", "quote_ts": "up_quote_ts"})
    test = test.merge(quotes[["slug", "decision", "up_quote", "up_quote_ts"]], left_on=["slug", "decision_stamp"], right_on=["slug", "decision"], how="left")
    test["up_age"] = test["decision_stamp"] - test["up_quote_ts"]
    test["edge_up"] = test["fair_up"] - test["up_quote"] - test["up_quote"].map(lambda value: fee(float(value)) if np.isfinite(value) else np.nan)
    # Down is fetched only after the up price says the up share is rich.
    rich = test[(test["edge_up"] < -EDGE_BUFFER) & test["up_quote"].between(MIN_ENTRY, MAX_ENTRY) & test["up_age"].between(0, MAX_QUOTE_AGE)]
    down_jobs = [(row.slug, "down", row.down_token, row.decision_stamp) for row in rich.itertuples(index=False)]
    print(f"down quotes {len(down_jobs)}", flush=True)
    if down_jobs:
        down = fetch_quotes(down_jobs).rename(columns={"quote": "down_quote", "quote_ts": "down_quote_ts"})
        test = test.merge(down[["slug", "decision", "down_quote", "down_quote_ts"]], left_on=["slug", "decision_stamp"], right_on=["slug", "decision"], how="left", suffixes=("", "_down"))
    else:
        test["down_quote"] = np.nan
        test["down_quote_ts"] = np.nan
    test["down_age"] = test["decision_stamp"] - test["down_quote_ts"]
    test["fair_down"] = 1.0 - test["fair_up"]
    test["edge_down"] = test["fair_down"] - test["down_quote"] - test["down_quote"].map(lambda value: fee(float(value)) if np.isfinite(value) else np.nan)

    test["buy_up"] = (test["edge_up"] > EDGE_BUFFER) & test["up_quote"].between(MIN_ENTRY, MAX_ENTRY) & test["up_age"].between(0, MAX_QUOTE_AGE)
    test["buy_down"] = (test["edge_down"] > EDGE_BUFFER) & test["down_quote"].between(MIN_ENTRY, MAX_ENTRY) & test["down_age"].between(0, MAX_QUOTE_AGE)
    test = test.sort_values(["slug", "decision_stamp"])
    chosen = []
    seen = set()
    for row in test.itertuples(index=False):
        if row.slug in seen:
            continue
        if row.buy_up:
            chosen.append((row, "up", row.up_quote, row.edge_up))
            seen.add(row.slug)
        elif row.buy_down:
            chosen.append((row, "down", row.down_quote, row.edge_down))
            seen.add(row.slug)
    trades = []
    for row, side, entry, edge in chosen:
        won = bool(row.up) if side == "up" else not bool(row.up)
        paid = fee(float(entry))
        pnl = (1.0 - float(entry) if won else -float(entry)) - paid
        trades.append(
            {
                "slug": row.slug,
                "timeframe": row.timeframe,
                "side": side,
                "entry": float(entry),
                "fair": float(row.fair_up if side == "up" else row.fair_down),
                "z": float(row.z),
                "tau": int(row.tau),
                "edge": float(edge),
                "won": won,
                "fee": paid,
                "pnl": pnl,
                "stake": float(entry),
            }
        )
    taken = pd.DataFrame(trades)
    bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    test["z_bin"] = pd.cut(test["z"], bins)
    calibration = (
        test.groupby("z_bin", observed=False)
        .agg(n=("up", "size"), realized=("up", "mean"), fair=("fair_up", "mean"))
        .reset_index()
    )
    result = {
        "sigma_per_second": sigma,
        "logistic_intercept": intercept,
        "logistic_slope": slope,
        "train_checkpoints": int(len(train)),
        "test_checkpoints": int(len(test)),
        "trades": int(len(taken)),
        "hit_rate": None if taken.empty else float(taken["won"].mean()),
        "stake": None if taken.empty else float(taken["stake"].sum()),
        "fees": None if taken.empty else float(taken["fee"].sum()),
        "pnl": None if taken.empty else float(taken["pnl"].sum()),
        "pnl_per_trade": None if taken.empty else float(taken["pnl"].mean()),
        "by_timeframe": {},
        "calibration": [
            {"z": str(row.z_bin), "n": int(row.n), "realized": None if pd.isna(row.realized) else float(row.realized), "fair": None if pd.isna(row.fair) else float(row.fair)}
            for row in calibration.itertuples(index=False)
        ],
    }
    if not taken.empty:
        for timeframe, part in taken.groupby("timeframe"):
            result["by_timeframe"][timeframe] = {
                "n": int(len(part)),
                "hit_rate": float(part["won"].mean()),
                "stake": float(part["stake"].sum()),
                "pnl": float(part["pnl"].sum()),
            }
    out = ROOT / "data" / "fit"
    out.mkdir(parents=True, exist_ok=True)
    if not taken.empty:
        taken.to_parquet(out / "open_hedge_trades.parquet", index=False)
    (out / "open_hedge.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
