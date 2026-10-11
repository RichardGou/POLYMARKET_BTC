"""One day of taker trades, rules fixed before looking at the result.

The oracle model is the one selected on 8-9 October. The book is read on
10 October only. A trade is a buy of one share, five seconds before the
window ends, and only when three things are true at once:

- the model is at least $8 away from the price to beat
- the weighted exchange price jumped by at least $15 in the previous 20 seconds
- the share we want is still priced at 80 cents or less, and the last print
  is at most 15 seconds old

Fee per share: 0.07 * p * (1 - p), the published crypto taker formula.
The fill is the last traded price. A second column adds one cent of slippage.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.fit_oracle import load_labels
from utils.gamma import iter_events
from utils.http import session
from utils.preprocess_fit import build_features, load_prices, load_usdt_usd
from utils.providers import LABEL_END, LABEL_START

ROOT = Path(__file__).resolve().parent
TEST_DAY = datetime(2026, 10, 10, tzinfo=timezone.utc)
TEST_DAY_END = datetime(2026, 10, 11, tzinfo=timezone.utc)
CONFIDENT_USD = 8.0
SUDDEN_USD = 15.0
SUDDEN_LOOKBACK = 20
MAX_ENTRY = 0.80
MIN_ENTRY = 0.02
MAX_QUOTE_AGE = 15
FEE_RATE = 0.07
SLIPPAGE = 0.01


def fee(price: float) -> float:
    raw = FEE_RATE * price * (1.0 - price)
    rounded = round(raw + 1e-15, 5)
    return 0.0 if rounded < 1e-5 else rounded


def load_model() -> dict:
    summary = json.loads((ROOT / "data" / "fit" / "preprocess_summary.json").read_text(encoding="utf-8"))
    return summary["best_on_train"]


def markets_for_day(client) -> pd.DataFrame:
    rows = []
    for timeframe, end_min, end_max in (
        ("5m", TEST_DAY, TEST_DAY_END),
        ("15m", TEST_DAY, TEST_DAY_END),
    ):
        for event in iter_events(client, timeframe, end_min, end_max - pd.Timedelta(seconds=1)):
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
                    "end": end,
                    "end_sec": int(end.timestamp()),
                    "up_token": token_ids[0],
                    "down_token": token_ids[1],
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("slug")
    return frame


def fetch_quotes(client, markets: pd.DataFrame) -> pd.DataFrame:
    jobs = []
    for record in markets.itertuples(index=False):
        decision = record.end_sec - 5
        for side, token in (("up", record.up_token), ("down", record.down_token)):
            jobs.append((record.slug, side, token, decision))

    def one(job):
        slug, side, token, decision = job
        local = session()
        response = local.get(
            "https://data-api.polymarket.com/v2/prices-history",
            params={"token_id": token, "as_of": decision},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        if not data:
            return {"slug": slug, "side": side, "quote_ts": pd.NA, "quote": np.nan}
        point = data[-1]
        return {
            "slug": slug,
            "side": side,
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
            if done % 100 == 0:
                print(f"quotes {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def spot_at(features_raw: dict[str, pd.Series], alpha: dict[str, float], second: int) -> float:
    total = 0.0
    for name, weight in alpha.items():
        series = features_raw[name]
        if second not in series.index or not np.isfinite(series.loc[second]):
            return np.nan
        total += weight * float(series.loc[second])
    return total


def main() -> None:
    model = load_model()
    alpha = model["alpha"]
    deltas = {name: int(value) for name, value in model["deltas"].items()}
    base = float(model["base_usd"])
    client = session()
    print("loading markets", flush=True)
    markets = markets_for_day(client)
    print(f"markets {len(markets)}", flush=True)
    quotes = fetch_quotes(client, markets)
    wide = quotes.pivot(index="slug", columns="side", values=["quote", "quote_ts"])
    wide.columns = [f"{side}_{field}" for field, side in wide.columns]
    book = markets.merge(wide.reset_index(), on="slug", how="left")

    opening, close = load_labels(ROOT)
    opening["oracle_ts"] = pd.to_datetime(opening["oracle_ts"], utc=True)
    close["oracle_ts"] = pd.to_datetime(close["oracle_ts"], utc=True)
    strike = opening[["slug", "price"]].rename(columns={"price": "strike"})
    settled = close[["slug", "price"]].rename(columns={"price": "settle"})
    book = book.merge(strike, on="slug").merge(settled, on="slug")

    prices = load_prices()
    usdt = load_usdt_usd()
    twap = build_features(prices, usdt, model["ffill_seconds"], model["window_seconds"], model["aggregator"])
    raw = {}
    from utils.preprocess_fit import dense

    usdt_dense = dense(usdt, limit=180)
    for name, series in prices.items():
        level = dense(series, model["ffill_seconds"])
        if name in {"binance", "okx"}:
            rate = usdt_dense.reindex(level.index).to_numpy()
            level = pd.Series(level.to_numpy() * rate, index=level.index)
        raw[name] = level

    predictions = []
    jumps = []
    for end_sec in book["end_sec"]:
        estimate = base
        ok = True
        for name, weight in alpha.items():
            stamp = int(end_sec) + deltas[name]
            value = twap[name].get(stamp, np.nan)
            if not np.isfinite(value):
                ok = False
                break
            estimate += weight * float(value)
        predictions.append(estimate if ok else np.nan)
        jumps.append(spot_at(raw, alpha, int(end_sec) - 5) - spot_at(raw, alpha, int(end_sec) - 5 - SUDDEN_LOOKBACK))
    book["pred_close"] = predictions
    book["jump_20s"] = jumps
    book["edge"] = book["pred_close"] - book["strike"]
    book["buy_up"] = book["edge"] >= 0
    book["entry"] = np.where(book["buy_up"], book["up_quote"], book["down_quote"])
    book["quote_ts"] = np.where(book["buy_up"], book["up_quote_ts"], book["down_quote_ts"])
    book["quote_age"] = (book["end_sec"] - 5) - book["quote_ts"].astype("float")
    book["won"] = np.where(book["buy_up"], book["settle"] >= book["strike"], book["settle"] < book["strike"])
    book["confident"] = book["edge"].abs() >= CONFIDENT_USD
    book["sudden"] = book["jump_20s"].abs() >= SUDDEN_USD
    book["fresh"] = book["quote_age"].between(0, MAX_QUOTE_AGE)
    book["misaligned"] = book["entry"].between(MIN_ENTRY, MAX_ENTRY)
    book["trade"] = book["confident"] & book["sudden"] & book["fresh"] & book["misaligned"] & book["pred_close"].notna()

    candidates = book.loc[book["confident"] & book["sudden"] & book["fresh"] & book["pred_close"].notna()].copy()
    candidates["fee_at_entry"] = candidates["entry"].map(lambda value: fee(float(value)) if np.isfinite(value) else np.nan)
    candidates["pnl_if_filled"] = np.where(
        candidates["won"], 1.0 - candidates["entry"], -candidates["entry"]
    ) - candidates["fee_at_entry"]
    taken = book.loc[book["trade"]].copy()
    taken["fee"] = taken["entry"].map(fee)
    taken["pnl"] = np.where(taken["won"], 1.0 - taken["entry"], -taken["entry"]) - taken["fee"]
    slipped = np.minimum(0.99, taken["entry"] + SLIPPAGE)
    taken["pnl_plus_1c"] = np.where(taken["won"], 1.0 - slipped, -slipped) - slipped.map(fee)

    def pack(frame: pd.DataFrame, pnl_col: str) -> dict:
        if frame.empty:
            return {"n": 0}
        return {
            "n": int(len(frame)),
            "hit_rate": float(frame["won"].mean()),
            "mean_entry": float(frame["entry"].mean()),
            "fees": float(frame["fee"].sum()),
            "pnl": float(frame[pnl_col].sum()),
            "pnl_per_trade": float(frame[pnl_col].mean()),
        }

    funnel = {
        "windows": int(len(book)),
        "confident": int(book["confident"].sum()),
        "confident_and_sudden": int((book["confident"] & book["sudden"]).sum()),
        "plus_fresh_quote": int((book["confident"] & book["sudden"] & book["fresh"]).sum()),
        "traded": int(book["trade"].sum()),
    }
    by_time = {}
    for timeframe, part in taken.groupby("timeframe"):
        by_time[timeframe] = pack(part, "pnl")
    result = {
        "day": "2026-10-10",
        "rules": {
            "confident_usd": CONFIDENT_USD,
            "sudden_usd": SUDDEN_USD,
            "sudden_lookback_seconds": SUDDEN_LOOKBACK,
            "max_entry": MAX_ENTRY,
            "max_quote_age_seconds": MAX_QUOTE_AGE,
            "fee": "0.07 * p * (1-p) per share",
        },
        "funnel": funnel,
        "at_last_trade": pack(taken, "pnl"),
        "with_one_cent_slippage": pack(taken.assign(pnl_plus_1c=taken["pnl_plus_1c"]) if len(taken) else taken, "pnl_plus_1c"),
        "by_timeframe": by_time,
    }
    # The slippage pack reused taken["won"] but pack() uses pnl_col only. Fix mean fee still from original entry.
    if len(taken):
        result["with_one_cent_slippage"] = {
            "n": int(len(taken)),
            "hit_rate": float(taken["won"].mean()),
            "mean_entry": float(np.minimum(0.99, taken["entry"] + SLIPPAGE).mean()),
            "fees": float(np.minimum(0.99, taken["entry"] + SLIPPAGE).map(fee).sum()),
            "pnl": float(taken["pnl_plus_1c"].sum()),
            "pnl_per_trade": float(taken["pnl_plus_1c"].mean()),
        }
    if len(candidates):
        result["when_confident_and_sudden"] = {
            "n": int(len(candidates)),
            "hit_rate": float(candidates["won"].mean()),
            "mean_entry": float(candidates["entry"].mean()),
            "median_entry": float(candidates["entry"].median()),
            "entry_at_or_below_80c": int((candidates["entry"] <= MAX_ENTRY).sum()),
            "entry_above_95c": int((candidates["entry"] > 0.95).sum()),
            "pnl_if_we_bought_anyway": float(candidates["pnl_if_filled"].sum()),
        }
    out = ROOT / "data" / "fit"
    out.mkdir(parents=True, exist_ok=True)
    taken.to_parquet(out / "lead_trades.parquet", index=False)
    (out / "lead_backtest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
