"""One-month study: refit the oracle formula, then test two trading ideas.

1. Expiry latency: just before (and just after) the end of a window our
   estimate of the final TWAP is almost exact. Are there fills on the
   winning side at prices that still leave money after fees?
2. Open window: a calibrated Up probability from our oracle estimate and
   the price to beat. Does the market price agree with it, and do the
   disagreements pay?

Formula and probability model are fitted on 11 Sep - 1 Oct. Everything
reported as "test" uses 1 Oct - 11 Oct only, with Polymarket taker fills.

Assumptions, stated once:
- The price to beat is known live from the Chainlink TWAP stream relayed by
  Polymarket, so the backtest uses the Gamma value.
- Our exchange data reaches us INFO_DELAY seconds late.
- A buy at clock t fills at the first taker BUY fill of that token in
  seconds t+1..t+3, for at most that fill's size and MAX_SHARES.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.exchanges import VENUES
from utils.month import LABEL_END, LABEL_START, ROOT, TEST_END, TEST_START, TRAIN_END
from utils.oracle_month import (
    PriceGrid,
    estimate_at,
    fit_formula,
    fit_probability,
    predict,
    probability,
    score,
    spot_vol,
    vol_at,
)
from utils.polymarket_trades import load_market

OUT = ROOT / "data" / "month"
INFO_DELAY = 1
FILL_HORIZON = 3
MAX_SHARES = 10.0
FEE_RATE = 0.07
EDGES = (0.02, 0.05, 0.10)
MAIN_EDGE = 0.05
MAX_QUOTE_AGE = 10
DURATION = {"5m": 300, "15m": 900}
ANCHOR_HALFLIVES = (None, 1, 2, 3, 5, 8, 12, 24)
ANCHOR_DELAY = 5
BUCKETS = (
    ("fenetre ouverte (tau > 60 s)", 60, 10_000),
    ("derniere minute (5 < tau <= 60 s)", 5, 60),
    ("5 dernieres secondes (0 < tau <= 5 s)", 0, 5),
    ("apres la fin, avant resolution (tau <= 0)", -10_000, 0),
)


def epoch(values: pd.Series) -> np.ndarray:
    return (values.astype("int64") // 10**9).to_numpy(dtype="int64")


def load_windows() -> pd.DataFrame:
    opening = pd.read_parquet(ROOT / "data" / "price_oracle_opening.parquet")
    close = pd.read_parquet(ROOT / "data" / "price_oracle_close.parquet")
    frame = opening[["timeframe", "slug", "condition_id", "oracle_ts", "price"]].rename(
        columns={"oracle_ts": "start", "price": "strike"}
    ).merge(close[["slug", "oracle_ts", "price"]].rename(columns={"oracle_ts": "end", "price": "final"}), on="slug")
    frame = frame[(frame["start"] >= LABEL_START) & (frame["end"] <= LABEL_END)].copy()
    frame["up"] = frame["final"] >= frame["strike"]
    frame["start_s"] = epoch(frame["start"])
    frame["end_s"] = epoch(frame["end"])
    frame["split"] = np.where(frame["end"] <= TRAIN_END, "train", np.where(frame["start"] >= TEST_START, "test", "gap"))
    return frame.reset_index(drop=True)


def oracle_points(windows: pd.DataFrame) -> pd.DataFrame:
    points = pd.concat(
        [
            windows[["start_s", "strike"]].set_axis(["sec", "price"], axis=1),
            windows[["end_s", "final"]].set_axis(["sec", "price"], axis=1),
        ]
    )
    points = points.groupby("sec", as_index=False)["price"].mean()
    split_at = int(TRAIN_END.timestamp())
    points["split"] = np.where(points["sec"] < split_at, "train", "test")
    return points


# ---------------------------------------------------------------- formula


class Anchor:
    """Running bias of the formula, from oracle prints already known at a given clock.

    The residual at stamp S is usable from clock S + ANCHOR_DELAY, since the
    oracle value at S (the price to beat of the window starting at S) is
    streamed live.
    """

    def __init__(self, stamps: np.ndarray, residuals: np.ndarray, halflife: float | None) -> None:
        order = np.argsort(stamps)
        self.stamps = stamps[order]
        if halflife is None:
            self.values = np.zeros(len(order))
        else:
            self.values = pd.Series(residuals[order]).ewm(halflife=halflife).mean().to_numpy()

    def at(self, clock: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.stamps, np.asarray(clock) - ANCHOR_DELAY, side="right") - 1
        out = np.zeros(len(idx))
        out[idx >= 0] = self.values[idx[idx >= 0]]
        return out


def fit_oracle_month(grid: PriceGrid, raw_grid: PriceGrid, points: pd.DataFrame) -> dict:
    train = points[points["split"] == "train"]
    test = points[points["split"] == "test"]
    report = {"n_train": len(train), "n_test": len(test), "candidates": {}}

    def run(name: str, source: PriceGrid, names: list[str]) -> dict:
        formula = fit_formula(source, train["sec"].to_numpy(), train["price"].to_numpy(), names)
        entry = {"formula": formula}
        for split, frame in (("train", train), ("test", test)):
            pred = predict(source, formula, frame["sec"].to_numpy())
            keep = np.isfinite(pred)
            residual = pred[keep] - frame["price"].to_numpy()[keep]
            entry[split] = {
                "n": int(keep.sum()),
                "mae": float(np.mean(np.abs(residual))),
                "mse": float(np.mean(residual**2)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "p95_abs": float(np.quantile(np.abs(residual), 0.95)),
                "bias": float(np.mean(residual)),
            }
        report["candidates"][name] = entry
        print(f"{name}: lags {formula['lags']} test MAE {entry['test']['mae']:.3f} RMSE {entry['test']['rmse']:.3f}", flush=True)
        return formula

    for venue in grid.venues:
        run(f"{venue} seul", grid, [venue])
    if {"coinbase", "kraken"} <= set(grid.venues):
        run("USD seulement (coinbase + kraken)", grid, ["coinbase", "kraken"])
    run("toutes venues, USDT non converti", raw_grid, list(grid.venues))
    main = run("toutes venues, USDT converti en USD", grid, list(grid.venues))
    report["main"] = "toutes venues, USDT converti en USD"

    curve = []
    for lag in range(-12, 4):
        formula = fit_formula(grid, train["sec"].to_numpy(), train["price"].to_numpy(), list(grid.venues), range(lag, lag + 1))
        row = {"shared_lag": lag}
        for split, frame in (("train", train), ("test", test)):
            pred = predict(grid, formula, frame["sec"].to_numpy())
            keep = np.isfinite(pred)
            residual = pred[keep] - frame["price"].to_numpy()[keep]
            row[f"{split}_mae"] = float(np.mean(np.abs(residual)))
            row[f"{split}_rmse"] = float(np.sqrt(np.mean(residual**2)))
        curve.append(row)
    pd.DataFrame(curve).to_csv(OUT / "lag_curve.csv", index=False)
    report["lag_curve"] = curve

    pred = predict(grid, main, points["sec"].to_numpy())
    residual = pred - points["price"].to_numpy()
    ok = np.isfinite(residual)
    stamps, residuals = points["sec"].to_numpy()[ok], residual[ok]
    is_train = points["split"].to_numpy()[ok] == "train"
    anchor_rows = []
    for halflife in ANCHOR_HALFLIVES:
        correction = Anchor(stamps, residuals, halflife).at(stamps)
        corrected = residuals - correction
        anchor_rows.append(
            {
                "halflife_windows": halflife,
                "train_mae": float(np.nanmean(np.abs(corrected[is_train]))),
                "test_mae": float(np.nanmean(np.abs(corrected[~is_train]))),
                "test_rmse": float(np.sqrt(np.nanmean(corrected[~is_train] ** 2))),
            }
        )
    report["anchor"] = anchor_rows
    best = min(anchor_rows, key=lambda row: row["train_mae"])
    report["anchor_halflife"] = best["halflife_windows"]
    report["anchor_points"] = {"sec": stamps.tolist(), "residual": residuals.tolist()}
    print(f"anchor halflife {best['halflife_windows']}: test MAE {best['test_mae']:.3f}", flush=True)

    weekly = []
    frame = points.assign(residual=pred - points["price"]).dropna()
    frame["day"] = pd.to_datetime(frame["sec"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    for day, group in frame.groupby("day"):
        weekly.append({"day": day, "n": len(group), "mae": float(group["residual"].abs().mean()), "bias": float(group["residual"].mean())})
    report["daily"] = weekly
    return report


# ---------------------------------------------------------------- probability


def checkpoints(windows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for timeframe, group in windows.groupby("timeframe"):
        duration = DURATION[timeframe]
        taus = np.array(list(range(duration - 30, 30, -10)) + list(range(30, -51, -1)), dtype="int64")
        repeated = group.loc[group.index.repeat(len(taus))].copy()
        repeated["tau"] = np.tile(taus, len(group))
        parts.append(repeated)
    frame = pd.concat(parts, ignore_index=True)
    frame["clock"] = frame["end_s"] - frame["tau"]
    return frame


def add_model(frame: pd.DataFrame, grid: PriceGrid, formula: dict, vol: np.ndarray, anchor: Anchor) -> pd.DataFrame:
    info_clock = frame["clock"].to_numpy() - INFO_DELAY
    estimate, spread = estimate_at(grid, formula, frame["end_s"].to_numpy(), info_clock)
    frame = frame.copy()
    frame["estimate_raw"] = estimate
    # Only prints up to the window start (its price to beat) feed the anchor, never the final print.
    anchor_clock = np.minimum(frame["clock"].to_numpy(), frame["start_s"].to_numpy() + ANCHOR_DELAY)
    frame["estimate"] = estimate - anchor.at(anchor_clock)
    frame["spread"] = spread
    frame["sigma"] = vol_at(vol, info_clock)
    return frame[np.isfinite(frame["estimate"]) & np.isfinite(frame["sigma"])].reset_index(drop=True)


# ---------------------------------------------------------------- market


def fee(price: np.ndarray) -> np.ndarray:
    return np.round(FEE_RATE * price * (1 - price), 5)


def attach_market(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (timeframe, slug), group in frame.groupby(["timeframe", "slug"], sort=False):
        fills = load_market(timeframe, slug)
        group = group.copy()
        clock = group["clock"].to_numpy()
        for column in ("market_up", "quote_age", "ask_up", "size_up", "ask_down", "size_down"):
            group[column] = np.nan
        if fills is None or fills.empty:
            parts.append(group)
            continue
        fills = fills.sort_values("sec", kind="stable")
        sec = fills["sec"].to_numpy(dtype="int64")
        is_up = (fills["outcome"] == "Up").to_numpy()
        implied = np.where(is_up, fills["price"].to_numpy(), 1 - fills["price"].to_numpy())
        last = np.searchsorted(sec, clock, side="left") - 1
        ok = last >= 0
        group.loc[ok, "market_up"] = implied[last[ok]]
        group.loc[ok, "quote_age"] = clock[ok] - sec[last[ok]]
        buys = (fills["side"] == "BUY").to_numpy()
        for outcome, mask in (("up", buys & is_up), ("down", buys & ~is_up)):
            buy_sec = sec[mask]
            buy_price = fills["price"].to_numpy()[mask]
            buy_size = fills["size"].to_numpy()[mask]
            nxt = np.searchsorted(buy_sec, clock + 1, side="left")
            inside = nxt < len(buy_sec)
            inside[inside] = buy_sec[nxt[inside]] <= clock[inside] + FILL_HORIZON
            group.loc[inside, f"ask_{outcome}"] = buy_price[nxt[inside]]
            group.loc[inside, f"size_{outcome}"] = buy_size[nxt[inside]]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _logit(values: np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(values, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def blend_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(frame)), _logit(frame["market_up"].to_numpy(), 0.005), _logit(frame["p_up"].to_numpy(), 1e-4)]
    )


def fit_blend(train: pd.DataFrame) -> dict:
    """Per horizon bucket: logit P = a + b * logit(market) + c * logit(model)."""
    from scipy.optimize import minimize

    weights = {}
    rows = train[(train["quote_age"] <= MAX_QUOTE_AGE) & train["market_up"].notna()]
    for name, lo, hi in BUCKETS:
        part = rows[(rows["tau"] > lo) & (rows["tau"] <= hi)]
        x = blend_features(part)
        y = part["up"].astype(float).to_numpy()

        def nll(theta: np.ndarray) -> float:
            z = x @ theta
            return float(np.mean(np.logaddexp(0, z) - y * z))

        result = minimize(nll, np.array([0.0, 1.0, 0.0]), method="BFGS")
        weights[name] = {"a": float(result.x[0]), "b_market": float(result.x[1]), "c_model": float(result.x[2]), "n_train": len(part)}
    return weights


def apply_blend(frame: pd.DataFrame, weights: dict) -> np.ndarray:
    out = np.full(len(frame), np.nan)
    usable = ((frame["quote_age"] <= MAX_QUOTE_AGE) & frame["market_up"].notna()).to_numpy()
    for name, lo, hi in BUCKETS:
        mask = usable & ((frame["tau"] > lo) & (frame["tau"] <= hi)).to_numpy()
        w = weights[name]
        z = blend_features(frame[mask]) @ np.array([w["a"], w["b_market"], w["c_model"]])
        out[mask] = 1 / (1 + np.exp(-z))
    return out


def expiry_pie(windows: pd.DataFrame) -> list[dict]:
    """Taker BUY fills of the eventual winner near the end, and what they earned after fees."""
    edges = [(-120, -60), (-60, -30), (-30, -10), (-10, -5), (-5, 0), (0, 10), (10, 30), (30, 60), (60, 600)]
    rows = []
    for window in windows.itertuples(index=False):
        fills = load_market(window.timeframe, window.slug)
        if fills is None or fills.empty:
            continue
        winner = "Up" if window.up else "Down"
        buys = fills[(fills["side"] == "BUY") & (fills["outcome"] == winner)]
        rel = buys["sec"].to_numpy() - window.end_s
        price = buys["price"].to_numpy()
        size = buys["size"].to_numpy()
        for lo, hi in edges:
            mask = (rel >= lo) & (rel < hi) & (price <= 0.98)
            if mask.any():
                gain = size[mask] * (1 - price[mask] - fee(price[mask]))
                rows.append({"timeframe": window.timeframe, "slug": window.slug, "from_end": f"[{lo},{hi})", "lo": lo,
                             "fills": int(mask.sum()), "shares": float(size[mask].sum()), "gain_usd": float(gain.sum()),
                             "mean_price": float(np.average(price[mask], weights=size[mask]))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    summary = frame.groupby(["from_end", "lo", "timeframe"], as_index=False).agg(
        windows=("slug", "nunique"), fills=("fills", "sum"), shares=("shares", "sum"), gain_usd=("gain_usd", "sum")
    ).sort_values(["timeframe", "lo"])
    summary["gain_per_day_usd"] = summary["gain_usd"] / ((TEST_END - TEST_START).days)
    return summary.drop(columns="lo").to_dict(orient="records")


def calibration(frame: pd.DataFrame) -> dict:
    rows = frame[(frame["quote_age"] <= MAX_QUOTE_AGE) & frame["market_up"].notna()].copy()
    y = rows["up"].astype(float).to_numpy()
    out = {"n": len(rows), "by_bucket": {}, "bins": []}
    for name, lo, hi in BUCKETS:
        part = rows[(rows["tau"] > lo) & (rows["tau"] <= hi)]
        if part.empty:
            continue
        yy = part["up"].astype(float).to_numpy()
        model = part["p_up"].to_numpy()
        market = part["market_up"].clip(0.001, 0.999).to_numpy()
        gap = model - market
        beta = float(np.dot(gap, yy - market) / np.dot(gap, gap)) if np.dot(gap, gap) > 0 else np.nan
        out["by_bucket"][name] = {
            "n": len(part),
            "brier_model": float(np.mean((model - yy) ** 2)),
            "brier_market": float(np.mean((market - yy) ** 2)),
            "brier_blend": float(np.nanmean((part["p_blend"].to_numpy() - yy) ** 2)) if "p_blend" in part else None,
            "logloss_model": float(-np.mean(yy * np.log(np.clip(model, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - model, 1e-6, 1)))),
            "logloss_market": float(-np.mean(yy * np.log(market) + (1 - yy) * np.log(1 - market))),
            "mean_abs_gap": float(np.mean(np.abs(gap))),
            "share_gap_over_10c": float(np.mean(np.abs(gap) > 0.10)),
            "beta_outcome_on_gap": beta,
        }
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        for source in ("p_up", "market_up"):
            values = rows[source].to_numpy()
            mask = (values >= lo) & ((values < hi) | (hi == 1))
            if mask.sum():
                out["bins"].append(
                    {"source": "modele" if source == "p_up" else "marche", "bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "mean_p": float(values[mask].mean()), "freq_up": float(y[mask].mean())}
                )
    return out


def backtest(frame: pd.DataFrame, min_edge: float, prob_column: str = "p_up") -> tuple[dict, pd.DataFrame]:
    data = frame[frame[prob_column].notna()].copy()
    data["p_trade"] = data[prob_column]
    for outcome, prob in (("up", data["p_trade"]), ("down", 1 - data["p_trade"])):
        ask = data[f"ask_{outcome}"]
        data[f"edge_{outcome}"] = prob - ask - fee(ask.fillna(0).to_numpy())
        data.loc[(ask < 0.02) | (ask > 0.98) | ask.isna(), f"edge_{outcome}"] = np.nan
    data["side"] = np.where(data["edge_up"].fillna(-9) >= data["edge_down"].fillna(-9), "up", "down")
    data["edge"] = np.where(data["side"] == "up", data["edge_up"], data["edge_down"])
    data = data[data["edge"] >= min_edge]
    summary = {}
    trades = []
    for name, lo, hi in BUCKETS:
        part = data[(data["tau"] > lo) & (data["tau"] <= hi)].sort_values("clock")
        first = part.groupby("slug", sort=False).head(1).copy()
        if first.empty:
            summary[name] = {"trades": 0}
            continue
        is_up = first["side"] == "up"
        first["entry"] = np.where(is_up, first["ask_up"], first["ask_down"])
        first["shares"] = np.minimum(MAX_SHARES, np.where(is_up, first["size_up"], first["size_down"]))
        first["win"] = np.where(is_up, first["up"], ~first["up"].astype(bool)).astype(float)
        first["fee"] = fee(first["entry"].to_numpy())
        first["pnl_per_share"] = first["win"] - first["entry"] - first["fee"]
        first["pnl"] = first["pnl_per_share"] * first["shares"]
        first["bucket"] = name
        stake = float((first["entry"] * first["shares"]).sum())
        summary[name] = {
            "trades": len(first),
            "windows_in_test": int(frame.loc[(frame["tau"] > lo) & (frame["tau"] <= hi), "slug"].nunique()),
            "win_rate": float(first["win"].mean()),
            "mean_entry": float(first["entry"].mean()),
            "mean_model_prob": float(np.where(is_up, first["p_trade"], 1 - first["p_trade"]).mean()),
            "pnl_per_share_mean": float(first["pnl_per_share"].mean()),
            "shares": float(first["shares"].sum()),
            "stake_usd": stake,
            "fees_usd": float((first["fee"] * first["shares"]).sum()),
            "pnl_usd": float(first["pnl"].sum()),
            "roi": float(first["pnl"].sum() / stake) if stake else np.nan,
            "pnl_5m": float(first.loc[first["timeframe"] == "5m", "pnl"].sum()),
            "pnl_15m": float(first.loc[first["timeframe"] == "15m", "pnl"].sum()),
        }
        trades.append(first)
    return summary, (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())


def main(venues: tuple[str, ...] = VENUES) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    points = oracle_points(windows)
    print(f"windows {len(windows)} oracle points {len(points)}", flush=True)
    grid = PriceGrid(convert_usdt=True, venues=venues)
    raw_grid = PriceGrid(convert_usdt=False, venues=venues)
    oracle_report = fit_oracle_month(grid, raw_grid, points)
    known = oracle_report.pop("anchor_points")
    (OUT / "oracle_fit.json").write_text(json.dumps(oracle_report, indent=2), encoding="utf-8")
    del raw_grid
    formula = oracle_report["candidates"][oracle_report["main"]]["formula"]
    anchor = Anchor(np.array(known["sec"]), np.array(known["residual"]), oracle_report["anchor_halflife"])

    vol = spot_vol(grid, formula)
    frame = add_model(checkpoints(windows[windows["split"] != "gap"]), grid, formula, vol, anchor)
    train = frame[frame["split"] == "train"]
    c0, c1 = fit_probability(
        train["estimate"].to_numpy(), train["strike"].to_numpy(), train["spread"].to_numpy(), train["sigma"].to_numpy(), train["up"].to_numpy()
    )
    frame["p_up"] = probability(frame["estimate"], frame["strike"], frame["spread"], frame["sigma"], c0, c1)
    prob_report = {"c0": c0, "c1": c1, "model_error_usd": float(np.sqrt(c0)), "vol_multiplier": float(np.sqrt(c1)), "info_delay_s": INFO_DELAY}
    for split in ("train", "test"):
        part = frame[frame["split"] == split]
        y = part["up"].astype(float)
        prob_report[f"{split}_brier"] = float(np.mean((part["p_up"] - y) ** 2))
        prob_report[f"{split}_rows"] = len(part)
    print(f"probability model c0 {c0:.3f} c1 {c1:.3f}", flush=True)

    prob_report["train_calibration_by_tau"] = (
        train.assign(p_up=frame.loc[frame["split"] == "train", "p_up"].to_numpy(), tau_bucket=pd.cut(train["tau"], [-60, 0, 5, 30, 60, 150, 400, 1000]).astype(str))
        .groupby("tau_bucket")
        .agg(n=("up", "size"), mean_p=("p_up", "mean"), freq_up=("up", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    market_train = attach_market(frame[frame["split"] == "train"])
    blend = fit_blend(market_train)
    prob_report["blend"] = blend
    del market_train
    test = attach_market(frame[frame["split"] == "test"])
    test["p_blend"] = apply_blend(test, blend)
    test.to_parquet(OUT / "test_checkpoints.parquet", index=False)
    prob_report["calibration_test"] = calibration(test)
    (OUT / "prob_model.json").write_text(json.dumps(prob_report, indent=2), encoding="utf-8")

    results = {"model": {}, "blend": {}}
    for column, key in (("p_up", "model"), ("p_blend", "blend")):
        for edge in EDGES:
            summary, trades = backtest(test, edge, column)
            results[key][f"edge>={edge:.2f}"] = summary
            if edge == MAIN_EDGE and not trades.empty:
                trades.to_parquet(OUT / f"trades_{key}.parquet", index=False)
    results["expiry_pie_test"] = expiry_pie(windows[windows["split"] == "test"])
    results["rules"] = {
        "info_delay_s": INFO_DELAY, "fill_horizon_s": FILL_HORIZON, "max_shares": MAX_SHARES,
        "fee": "0.07 * p * (1 - p) per share", "entry_range": [0.02, 0.98], "one_trade_per_window_per_bucket": True,
    }
    (OUT / "backtest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: results[key][f"edge>={MAIN_EDGE:.2f}"] for key in ("model", "blend")}, indent=1), flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        OUT = ROOT / "data" / "month_debug"
        main(tuple(sys.argv[1:]))
    else:
        main()
