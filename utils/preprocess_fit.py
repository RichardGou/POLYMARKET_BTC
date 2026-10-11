"""Preprocessing trials for the oracle regression.

Every choice (window, aggregator, lag, venue list) is picked on the first
two days. The third day is only scored. A trial that drops more than 2% of
the training timestamps is discarded, so a lower error cannot come from
throwing away the difficult minutes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils.fit_oracle import (
    TRAIN_END,
    errors,
    fit_simplex,
    load_labels,
    load_prices,
    oracle_points,
)
from utils.providers import LABEL_END, LABEL_START, PRICE_END, PRICE_START, PROVIDERS, _get, _session

ROOT = Path(__file__).resolve().parents[1]


def download_usdt_usd() -> pd.Series:
    client = _session()
    start = PRICE_START
    stop = PRICE_END
    rows: list[tuple[int, float]] = []
    cursor = start
    while cursor < stop:
        chunk_end = min(cursor + timedelta(minutes=300), stop)
        response = _get(
            client,
            "https://api.exchange.coinbase.com/products/USDT-USD/candles",
            params={
                "granularity": 60,
                "start": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        for candle in response.json():
            rows.append((int(candle[0]), float(candle[4])))
        cursor = chunk_end
    frame = pd.DataFrame(rows, columns=["sec", "usdt_usd"]).drop_duplicates("sec")
    # The candle close is known at the end of its minute, not at the open.
    frame["sec"] = frame["sec"] + 60
    path = PROVIDERS / "coinbase" / "usdt_usd_1m.parquet"
    frame.to_parquet(path, index=False)
    return pd.Series(frame["usdt_usd"].to_numpy(), index=frame["sec"].to_numpy()).sort_index()


def load_usdt_usd() -> pd.Series:
    path = PROVIDERS / "coinbase" / "usdt_usd_1m.parquet"
    if not path.exists():
        return download_usdt_usd()
    frame = pd.read_parquet(path)
    return pd.Series(frame["usdt_usd"].to_numpy(), index=frame["sec"].to_numpy()).sort_index()


def dense(series: pd.Series, limit: int) -> pd.Series:
    index = np.arange(int(series.index.min()), int(series.index.max()) + 1)
    filled = series.reindex(index)
    if limit > 0:
        filled = filled.ffill(limit=limit)
    return filled


def rolling_stat(values: np.ndarray, length: int, kind: str) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, length)
    valid = np.isfinite(windows).all(axis=1)
    if kind == "mean":
        stat = windows.mean(axis=1)
    elif kind == "trim":
        cut = max(1, length // 20)
        ordered = np.partition(windows, (cut, length - 1 - cut), axis=1)
        stat = ordered[:, cut : length - cut].mean(axis=1)
    elif kind == "ewma15":
        age = np.arange(length - 1, -1, -1)
        weights = np.exp(-np.log(2) * age / 15.0)
        weights = weights / weights.sum()
        stat = windows @ weights
    else:
        raise ValueError(kind)
    stat = stat.astype("float64")
    stat[~valid] = np.nan
    out = np.full(len(values), np.nan)
    out[length - 1 :] = stat
    return out


def build_features(prices: dict[str, pd.Series], usdt: pd.Series, limit: int, length: int, kind: str) -> dict[str, pd.Series]:
    usdt_dense = dense(usdt, limit=180)
    features = {}
    for name, series in prices.items():
        level = dense(series, limit)
        if name in {"binance", "okx"}:
            rate = usdt_dense.reindex(level.index).to_numpy()
            level = pd.Series(level.to_numpy() * rate, index=level.index)
        stat = rolling_stat(level.to_numpy(dtype="float64"), length, kind)
        features[name] = pd.Series(stat, index=level.index)
    return features


def stamps_of(points: pd.DataFrame) -> np.ndarray:
    return (points["oracle_ts"].astype("int64") // 10**9).to_numpy()


def matrix_at(features: dict[str, pd.Series], names: list[str], stamps: np.ndarray, deltas: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    keep = np.ones(len(stamps), dtype=bool)
    for name in names:
        looked = features[name].reindex(stamps + deltas[name]).to_numpy(dtype="float64")
        columns.append(looked)
        keep &= np.isfinite(looked)
    return np.column_stack(columns), keep


def score_split(features, names, deltas, train_stamps, train_y, test_stamps, test_y) -> dict | None:
    train_x, train_keep = matrix_at(features, names, train_stamps, deltas)
    if train_keep.mean() < 0.98:
        return None
    base, alpha = fit_simplex(train_x[train_keep], train_y[train_keep])
    test_x, test_keep = matrix_at(features, names, test_stamps, deltas)
    if test_keep.sum() < 30:
        return None
    train_error = errors(train_x[train_keep], train_y[train_keep], base, alpha)
    test_error = errors(test_x[test_keep], test_y[test_keep], base, alpha)
    return {
        "base_usd": base,
        "alpha": {name: float(weight) for name, weight in zip(names, alpha)},
        "train": train_error,
        "test": test_error,
        "train_coverage": float(train_keep.mean()),
        "test_coverage": float(test_keep.mean()),
    }


def run() -> dict:
    prices = load_prices()
    usdt = load_usdt_usd()
    opening, close = load_labels(ROOT)
    points = oracle_points(opening, close)
    train = points[points["oracle_ts"] < TRAIN_END].reset_index(drop=True)
    test = points[points["oracle_ts"] >= TRAIN_END].reset_index(drop=True)
    train_stamps, test_stamps = stamps_of(train), stamps_of(test)
    train_y, test_y = train["mean"].to_numpy(), test["mean"].to_numpy()

    usd_names = ["coinbase", "kraken"]
    all_names = ["binance", "coinbase", "kraken", "okx"]
    trials = []

    # Shared lag, both USD books, several windows and aggregators.
    for limit in (15, 120):
        for length in (45, 60, 75):
            for kind in ("mean", "trim", "ewma15"):
                features = build_features(prices, usdt, limit, length, kind)
                for delta in (-8, -6, -5, -4, -3, -2, -1, 0):
                    scored = score_split(
                        features,
                        usd_names,
                        {name: delta for name in usd_names},
                        train_stamps,
                        train_y,
                        test_stamps,
                        test_y,
                    )
                    if scored is None:
                        continue
                    scored.update(
                        {
                            "venues": usd_names,
                            "ffill_seconds": limit,
                            "window_seconds": length,
                            "aggregator": kind,
                            "deltas": {name: delta for name in usd_names},
                        }
                    )
                    trials.append(scored)

    best_shared = min(trials, key=lambda row: row["train"]["mse"])
    # Separate lags around the shared winner, same window and aggregator.
    features = build_features(
        prices,
        usdt,
        best_shared["ffill_seconds"],
        best_shared["window_seconds"],
        best_shared["aggregator"],
    )
    for delta_c in range(-8, 2):
        for delta_k in range(-8, 2):
            deltas = {"coinbase": delta_c, "kraken": delta_k}
            scored = score_split(features, usd_names, deltas, train_stamps, train_y, test_stamps, test_y)
            if scored is None:
                continue
            scored.update(
                {
                    "venues": usd_names,
                    "ffill_seconds": best_shared["ffill_seconds"],
                    "window_seconds": best_shared["window_seconds"],
                    "aggregator": best_shared["aggregator"],
                    "deltas": deltas,
                }
            )
            trials.append(scored)

    best_usd = min(
        (row for row in trials if row["venues"] == usd_names),
        key=lambda row: row["train"]["mse"],
    )
    # Bring the USDT books back after converting them with USDT/USD.
    for delta_usdt in range(-8, 2):
        deltas = {
            "coinbase": best_usd["deltas"]["coinbase"],
            "kraken": best_usd["deltas"]["kraken"],
            "binance": delta_usdt,
            "okx": delta_usdt,
        }
        scored = score_split(features, all_names, deltas, train_stamps, train_y, test_stamps, test_y)
        if scored is None:
            continue
        scored.update(
            {
                "venues": all_names,
                "ffill_seconds": best_usd["ffill_seconds"],
                "window_seconds": best_usd["window_seconds"],
                "aggregator": best_usd["aggregator"],
                "deltas": deltas,
                "usdt_converted": True,
            }
        )
        trials.append(scored)

    best = min(trials, key=lambda row: row["train"]["mse"])
    baseline = next(
        row
        for row in trials
        if row["venues"] == usd_names
        and row["aggregator"] == "mean"
        and row["window_seconds"] == 60
        and row["ffill_seconds"] == 120
        and row["deltas"] == {"coinbase": -5, "kraken": -5}
    )
    summary = {
        "baseline": _public(baseline),
        "best_on_train": _public(best),
        "n_trials": len(trials),
    }
    out = ROOT / "data" / "fit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat = []
    for row in trials:
        flat.append(
            {
                "aggregator": row["aggregator"],
                "window_seconds": row["window_seconds"],
                "ffill_seconds": row["ffill_seconds"],
                "n_venues": len(row["venues"]),
                "delta_coinbase": row["deltas"].get("coinbase"),
                "delta_kraken": row["deltas"].get("kraken"),
                "delta_usdt": row["deltas"].get("binance"),
                "train_mae": row["train"]["mae"],
                "train_mse": row["train"]["mse"],
                "test_mae": row["test"]["mae"],
                "test_mse": row["test"]["mse"],
            }
        )
    pd.DataFrame(flat).to_csv(out / "preprocess_grid.csv", index=False)
    return summary


def _public(row: dict) -> dict:
    return {
        "venues": row["venues"],
        "aggregator": row["aggregator"],
        "window_seconds": row["window_seconds"],
        "ffill_seconds": row["ffill_seconds"],
        "deltas": row["deltas"],
        "base_usd": row["base_usd"],
        "alpha": row["alpha"],
        "train": row["train"],
        "test": row["test"],
        "train_coverage": row["train_coverage"],
        "test_coverage": row["test_coverage"],
        "usdt_converted": row.get("usdt_converted", False),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
