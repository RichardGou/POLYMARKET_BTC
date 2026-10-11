"""Fit a delayed, weighted average of exchange TWAPs to the Gamma oracle.

The label at time T is the Chainlink 60-second print stored by Gamma.
The model is

    oracle(T) = base + sum_i alpha_i * TWAP_i(T + delta)

with alpha_i >= 0 and sum alpha_i = 1. delta is a whole number of seconds.
A negative delta means the exchange average that matches the oracle stamped
at T was already available before T.

Coefficients and delta are chosen on the first two days only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from datetime import timedelta

from utils.providers import LABEL_END, LABEL_START, PROVIDERS

TRAIN_END = LABEL_START + timedelta(days=2)
DELTAS = range(-5, 6)
WINDOWS = ("left_open", "right_closed")
FFILL_LIMIT = 120


def load_prices() -> dict[str, pd.Series]:
    series = {}
    for path in sorted(PROVIDERS.glob("*/btc_1s.parquet")):
        frame = pd.read_parquet(path)
        stamp = pd.to_datetime(frame["ts"], utc=True).astype("int64") // 10**9
        values = pd.Series(frame["price"].to_numpy(), index=stamp.to_numpy(), dtype="float64")
        values = values[~values.index.duplicated(keep="last")].sort_index()
        series[path.parent.name] = values
    if not series:
        raise FileNotFoundError(f"no provider prices under {PROVIDERS}")
    return series


def load_labels(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    opening = pd.read_parquet(root / "data" / "price_oracle_opening.parquet")
    close = pd.read_parquet(root / "data" / "price_oracle_close.parquet")
    for frame in (opening, close):
        frame["oracle_ts"] = pd.to_datetime(frame["oracle_ts"], utc=True)
    return opening, close


def _dense(series: pd.Series) -> pd.Series:
    start = int(series.index.min())
    end = int(series.index.max())
    index = np.arange(start, end + 1)
    dense = series.reindex(index).ffill(limit=FFILL_LIMIT)
    return dense


def twap_series(series: pd.Series) -> dict[str, pd.Series]:
    dense = _dense(series)
    # rolling(60) at second S is the mean of S-59..S.
    right_closed = dense.rolling(60, min_periods=60).mean()
    # mean of S-60..S-1, stored at S.
    left_open = right_closed.shift(1)
    return {"right_closed": right_closed, "left_open": left_open}


def oracle_points(opening: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    stacked = pd.concat(
        [
            opening[["oracle_ts", "price", "timeframe"]],
            close[["oracle_ts", "price", "timeframe"]],
        ],
        ignore_index=True,
    )
    stacked = stacked[(stacked["oracle_ts"] >= LABEL_START) & (stacked["oracle_ts"] < LABEL_END)]
    grouped = stacked.groupby("oracle_ts")["price"].agg(["mean", "min", "max", "count"])
    grouped["spread"] = grouped["max"] - grouped["min"]
    return grouped.reset_index()


def design(points: pd.DataFrame, twaps: dict[str, dict[str, pd.Series]], window: str, delta: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = sorted(twaps)
    stamps = (points["oracle_ts"].astype("int64") // 10**9).to_numpy() + delta
    columns = []
    keep = np.ones(len(stamps), dtype=bool)
    for name in names:
        values = twaps[name][window]
        looked = values.reindex(stamps).to_numpy(dtype="float64")
        columns.append(looked)
        keep &= np.isfinite(looked)
    matrix = np.column_stack(columns)[keep]
    target = points["mean"].to_numpy(dtype="float64")[keep]
    return matrix, target, names


def fit_simplex(matrix: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    width = matrix.shape[1]

    def objective(theta: np.ndarray) -> float:
        prediction = theta[0] + matrix @ theta[1:]
        error = prediction - target
        return float(np.dot(error, error) / len(error))

    start = np.zeros(width + 1)
    start[1:] = 1.0 / width
    start[0] = float(np.median(target - matrix @ start[1:]))
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(None, None)] + [(0.0, 1.0)] * width,
        constraints={"type": "eq", "fun": lambda theta: np.sum(theta[1:]) - 1.0},
        options={"maxiter": 400, "ftol": 1e-14},
    )
    theta = result.x if result.success else start
    theta = theta.copy()
    theta[1:] = np.clip(theta[1:], 0, None)
    total = theta[1:].sum()
    if total <= 0:
        theta[1:] = 1.0 / width
    else:
        theta[1:] = theta[1:] / total
    # Refit the level once the weights are fixed. It does not change the direction.
    prediction_without_base = matrix @ theta[1:]
    theta[0] = float(np.mean(target - prediction_without_base))
    return float(theta[0]), theta[1:]


def errors(matrix: np.ndarray, target: np.ndarray, base: float, alpha: np.ndarray) -> dict[str, float]:
    prediction = base + matrix @ alpha
    residual = prediction - target
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(residual**2)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def sign_table(
    opening: pd.DataFrame,
    close: pd.DataFrame,
    twaps: dict[str, dict[str, pd.Series]],
    names: list[str],
    window: str,
    delta: int,
    base: float,
    alpha: np.ndarray,
    split: str,
) -> dict[str, float]:
    merged = opening.merge(close, on=["timeframe", "slug"], suffixes=("_open", "_close"))
    merged = merged[(merged["oracle_ts_close"] >= LABEL_START) & (merged["oracle_ts_close"] < LABEL_END)]
    if split == "train":
        merged = merged[merged["oracle_ts_close"] < TRAIN_END]
    else:
        merged = merged[merged["oracle_ts_close"] >= TRAIN_END]
    out: dict[str, float] = {}
    for timeframe, part in merged.groupby("timeframe"):
        start = (part["oracle_ts_open"].astype("int64") // 10**9).to_numpy() + delta
        end = (part["oracle_ts_close"].astype("int64") // 10**9).to_numpy() + delta
        predicted_open = _predict_at(twaps, names, window, start, base, alpha)
        predicted_close = _predict_at(twaps, names, window, end, base, alpha)
        actual_up = part["price_close"].to_numpy() >= part["price_open"].to_numpy()
        valid = np.isfinite(predicted_open) & np.isfinite(predicted_close)
        if not valid.any():
            out[f"{timeframe}_sign_hit"] = float("nan")
            out[f"{timeframe}_n"] = 0
            continue
        predicted_up = predicted_close[valid] >= predicted_open[valid]
        move = np.abs(part["price_close"].to_numpy() - part["price_open"].to_numpy())[valid]
        hit = predicted_up == actual_up[valid]
        small = move < 10
        out[f"{timeframe}_sign_hit"] = float(hit.mean())
        out[f"{timeframe}_n"] = int(hit.size)
        out[f"{timeframe}_sign_hit_under_10usd"] = float(hit[small].mean()) if small.any() else float("nan")
        out[f"{timeframe}_n_under_10usd"] = int(small.sum())
    return out


def _predict_at(twaps, names, window, stamps, base, alpha) -> np.ndarray:
    columns = [twaps[name][window].reindex(stamps).to_numpy(dtype="float64") for name in names]
    matrix = np.column_stack(columns)
    prediction = np.full(len(stamps), np.nan)
    valid = np.isfinite(matrix).all(axis=1)
    prediction[valid] = base + matrix[valid] @ alpha
    return prediction


def run_fit(root: Path) -> dict:
    prices = load_prices()
    twaps = {name: twap_series(series) for name, series in prices.items()}
    opening, close = load_labels(root)
    points = oracle_points(opening, close)
    spread = float(points["spread"].max()) if len(points) else float("nan")
    train_mask = points["oracle_ts"] < TRAIN_END
    test_mask = ~train_mask

    grid = []
    best = None
    for window in WINDOWS:
        for delta in DELTAS:
            matrix_train, target_train, names = design(points.loc[train_mask], twaps, window, delta)
            if len(target_train) < 50:
                continue
            base, alpha = fit_simplex(matrix_train, target_train)
            train_error = errors(matrix_train, target_train, base, alpha)
            matrix_test, target_test, _names = design(points.loc[test_mask], twaps, window, delta)
            test_error = errors(matrix_test, target_test, base, alpha)
            row = {
                "window": window,
                "delta_seconds": int(delta),
                "base_usd": base,
                "alpha": {name: float(weight) for name, weight in zip(names, alpha)},
                "train": train_error,
                "test": test_error,
            }
            grid.append(row)
            if best is None or train_error["mse"] < best["train"]["mse"]:
                best = row

    if best is None:
        raise RuntimeError("no fittable rows")

    names = sorted(best["alpha"])
    alpha = np.array([best["alpha"][name] for name in names])
    best["train_sign"] = sign_table(
        opening, close, twaps, names, best["window"], best["delta_seconds"], best["base_usd"], alpha, "train"
    )
    best["test_sign"] = sign_table(
        opening, close, twaps, names, best["window"], best["delta_seconds"], best["base_usd"], alpha, "test"
    )
    summary = {
        "label_start": LABEL_START.isoformat(),
        "label_end": LABEL_END.isoformat(),
        "train_end": str(TRAIN_END),
        "providers": {
            name: json.loads((PROVIDERS / name / "meta.json").read_text(encoding="utf-8")) for name in names
        },
        "max_disagreement_between_5m_and_15m_usd": spread,
        "n_oracle_timestamps": int(len(points)),
        "selected_on": "train MSE",
        "selected": best,
        "grid": [
            {
                "window": row["window"],
                "delta_seconds": row["delta_seconds"],
                "train_mae": row["train"]["mae"],
                "train_mse": row["train"]["mse"],
                "test_mae": row["test"]["mae"],
                "test_mse": row["test"]["mse"],
            }
            for row in grid
        ],
    }
    out = root / "data" / "fit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["grid"]).to_csv(out / "delta_grid.csv", index=False)
    return summary
