"""Oracle estimate and Up probability at any second of a window.

Prices live on one second grid starting at FIRST_PRICE_DAY. The value of
second s is known at clock s + 1. USDT books are converted with USDT/USD.

Oracle formula, fitted on the training days:

    oracle(T) = base + sum_i alpha_i * mean(P_i[T + d_i - 60 : T + d_i])

At clock t, the seconds of each window that are not known yet are filled
with the last known price, so the estimate is a martingale guess of the
final print. The probability that Up wins is

    P(up) = Phi((E_t - K) / sqrt(c0 + c1 * sigma_t^2 * v_t))

with K the price to beat, sigma_t the recent one-second volatility of the
weighted spot, and v_t the variance a random walk would add to the
average over the seconds still unknown. c0 and c1 are fitted on the
training days by maximum likelihood.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from utils.exchanges import USDT_VENUES, VENUES, load_usdt_usd, load_venue
from utils.fit_oracle import fit_simplex
from utils.month import FIRST_PRICE_DAY, LAST_PRICE_DAY

FFILL_SECONDS = 120
VOL_SECONDS = 600
VOL_STEP = 5
CENTER = 80_000.0
GRID_START = int(datetime(FIRST_PRICE_DAY.year, FIRST_PRICE_DAY.month, FIRST_PRICE_DAY.day, tzinfo=timezone.utc).timestamp())
GRID_END = int(datetime(LAST_PRICE_DAY.year, LAST_PRICE_DAY.month, LAST_PRICE_DAY.day, tzinfo=timezone.utc).timestamp()) + 86400


class PriceGrid:
    def __init__(self, convert_usdt: bool = True, venues: tuple[str, ...] = VENUES) -> None:
        length = GRID_END - GRID_START
        rate = _dense(load_usdt_usd(), length, limit=600)
        self.venues = venues
        self.prices: dict[str, np.ndarray] = {}
        for venue in venues:
            values = _dense(load_venue(venue, FIRST_PRICE_DAY, LAST_PRICE_DAY), length, FFILL_SECONDS)
            if convert_usdt and venue in USDT_VENUES:
                values = values * rate
            self.prices[venue] = values
        self.sums = {}
        self.counts = {}
        for venue, values in self.prices.items():
            finite = np.isfinite(values)
            self.sums[venue] = np.concatenate([[0.0], np.cumsum(np.where(finite, values - CENTER, 0.0))])
            self.counts[venue] = np.concatenate([[0], np.cumsum(finite)])

    def index(self, seconds: np.ndarray) -> np.ndarray:
        return np.asarray(seconds, dtype="int64") - GRID_START

    def window_mean(self, venue: str, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        """Mean of seconds [start, stop), NaN unless every second has a price."""
        a = np.clip(self.index(start), 0, len(self.prices[venue]))
        b = np.clip(self.index(stop), 0, len(self.prices[venue]))
        total = self.sums[venue][b] - self.sums[venue][a]
        count = self.counts[venue][b] - self.counts[venue][a]
        width = b - a
        out = total / np.maximum(width, 1) + CENTER
        out[(count != width) | (width <= 0)] = np.nan
        return out

    def last_known(self, venue: str, clock: np.ndarray) -> np.ndarray:
        idx = self.index(clock) - 1
        values = self.prices[venue]
        ok = (idx >= 0) & (idx < len(values))
        out = np.full(len(idx), np.nan)
        out[ok] = values[idx[ok]]
        return out


def _dense(series: pd.Series, length: int, limit: int) -> np.ndarray:
    out = np.full(length, np.nan)
    idx = series.index.to_numpy(dtype="int64") - GRID_START
    keep = (idx >= 0) & (idx < length)
    out[idx[keep]] = series.to_numpy()[keep]
    return pd.Series(out).ffill(limit=limit).to_numpy()


def design(grid: PriceGrid, stamps: np.ndarray, lags: dict[str, int], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    for name in names:
        stop = stamps + lags[name]
        columns.append(grid.window_mean(name, stop - 60, stop))
    matrix = np.column_stack(columns)
    return matrix, np.isfinite(matrix).all(axis=1)


def score(matrix: np.ndarray, target: np.ndarray, base: float, alpha: np.ndarray) -> dict:
    residual = base + matrix @ alpha - target
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(residual**2)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "p95_abs": float(np.quantile(np.abs(residual), 0.95)),
    }


def fit_formula(
    grid: PriceGrid,
    train_stamps: np.ndarray,
    train_y: np.ndarray,
    names: list[str],
    lag_range: range = range(-10, 3),
) -> dict:
    def evaluate(lags: dict[str, int]) -> tuple[float, float, np.ndarray]:
        matrix, keep = design(grid, train_stamps, lags, names)
        if keep.mean() < 0.98:
            return np.inf, 0.0, np.zeros(len(names))
        base, alpha = fit_simplex(matrix[keep], train_y[keep])
        mse = score(matrix[keep], train_y[keep], base, alpha)["mse"]
        return mse, base, alpha

    best_shared = None
    for lag in lag_range:
        lags = {name: lag for name in names}
        mse, base, alpha = evaluate(lags)
        if best_shared is None or mse < best_shared[0]:
            best_shared = (mse, lags)
    lags = dict(best_shared[1])
    best_mse = best_shared[0]
    for _ in range(3):
        improved = False
        for name in names:
            for lag in lag_range:
                trial = dict(lags)
                trial[name] = lag
                mse, _base, _alpha = evaluate(trial)
                if mse < best_mse - 1e-9:
                    best_mse, lags, improved = mse, trial, True
        if not improved:
            break
    mse, base, alpha = evaluate(lags)
    return {"names": names, "lags": lags, "base": base, "alpha": {n: float(a) for n, a in zip(names, alpha)}}


def predict(grid: PriceGrid, formula: dict, stamps: np.ndarray) -> np.ndarray:
    names = formula["names"]
    matrix, keep = design(grid, stamps, formula["lags"], names)
    alpha = np.array([formula["alpha"][n] for n in names])
    out = formula["base"] + matrix @ alpha
    out[~keep] = np.nan
    return out


def estimate_at(grid: PriceGrid, formula: dict, target: np.ndarray, clock: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best guess of oracle(target) at clock, plus the random-walk spread of what is unknown."""
    estimate = np.full(len(target), formula["base"], dtype="float64")
    spread = np.zeros(len(target))
    valid = np.ones(len(target), dtype=bool)
    for name in formula["names"]:
        weight = formula["alpha"][name]
        stop = target + formula["lags"][name]
        start = stop - 60
        known_stop = np.clip(clock, start, stop)
        unknown = stop - known_stop
        gap = np.maximum(0, start - clock)
        last = grid.last_known(name, clock)
        known_count = known_stop - start
        known = grid.window_mean(name, start, known_stop)
        valid &= np.isfinite(last) & ((known_count == 0) | np.isfinite(known))
        mean = (np.where(known_count > 0, np.nan_to_num(known), 0.0) * known_count + last * unknown) / 60.0
        estimate = estimate + weight * mean
        n = unknown.astype("float64")
        variance = gap + n * (n + 1) * (2 * n + 1) / (6 * 3600.0)
        spread = spread + weight * np.sqrt(variance)
    estimate[~valid] = np.nan
    return estimate, spread


def spot_vol(grid: PriceGrid, formula: dict) -> np.ndarray:
    """Per-second volatility of the weighted spot, from VOL_STEP-second moves over the previous VOL_SECONDS."""
    combined = np.zeros(GRID_END - GRID_START)
    for name in formula["names"]:
        combined = combined + formula["alpha"][name] * grid.prices[name]
    step = np.full(len(combined), np.nan)
    step[VOL_STEP:] = combined[VOL_STEP:] - combined[:-VOL_STEP]
    squared = step**2 / VOL_STEP
    finite = np.isfinite(squared)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, squared, 0.0))])
    counts = np.concatenate([[0], np.cumsum(finite)])
    idx = np.arange(len(combined) + 1)
    lo = np.maximum(idx - VOL_SECONDS, 0)
    vol = np.sqrt((sums[idx] - sums[lo]) / np.maximum(counts[idx] - counts[lo], 1))
    vol[(counts[idx] - counts[lo]) < VOL_SECONDS * 0.8] = np.nan
    # vol[k] uses seconds before grid index k, so it is known at clock GRID_START + k.
    return vol


def vol_at(vol: np.ndarray, clock: np.ndarray) -> np.ndarray:
    idx = np.asarray(clock, dtype="int64") - GRID_START
    ok = (idx >= 0) & (idx < len(vol))
    out = np.full(len(idx), np.nan)
    out[ok] = vol[idx[ok]]
    return out


def probability(estimate, strike, spread, sigma, c0: float, c1: float) -> np.ndarray:
    scale = np.sqrt(c0 + c1 * (sigma * spread) ** 2)
    return norm.cdf((estimate - strike) / scale)


def fit_probability(estimate, strike, spread, sigma, up) -> tuple[float, float]:
    distance = estimate - strike
    noise = (sigma * spread) ** 2
    y = up.astype(bool)

    def nll(theta: np.ndarray) -> float:
        c0, c1 = np.exp(theta)
        z = distance / np.sqrt(c0 + c1 * noise)
        return float(-(norm.logcdf(z[y]).sum() + norm.logcdf(-z[~y]).sum()) / len(z))

    result = minimize(nll, np.log([4.0, 1.0]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    c0, c1 = np.exp(result.x)
    return float(c0), float(c1)
