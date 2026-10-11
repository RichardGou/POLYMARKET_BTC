"""Two settlement samples per window, stored apart on purpose.

price_oracle_opening holds the TWAP sampled at the window start.
price_oracle_close holds the TWAP sampled at the window end.

Gamma only writes the opening price after the window has closed. oracle_ts
is the instant the sample refers to. gamma_closed_time is when Gamma
published the resolution, which is later. Keeping the two files separate
avoids treating the opening print as something observed at the open.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

OPENING_COLUMNS = [
    "timeframe",
    "event_id",
    "market_id",
    "condition_id",
    "slug",
    "oracle_ts",
    "price",
    "gamma_closed_time",
    "gamma_updated_at",
    "seconds_from_oracle_ts_to_gamma_close",
]

CLOSE_COLUMNS = [
    "timeframe",
    "event_id",
    "market_id",
    "condition_id",
    "slug",
    "oracle_ts",
    "price",
    "gamma_closed_time",
    "gamma_updated_at",
    "seconds_from_oracle_ts_to_gamma_close",
    "seconds_from_oracle_ts_to_gamma_update",
]


def _frame(rows: List[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("window_start", "window_end", "gamma_closed_time", "gamma_updated_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def split_oracle_frames(rows: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = _frame(rows)
    if raw.empty:
        empty = pd.DataFrame()
        return empty, empty

    opening = raw.loc[raw["price_to_beat"].notna()].copy()
    opening["oracle_ts"] = opening["window_start"]
    opening["price"] = opening["price_to_beat"]
    opening["seconds_from_oracle_ts_to_gamma_close"] = (
        opening["gamma_closed_time"] - opening["oracle_ts"]
    ).dt.total_seconds()
    opening = opening[OPENING_COLUMNS].sort_values(["timeframe", "oracle_ts"]).reset_index(drop=True)

    close = raw.loc[raw["final_price"].notna()].copy()
    close["oracle_ts"] = close["window_end"]
    close["price"] = close["final_price"]
    close["seconds_from_oracle_ts_to_gamma_close"] = (
        close["gamma_closed_time"] - close["oracle_ts"]
    ).dt.total_seconds()
    close["seconds_from_oracle_ts_to_gamma_update"] = (
        close["gamma_updated_at"] - close["oracle_ts"]
    ).dt.total_seconds()
    close = close[CLOSE_COLUMNS].sort_values(["timeframe", "oracle_ts"]).reset_index(drop=True)
    return opening, close


def save_oracle_frames(
    opening: pd.DataFrame,
    close: pd.DataFrame,
    directory: Path,
) -> Tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    opening_path = directory / "price_oracle_opening.parquet"
    close_path = directory / "price_oracle_close.parquet"
    opening.to_parquet(opening_path, index=False)
    close.to_parquet(close_path, index=False)
    return opening_path, close_path


def publication_lag_summary(opening: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """How late Gamma's timestamps are relative to the sample they describe.

    close_to_outcome is the usable lag: closedTime minus the window end.
    The median is about a minute. close_to_last_gamma_write uses updatedAt,
    which is later overwritten (often about a day afterwards), so it is not
    the moment finalPrice appeared. open_sample_to_outcome includes the whole
    window: it shows the opening price in this file was not observed live.
    """
    pieces = []
    if not close.empty:
        for column, label in (
            ("seconds_from_oracle_ts_to_gamma_close", "close_to_outcome"),
            ("seconds_from_oracle_ts_to_gamma_update", "close_to_last_gamma_write"),
        ):
            grouped = close.groupby("timeframe")[column].describe(percentiles=[0.5, 0.9])
            grouped.insert(0, "lag", label)
            pieces.append(grouped.reset_index())
    if not opening.empty:
        grouped = opening.groupby("timeframe")["seconds_from_oracle_ts_to_gamma_close"].describe(
            percentiles=[0.5, 0.9]
        )
        grouped.insert(0, "lag", "open_sample_to_outcome")
        pieces.append(grouped.reset_index())
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)
