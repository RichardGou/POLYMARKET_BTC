"""Download the 40-day oracle samples and write the two parquet files."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from utils.datasets import publication_lag_summary, save_oracle_frames, split_oracle_frames
from utils.gamma import pull_windows
from utils.live import compare_with_gamma, probe_live_clock

ROOT = Path(__file__).resolve().parent


def main() -> None:
    end_max = datetime.now(timezone.utc)
    end_min = end_max - timedelta(days=40)
    print(f"window {end_min.isoformat()} -> {end_max.isoformat()}", flush=True)
    rows = pull_windows(end_min, end_max, timeframes=("5m", "15m"))
    opening, close = split_oracle_frames(rows)
    opening_path, close_path = save_oracle_frames(opening, close, ROOT / "data")
    missing_open = sum(row["price_to_beat"] is None for row in rows)
    missing_close = sum(row["final_price"] is None for row in rows)
    print(f"opening {len(opening)} -> {opening_path}", flush=True)
    print(f"close {len(close)} -> {close_path}", flush=True)
    print(f"missing open {missing_open} missing close {missing_close}", flush=True)
    print(publication_lag_summary(opening, close).to_string(index=False), flush=True)
    print(probe_live_clock().to_string(index=False), flush=True)
    print(compare_with_gamma(opening, close, n=4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
