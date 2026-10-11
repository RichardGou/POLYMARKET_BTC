"""Download one month of inputs. Each source can run in its own process:

    python download_month.py binance
    python download_month.py okx
    python download_month.py coinbase
    python download_month.py kraken
    python download_month.py usdt
    python download_month.py polymarket          # test days
    python download_month.py polymarket-train    # training days

Files already on disk are skipped, so a run can be restarted.
"""

import sys

import pandas as pd

from utils.exchanges import days, download_usdt_usd, download_venue
from utils.month import FIRST_PRICE_DAY, LABEL_START, LAST_PRICE_DAY, ROOT, TEST_END, TEST_START
from utils.polymarket_trades import download_windows

WORKERS = {"binance": 4, "okx": 20, "coinbase": 12, "kraken": 1}


def polymarket(first, last) -> None:
    opening = pd.read_parquet(ROOT / "data" / "price_oracle_opening.parquet")
    opening["oracle_ts"] = pd.to_datetime(opening["oracle_ts"], utc=True)
    chosen = opening[(opening["oracle_ts"] >= first) & (opening["oracle_ts"] < last)]
    download_windows(chosen[["timeframe", "slug", "condition_id"]])


if __name__ == "__main__":
    source = sys.argv[1]
    if source == "usdt":
        download_usdt_usd(FIRST_PRICE_DAY, LAST_PRICE_DAY)
    elif source == "polymarket":
        polymarket(TEST_START, TEST_END)
    elif source == "polymarket-train":
        polymarket(LABEL_START, TEST_START)
    else:
        download_venue(source, days(FIRST_PRICE_DAY, LAST_PRICE_DAY), workers=WORKERS[source])
    print(f"{source} done", flush=True)
