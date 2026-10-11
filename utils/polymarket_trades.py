"""Taker fills of each BTC up/down window, one file per market.

data/polymarket/trades/<timeframe>/<slug>.parquet, from the public
data-api /trades route. Each row is a taker execution: second, outcome
(Up or Down), taker side, price and size. A BUY on a token was filled at
that token's ask; a SELL at its bid.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from utils.http import session

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "data" / "polymarket" / "trades"
TRADES_URL = "https://data-api.polymarket.com/trades"
PAGE = 1000


def trade_path(timeframe: str, slug: str) -> Path:
    return TRADES / timeframe / f"{slug}.parquet"


def fetch_market(client: requests.Session, condition_id: str) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    offset = 0
    complete = True
    while True:
        delay = 1.0
        for _ in range(8):
            response = client.get(
                TRADES_URL,
                params={"market": condition_id, "limit": PAGE, "offset": offset, "takerOnly": "true"},
                timeout=60,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            break
        if response.status_code != 200:
            complete = False
            break
        batch = response.json()
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    return rows, complete


def download_one(timeframe: str, slug: str, condition_id: str) -> dict:
    path = trade_path(timeframe, slug)
    if path.exists():
        return {"slug": slug, "skipped": True}
    rows, complete = fetch_market(session(), condition_id)
    frame = pd.DataFrame(
        {
            "sec": [int(row["timestamp"]) for row in rows],
            "outcome": [row.get("outcome") for row in rows],
            "side": [row.get("side") for row in rows],
            "price": [float(row["price"]) for row in rows],
            "size": [float(row["size"]) for row in rows],
            "tx": [row.get("transactionHash") for row in rows],
        }
    )
    frame = frame.drop_duplicates().sort_values("sec", kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"slug": slug, "rows": len(frame), "complete": complete}


def download_windows(windows: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    todo = [
        (row.timeframe, row.slug, row.condition_id)
        for row in windows.itertuples(index=False)
        if not trade_path(row.timeframe, row.slug).exists()
    ]
    print(f"polymarket trades: {len(todo)} markets to fetch", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, *job) for job in todo]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0:
                print(f"polymarket trades {index}/{len(todo)}", flush=True)
    status = pd.DataFrame(results)
    log = TRADES / "download_log.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    log.write_text(json.dumps(previous + results, indent=1), encoding="utf-8")
    return status


def load_market(timeframe: str, slug: str) -> pd.DataFrame | None:
    path = trade_path(timeframe, slug)
    if not path.exists():
        return None
    return pd.read_parquet(path)
