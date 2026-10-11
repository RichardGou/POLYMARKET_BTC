"""One second BTC prices, one folder per venue.

Binance and OKX are USDT books. Coinbase and Kraken are USD books.
The stored price is the last price of each second. Gaps are left empty;
the TWAP step carries the last price forward for at most two minutes.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / "data" / "providers"

# Labels cover three full UTC days. The extra margin feeds the 60s window
# and a delta of a few seconds on either side.
LABEL_START = datetime(2026, 10, 8, tzinfo=timezone.utc)
LABEL_END = datetime(2026, 10, 11, tzinfo=timezone.utc)
PRICE_START = LABEL_START - timedelta(minutes=10)
PRICE_END = LABEL_END + timedelta(minutes=2)


def _session() -> requests.Session:
    client = requests.Session()
    client.headers.update({"User-Agent": "polymarket-oracle-fit/0.1", "Accept": "application/json"})
    return client


def _write(provider: str, frame: pd.DataFrame, meta: dict) -> Path:
    directory = PROVIDERS / provider
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame.drop_duplicates("ts").sort_values("ts")
    frame = frame[(frame["ts"] >= PRICE_START) & (frame["ts"] < PRICE_END)]
    frame["price"] = frame["price"].astype("float64")
    path = directory / "btc_1s.parquet"
    frame.to_parquet(path, index=False)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{provider}: {len(frame)} seconds -> {path}", flush=True)
    return path


def _get(client: requests.Session, url: str, **kwargs) -> requests.Response:
    delay = 1.0
    for _ in range(6):
        response = client.get(url, timeout=60, **kwargs)
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response


def download_binance(client: requests.Session | None = None) -> Path:
    client = client or _session()
    frames = []
    day = datetime(2026, 10, 7, tzinfo=timezone.utc)
    last = datetime(2026, 10, 10, tzinfo=timezone.utc)
    while day <= last:
        stamp = day.strftime("%Y-%m-%d")
        url = (
            "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/"
            f"BTCUSDT-1s-{stamp}.zip"
        )
        response = _get(client, url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            name = archive.namelist()[0]
            raw = pd.read_csv(archive.open(name), header=None, usecols=[0, 4], names=["open_ms", "price"])
        raw["ts"] = pd.to_datetime(raw["open_ms"], unit="us", utc=True)
        frames.append(raw[["ts", "price"]])
        print(f"binance {stamp}: {len(raw)}", flush=True)
        day += timedelta(days=1)
    frame = pd.concat(frames, ignore_index=True)
    return _write(
        "binance",
        frame,
        {
            "provider": "binance",
            "symbol": "BTCUSDT",
            "quote": "USDT",
            "granularity": "1s",
            "price": "spot kline close",
            "source": "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/",
        },
    )


def download_okx(client: requests.Session | None = None) -> Path:
    client = client or _session()
    after_ms = int(PRICE_END.timestamp() * 1000)
    stop_ms = int(PRICE_START.timestamp() * 1000)
    rows: list[tuple[int, float]] = []
    pages = 0
    while True:
        response = _get(
            client,
            "https://www.okx.com/api/v5/market/history-candles",
            params={"instId": "BTC-USDT", "bar": "1s", "limit": "100", "after": str(after_ms)},
        )
        payload = response.json()
        batch = payload.get("data") or []
        if not batch:
            break
        oldest = min(int(item[0]) for item in batch)
        for item in batch:
            rows.append((int(item[0]), float(item[4])))
        pages += 1
        if pages % 200 == 0:
            print(f"okx page {pages}, oldest {datetime.fromtimestamp(oldest/1000, timezone.utc)}", flush=True)
        if oldest <= stop_ms or oldest >= after_ms:
            break
        after_ms = oldest
        time.sleep(0.12)
    frame = pd.DataFrame(rows, columns=["open_ms", "price"])
    frame["ts"] = pd.to_datetime(frame["open_ms"], unit="ms", utc=True)
    print(f"okx pages {pages}", flush=True)
    return _write(
        "okx",
        frame[["ts", "price"]],
        {
            "provider": "okx",
            "symbol": "BTC-USDT",
            "quote": "USDT",
            "granularity": "1s",
            "price": "spot candle close",
            "source": "https://www.okx.com/api/v5/market/history-candles",
        },
    )


def download_coinbase(client: requests.Session | None = None) -> Path:
    client = client or _session()
    after: str | None = None
    rows: list[tuple[datetime, float]] = []
    pages = 0
    while True:
        params = {"limit": 1000}
        if after is not None:
            params["after"] = after
        response = _get(client, "https://api.exchange.coinbase.com/products/BTC-USD/trades", params=params)
        trades = response.json()
        if not isinstance(trades, list) or not trades:
            break
        oldest_id = None
        oldest_time = None
        for trade in trades:
            stamp = datetime.fromisoformat(trade["time"].replace("Z", "+00:00"))
            rows.append((stamp, float(trade["price"])))
            trade_id = int(trade["trade_id"])
            if oldest_id is None or trade_id < oldest_id:
                oldest_id = trade_id
            if oldest_time is None or stamp < oldest_time:
                oldest_time = stamp
        oldest_id = None if oldest_id is None else str(oldest_id)
        pages += 1
        if pages % 100 == 0:
            print(f"coinbase page {pages}, oldest {oldest_time}", flush=True)
        if oldest_time is None or oldest_time <= PRICE_START or oldest_id is None or oldest_id == after:
            break
        if pages > 1 and after is not None and int(oldest_id) >= int(after):
            raise RuntimeError("coinbase pagination did not move backward")
        after = oldest_id
        time.sleep(0.06)
    frame = _last_price_per_second(rows)
    print(f"coinbase pages {pages}, trades {len(rows)}", flush=True)
    return _write(
        "coinbase",
        frame,
        {
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "quote": "USD",
            "granularity": "1s",
            "price": "last trade in the second",
            "source": "https://api.exchange.coinbase.com/products/BTC-USD/trades",
        },
    )


def download_kraken(client: requests.Session | None = None) -> Path:
    client = client or _session()
    since = str(int(PRICE_START.timestamp() * 1_000_000_000))
    stop = PRICE_END.timestamp()
    rows: list[tuple[datetime, float]] = []
    pages = 0
    while True:
        response = _get(
            client,
            "https://api.kraken.com/0/public/Trades",
            params={"pair": "XBTUSD", "since": since},
        )
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        result = payload["result"]
        key = next(name for name in result if name != "last")
        batch = result[key]
        cursor = str(result["last"])
        if not batch:
            break
        for price, _volume, stamp, *_rest in batch:
            rows.append((datetime.fromtimestamp(float(stamp), timezone.utc), float(price)))
        pages += 1
        last_stamp = float(batch[-1][2])
        if pages % 50 == 0:
            print(f"kraken page {pages}, at {datetime.fromtimestamp(last_stamp, timezone.utc)}", flush=True)
        if last_stamp >= stop or cursor == since:
            break
        since = cursor
        time.sleep(1.1)
    frame = _last_price_per_second(rows)
    print(f"kraken pages {pages}, trades {len(rows)}", flush=True)
    return _write(
        "kraken",
        frame,
        {
            "provider": "kraken",
            "symbol": "XBTUSD",
            "quote": "USD",
            "granularity": "1s",
            "price": "last trade in the second",
            "source": "https://api.kraken.com/0/public/Trades",
        },
    )


def _last_price_per_second(rows: list[tuple[datetime, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["raw_ts", "price"])
    if frame.empty:
        return pd.DataFrame(columns=["ts", "price"])
    frame["ts"] = frame["raw_ts"].dt.floor("s")
    frame = frame.sort_values("raw_ts").groupby("ts", as_index=False)["price"].last()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame[["ts", "price"]]
