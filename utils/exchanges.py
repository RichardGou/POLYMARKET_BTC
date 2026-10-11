"""Daily one-second BTC prices, one folder per venue.

data/providers/<venue>/1s/YYYY-MM-DD.parquet holds the last price of every
second of that UTC day that had a print, as columns sec (epoch seconds)
and price. Missing seconds stay missing; the model carries prices forward.

Binance and OKX are BTCUSDT books. Coinbase and Kraken are BTC-USD books.
USDT/USD comes from Coinbase one-minute candles, stamped at the end of the
minute so a value is only used once it was known.
"""

from __future__ import annotations

import io
import threading
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / "data" / "providers"
VENUES = ("binance", "okx", "coinbase", "kraken")
USDT_VENUES = ("binance", "okx")


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if self.next_at > now:
                time.sleep(self.next_at - now)
                now = self.next_at
            self.next_at = now + self.interval


def _session() -> requests.Session:
    client = requests.Session()
    client.headers.update({"User-Agent": "polymarket-oracle-fit/0.2", "Accept": "application/json"})
    return client


def _get(client: requests.Session, url: str, limiter: RateLimiter | None = None, **kwargs) -> requests.Response:
    delay = 1.0
    last: Exception | None = None
    for _ in range(8):
        if limiter is not None:
            limiter.wait()
        try:
            response = client.get(url, timeout=60, **kwargs)
        except requests.RequestException as exc:
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"GET {url} failed: {last}")


def day_path(venue: str, day: date) -> Path:
    return PROVIDERS / venue / "1s" / f"{day.isoformat()}.parquet"


def _bounds(day: date) -> tuple[int, int]:
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    return start, start + 86400


def _save_day(venue: str, day: date, secs: np.ndarray, prices: np.ndarray) -> Path:
    start, end = _bounds(day)
    frame = pd.DataFrame({"sec": secs.astype("int64"), "price": prices.astype("float64")})
    frame = frame[(frame["sec"] >= start) & (frame["sec"] < end)]
    frame = frame.drop_duplicates("sec", keep="last").sort_values("sec")
    path = day_path(venue, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"{venue} {day}: {len(frame)} seconds", flush=True)
    return path


def _last_per_second(stamps: np.ndarray, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(stamps, kind="stable")
    secs = np.floor(stamps[order]).astype("int64")
    values = prices[order]
    keep = np.r_[secs[1:] != secs[:-1], True]
    return secs[keep], values[keep]


def binance_day(day: date, client: requests.Session | None = None) -> Path:
    client = client or _session()
    url = (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/"
        f"BTCUSDT-1s-{day.isoformat()}.zip"
    )
    response = _get(client, url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        raw = pd.read_csv(archive.open(archive.namelist()[0]), header=None, usecols=[0, 4], names=["open", "price"])
    open_time = raw["open"].to_numpy(dtype="int64")
    divisor = 1_000_000 if open_time.max() > 10**14 else 1000
    return _save_day("binance", day, open_time // divisor, raw["price"].to_numpy())


OKX_LIMIT = RateLimiter(9.0)


def okx_day(day: date, client: requests.Session | None = None) -> Path:
    client = client or _session()
    start, end = _bounds(day)
    after_ms = end * 1000
    secs: list[int] = []
    prices: list[float] = []
    while True:
        response = _get(
            client,
            "https://www.okx.com/api/v5/market/history-candles",
            OKX_LIMIT,
            params={"instId": "BTC-USDT", "bar": "1s", "limit": "100", "after": str(after_ms)},
        )
        batch = response.json().get("data") or []
        if not batch:
            break
        oldest = min(int(item[0]) for item in batch)
        for item in batch:
            secs.append(int(item[0]) // 1000)
            prices.append(float(item[4]))
        if oldest <= start * 1000 or oldest >= after_ms:
            break
        after_ms = oldest
    return _save_day("okx", day, np.array(secs), np.array(prices))


COINBASE_LIMIT = RateLimiter(9.0)
COINBASE_TRADES = "https://api.exchange.coinbase.com/products/BTC-USD/trades"


def _coinbase_at_or_below(client: requests.Session, trade_id: int) -> tuple[int, float]:
    response = _get(client, COINBASE_TRADES, COINBASE_LIMIT, params={"limit": 1, "after": str(trade_id + 1)})
    trade = response.json()[0]
    stamp = datetime.fromisoformat(trade["time"].replace("Z", "+00:00")).timestamp()
    return int(trade["trade_id"]), stamp


def _coinbase_first_id_at(client: requests.Session, target: int, latest_id: int) -> int:
    """Smallest trade id whose time is at or after target."""
    lo = max(1, latest_id - 40_000_000)
    hi = latest_id
    while hi - lo > 1:
        mid = (lo + hi) // 2
        found_id, stamp = _coinbase_at_or_below(client, mid)
        if stamp >= target:
            hi = mid
        else:
            lo = max(mid, found_id)
    return hi


def coinbase_day(day: date, client: requests.Session | None = None) -> Path:
    client = client or _session()
    start, end = _bounds(day)
    head = _get(client, COINBASE_TRADES, COINBASE_LIMIT, params={"limit": 1}).json()[0]
    latest_id = int(head["trade_id"])
    upper = _coinbase_first_id_at(client, end, latest_id)
    cursor = upper
    stamps: list[float] = []
    prices: list[float] = []
    while True:
        response = _get(client, COINBASE_TRADES, COINBASE_LIMIT, params={"limit": 1000, "after": str(cursor)})
        trades = response.json()
        if not trades:
            break
        oldest_id = cursor
        oldest_time = float("inf")
        for trade in trades:
            stamp = datetime.fromisoformat(trade["time"].replace("Z", "+00:00")).timestamp()
            stamps.append(stamp)
            prices.append(float(trade["price"]))
            oldest_id = min(oldest_id, int(trade["trade_id"]))
            oldest_time = min(oldest_time, stamp)
        if oldest_time < start or oldest_id >= cursor:
            break
        cursor = oldest_id
    secs, values = _last_per_second(np.array(stamps), np.array(prices))
    return _save_day("coinbase", day, secs, values)


def kraken_day(day: date, client: requests.Session | None = None) -> Path:
    client = client or _session()
    start, end = _bounds(day)
    since = str(start * 1_000_000_000)
    stamps: list[float] = []
    prices: list[float] = []
    pause = 0.7
    while True:
        response = _get(client, "https://api.kraken.com/0/public/Trades", params={"pair": "XBTUSD", "since": since})
        payload = response.json()
        errors = payload.get("error") or []
        if errors:
            if any("Too many requests" in error or "Rate limit" in error for error in errors):
                pause = min(pause * 2, 10)
                time.sleep(pause)
                continue
            raise RuntimeError(str(errors))
        result = payload["result"]
        key = next(name for name in result if name != "last")
        batch = result[key]
        cursor = str(result["last"])
        if not batch:
            break
        for price, _volume, stamp, *_rest in batch:
            stamps.append(float(stamp))
            prices.append(float(price))
        if float(batch[-1][2]) >= end or cursor == since:
            break
        since = cursor
        time.sleep(pause)
        pause = max(0.7, pause * 0.9)
    secs, values = _last_per_second(np.array(stamps), np.array(prices))
    return _save_day("kraken", day, secs, values)


DOWNLOADERS: dict[str, Callable[[date], Path]] = {
    "binance": binance_day,
    "okx": okx_day,
    "coinbase": coinbase_day,
    "kraken": kraken_day,
}


def days(first: date, last: date) -> list[date]:
    out = []
    current = first
    while current <= last:
        out.append(current)
        current += timedelta(days=1)
    return out


def download_venue(venue: str, wanted: Iterable[date], workers: int = 1) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = [day for day in wanted if not day_path(venue, day).exists()]
    print(f"{venue}: {len(todo)} days to fetch", flush=True)
    function = DOWNLOADERS[venue]
    if workers <= 1:
        for day in todo:
            function(day)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, day): day for day in todo}
        for future in as_completed(futures):
            future.result()


def usdt_usd_path() -> Path:
    return PROVIDERS / "coinbase" / "usdt_usd" / "1m.parquet"


def download_usdt_usd(first: date, last: date) -> Path:
    client = _session()
    start, _ = _bounds(first)
    _, stop = _bounds(last)
    rows: list[tuple[int, float]] = []
    cursor = start
    while cursor < stop:
        chunk_end = min(cursor + 300 * 60, stop)
        response = _get(
            client,
            "https://api.exchange.coinbase.com/products/USDT-USD/candles",
            COINBASE_LIMIT,
            params={
                "granularity": 60,
                "start": datetime.fromtimestamp(cursor, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": datetime.fromtimestamp(chunk_end, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        for candle in response.json():
            rows.append((int(candle[0]) + 60, float(candle[4])))
        cursor = chunk_end
    frame = pd.DataFrame(rows, columns=["sec", "usdt_usd"]).drop_duplicates("sec").sort_values("sec")
    path = usdt_usd_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"usdt_usd: {len(frame)} minutes", flush=True)
    return path


def load_venue(venue: str, first: date, last: date) -> pd.Series:
    frames = [pd.read_parquet(day_path(venue, day)) for day in days(first, last) if day_path(venue, day).exists()]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("sec", keep="last").sort_values("sec")
    return pd.Series(frame["price"].to_numpy(dtype="float64"), index=frame["sec"].to_numpy(dtype="int64"))


def load_usdt_usd() -> pd.Series:
    frame = pd.read_parquet(usdt_usd_path())
    return pd.Series(frame["usdt_usd"].to_numpy(dtype="float64"), index=frame["sec"].to_numpy(dtype="int64"))
