"""What the website shows while a window is open.

GET https://polymarket.com/api/crypto/crypto-price
    ?symbol=BTC&eventStartTime=...&endDate=...&variant=fiveminute|fifteen

openPrice is filled during the window. closePrice stays null until the
window is over. This is the number drawn on the site. It is not the Gamma
settlement print: on completed windows the two differ by several dollars.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import pandas as pd
import requests

from utils.http import session

CRYPTO_PRICE = "https://polymarket.com/api/crypto/crypto-price"
VARIANTS = {"5m": "fiveminute", "15m": "fifteen"}
DURATIONS = {"5m": timedelta(minutes=5), "15m": timedelta(minutes=15)}


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_window_start(timeframe: str, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    step = int(DURATIONS[timeframe].total_seconds())
    epoch = int(now.timestamp())
    start = epoch - (epoch % step)
    return datetime.fromtimestamp(start, tz=timezone.utc)


def fetch_crypto_price(
    client: requests.Session,
    timeframe: str,
    window_start: datetime,
) -> dict:
    window_end = window_start + DURATIONS[timeframe]
    params = {
        "symbol": "BTC",
        "eventStartTime": _iso(window_start),
        "endDate": _iso(window_end),
        "variant": VARIANTS[timeframe],
    }
    started = time.perf_counter()
    local_before = datetime.now(timezone.utc)
    response = client.get(CRYPTO_PRICE, params=params, timeout=20)
    elapsed_ms = (time.perf_counter() - started) * 1000
    local_after = datetime.now(timezone.utc)
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    server_date = response.headers.get("Date")
    server_ts = parsedate_to_datetime(server_date) if server_date else None
    quote_ms = payload.get("timestamp")
    quote_ts = (
        datetime.fromtimestamp(quote_ms / 1000, tz=timezone.utc) if isinstance(quote_ms, (int, float)) else None
    )
    return {
        "timeframe": timeframe,
        "window_start": window_start,
        "window_end": window_end,
        "http_status": response.status_code,
        "rtt_ms": elapsed_ms,
        "local_before": local_before,
        "local_after": local_after,
        "server_date": server_ts,
        "server_date_minus_local_after_ms": (
            (server_ts - local_after).total_seconds() * 1000 if server_ts is not None else None
        ),
        "quote_ts": quote_ts,
        "quote_ts_minus_local_after_ms": (
            (quote_ts - local_after).total_seconds() * 1000 if quote_ts is not None else None
        ),
        "open_price": payload.get("openPrice"),
        "close_price": payload.get("closePrice"),
        "completed": payload.get("completed"),
        "incomplete": payload.get("incomplete"),
        "cached": payload.get("cached"),
    }


def probe_live_clock(client: Optional[requests.Session] = None) -> pd.DataFrame:
    """RTT and clock offset of the endpoint that draws the live open price."""
    client = client or session()
    rows = []
    now = datetime.now(timezone.utc)
    for timeframe in ("5m", "15m"):
        start = current_window_start(timeframe, now)
        # One in-progress window, plus the window that just finished.
        for window_start in (start, start - DURATIONS[timeframe]):
            try:
                rows.append(fetch_crypto_price(client, timeframe, window_start))
            except requests.RequestException as exc:
                rows.append(
                    {
                        "timeframe": timeframe,
                        "window_start": window_start,
                        "http_status": None,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def compare_with_gamma(
    opening: pd.DataFrame,
    close: pd.DataFrame,
    n: int = 6,
) -> pd.DataFrame:
    """Latest settled Gamma prints versus the website crypto-price route."""
    if close.empty or opening.empty:
        return pd.DataFrame()
    client = session()
    recent = close.sort_values("oracle_ts").groupby("timeframe").tail(n)
    opens = opening.set_index("slug")["price"]
    rows = []
    for record in recent.itertuples(index=False):
        start = record.oracle_ts.to_pydatetime() - DURATIONS[record.timeframe]
        try:
            live = fetch_crypto_price(client, record.timeframe, start)
        except requests.RequestException as exc:
            rows.append({"slug": record.slug, "error": str(exc)})
            continue
        gamma_open = opens.get(record.slug)
        site_open = live["open_price"]
        site_close = live["close_price"]
        rows.append(
            {
                "timeframe": record.timeframe,
                "slug": record.slug,
                "gamma_open": None if gamma_open is None else float(gamma_open),
                "site_open": None if site_open is None else float(site_open),
                "open_gamma_minus_site": (
                    None if gamma_open is None or site_open is None else float(gamma_open) - float(site_open)
                ),
                "gamma_close": float(record.price),
                "site_close": None if site_close is None else float(site_close),
                "close_gamma_minus_site": (
                    None if site_close is None else float(record.price) - float(site_close)
                ),
                "site_completed": live["completed"],
                "http_status": live["http_status"],
            }
        )
        time.sleep(0.05)
    return pd.DataFrame(rows)
