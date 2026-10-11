"""Gamma is the archive of settled up/down windows.

Each closed event carries eventMetadata.priceToBeat (TWAP at window start)
and, a few minutes later, eventMetadata.finalPrice (TWAP at window end).
Those are the numbers the market actually settled on. The live chart on the
website is a different endpoint and is not used for this backfill.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional

import requests

from utils.http import session

GAMMA_KEYSET = "https://gamma-api.polymarket.com/events/keyset"

SERIES = {
    "5m": "btc-up-or-down-5m",
    "15m": "btc-up-or-down-15m",
}


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().replace(" ", "T")
    if text.endswith("+00"):
        text = text[:-3] + "+00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Python 3.10 accepts 3 or 6 fractional digits, not 5.
    if "." in text:
        head, rest = text.split(".", 1)
        tz_at = next((i for i, char in enumerate(rest) if char in "+-"), len(rest))
        frac, tz = rest[:tz_at], rest[tz_at:]
        text = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_market(event: dict) -> dict:
    markets = event.get("markets") or []
    return markets[0] if markets else {}


def iter_events(
    client: requests.Session,
    timeframe: str,
    end_min: datetime,
    end_max: datetime,
    page_size: int = 100,
    pause_s: float = 0.12,
) -> Iterator[dict]:
    """Walk settled events whose end falls in [end_min, end_max]."""
    series_slug = SERIES[timeframe]
    cursor: Optional[str] = None
    pages = 0
    while True:
        params: Dict[str, object] = {
            "series_slug": series_slug,
            "closed": "true",
            "limit": page_size,
            "order": "endDate",
            "ascending": "true",
            "end_date_min": end_min.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": end_max.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if cursor:
            params["after_cursor"] = cursor
        response = _get_with_retry(client, params)
        body = response.json()
        events = body.get("events") or []
        pages += 1
        if not events:
            break
        stop = False
        kept = 0
        for event in events:
            end = _parse_ts(event.get("endDate"))
            if end is None or end < end_min:
                continue
            if end > end_max:
                stop = True
                break
            kept += 1
            yield event
        print(
            f"{timeframe} page {pages}: kept {kept}/{len(events)}, "
            f"last end={events[-1].get('endDate')}",
            flush=True,
        )
        cursor = body.get("next_cursor")
        if stop or not cursor:
            break
        time.sleep(pause_s)


def _get_with_retry(client: requests.Session, params: Dict[str, object]) -> requests.Response:
    delay = 1.0
    last_error: Exception | None = None
    for _ in range(6):
        try:
            response = client.get(GAMMA_KEYSET, params=params, timeout=90)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(delay)
            delay = min(delay * 2, 20)
            continue
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(delay)
            delay = min(delay * 2, 20)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"Gamma keyset failed: {last_error}")


def event_row(event: dict, timeframe: str) -> dict:
    market = _first_market(event)
    meta = event.get("eventMetadata") or {}
    start = _parse_ts(event.get("startTime") or market.get("eventStartTime"))
    end = _parse_ts(event.get("endDate"))
    return {
        "timeframe": timeframe,
        "event_id": str(event.get("id") or ""),
        "market_id": str(market.get("id") or ""),
        "condition_id": market.get("conditionId") or "",
        "slug": event.get("slug") or market.get("slug") or "",
        "window_start": start,
        "window_end": end,
        "price_to_beat": _float_or_none(meta.get("priceToBeat")),
        "final_price": _float_or_none(meta.get("finalPrice")),
        "gamma_closed_time": _parse_ts(event.get("closedTime") or market.get("closedTime")),
        "gamma_updated_at": _parse_ts(event.get("updatedAt")),
        "outcome_prices": market.get("outcomePrices") or "",
        "automatically_resolved": bool(
            event.get("automaticallyResolved") or market.get("automaticallyResolved")
        ),
    }


def _float_or_none(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def pull_windows(
    end_min: datetime,
    end_max: datetime,
    timeframes: Iterable[str] = ("5m", "15m"),
) -> List[dict]:
    client = session()
    rows: List[dict] = []
    seen = set()
    for timeframe in timeframes:
        for event in iter_events(client, timeframe, end_min, end_max):
            row = event_row(event, timeframe)
            key = (row["timeframe"], row["slug"])
            if key in seen or not row["slug"]:
                continue
            seen.add(key)
            rows.append(row)
    return rows
