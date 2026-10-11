"""Record Polymarket's live data socket (RTDS).

Topic crypto_prices_chainlink relays the Chainlink streams that the up/down
markets settle on, one report per second and symbol. A symbol filter only
returns a snapshot, so the socket subscribes to every symbol and keeps
btc/usd. Each message is written as one JSON line with the local receive
time, under data/live/rtds_<UTC date>.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

from utils.http import _resolve_a

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
HOST = "ws-live-data.polymarket.com"
SUBSCRIPTIONS = [{"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}]
SYMBOL = "btc/usd"


async def _ping(socket) -> None:
    while True:
        await asyncio.sleep(5)
        await socket.send("PING")


async def record(seconds: float) -> Path:
    LIVE.mkdir(parents=True, exist_ok=True)
    path = LIVE / f"rtds_{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    stop_at = time.time() + seconds
    ip = _resolve_a(HOST)[0]
    context = ssl.create_default_context()
    with path.open("a", encoding="utf-8") as sink:
        while time.time() < stop_at:
            try:
                async with websockets.connect(
                    f"wss://{HOST}", host=ip, port=443, ssl=context, server_hostname=HOST, open_timeout=20
                ) as socket:
                    await socket.send(json.dumps({"action": "subscribe", "subscriptions": SUBSCRIPTIONS}))
                    pinger = asyncio.create_task(_ping(socket))
                    try:
                        while time.time() < stop_at:
                            raw = await asyncio.wait_for(socket.recv(), timeout=30)
                            if not raw or raw == "PONG":
                                continue
                            message = json.loads(raw)
                            if (message.get("payload") or {}).get("symbol") != SYMBOL:
                                continue
                            sink.write(json.dumps({"recv_ms": int(time.time() * 1000), "msg": message}) + "\n")
                            sink.flush()
                    finally:
                        pinger.cancel()
            except (OSError, asyncio.TimeoutError, websockets.WebSocketException, json.JSONDecodeError) as exc:
                print(f"rtds reconnect after {exc!r}", flush=True)
                await asyncio.sleep(2)
    return path


def run(seconds: float) -> Path:
    return asyncio.run(record(seconds))
