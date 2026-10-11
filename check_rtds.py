"""Compare a live RTDS Chainlink recording with the settlement prices.

1. Which average of the per-second Chainlink reports equals the Gamma
   price to beat / final price?
2. How late are Chainlink reports compared with Binance one-second candles,
   and how late do they reach us?
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils.gamma import pull_windows

ROOT = Path(__file__).resolve().parent


def load_recording(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        payload = item["msg"].get("payload") or {}
        if "value" in payload:
            rows.append({"sec": int(payload["timestamp"]) // 1000, "value": float(payload["value"]), "recv_ms": item["recv_ms"], "report_ms": int(payload["timestamp"])})
    frame = pd.DataFrame(rows).drop_duplicates("sec", keep="first").sort_values("sec")
    return frame


def binance_seconds(start: int, stop: int) -> pd.Series:
    out = {}
    cursor = start * 1000
    while cursor < stop * 1000:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1s", "startTime": cursor, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        for candle in batch:
            out[int(candle[0]) // 1000] = float(candle[4])
        cursor = int(batch[-1][0]) + 1000
    return pd.Series(out).sort_index()


def main(path: Path) -> None:
    rec = load_recording(path)
    chain = rec.set_index("sec")["value"]
    first, last = int(chain.index.min()), int(chain.index.max())
    delay = (rec["recv_ms"] - rec["report_ms"]) / 1000
    report = {
        "seconds_recorded": int(len(chain)),
        "span_seconds": last - first + 1,
        "receive_delay_s": {"median": float(delay.median()), "p90": float(delay.quantile(0.9)), "max": float(delay.max())},
    }

    windows = pd.DataFrame(
        pull_windows(datetime.fromtimestamp(first + 60, timezone.utc), datetime.fromtimestamp(last, timezone.utc) + timedelta(minutes=1))
    )
    stamps = []
    for row in windows.itertuples(index=False):
        for stamp, price in ((row.window_start, row.price_to_beat), (row.window_end, row.final_price)):
            if price is not None and np.isfinite(price):
                stamps.append({"sec": int(stamp.timestamp()), "price": float(price), "timeframe": row.timeframe})
    stamps = pd.DataFrame(stamps).drop_duplicates("sec")
    stamps = stamps[(stamps["sec"] - 61 >= first) & (stamps["sec"] <= last)]
    conventions = {"[T-60,T-1]": (-60, -1), "[T-59,T]": (-59, 0), "[T-60,T]": (-60, 0), "[T-61,T-2]": (-61, -2)}
    checks = []
    for row in stamps.itertuples(index=False):
        entry = {"stamp": datetime.fromtimestamp(row.sec, timezone.utc).isoformat(), "gamma": row.price}
        for name, (lo, hi) in conventions.items():
            values = chain.reindex(range(row.sec + lo, row.sec + hi + 1))
            entry[name] = float(values.mean()) - row.price if values.notna().all() else None
        entry["last_report_minus_gamma"] = float(chain.get(row.sec, np.nan)) - row.price
        checks.append(entry)
    report["settlement_checks"] = checks

    binance = binance_seconds(first, last + 1)
    joined = pd.DataFrame({"chain": chain, "binance": binance}).dropna()
    d_chain = joined["chain"].diff()
    d_binance = joined["binance"].diff()
    corr = {}
    for lag in range(-3, 11):
        corr[lag] = float(d_chain.corr(d_binance.shift(lag)))
    report["corr_chainlink_move_vs_binance_move_k_seconds_earlier"] = corr
    report["best_lag_s"] = max(corr, key=corr.get)
    out = ROOT / "data" / "live" / "rtds_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "data" / "live").glob("rtds_*.jsonl"))[-1])
