"""Download three days of 1-second prices and fit the oracle."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.fit_oracle import run_fit
from utils.providers import download_binance, download_coinbase, download_kraken, download_okx

ROOT = Path(__file__).resolve().parent


def download_all() -> None:
    jobs = {
        "binance": download_binance,
        "okx": download_okx,
        "coinbase": download_coinbase,
        "kraken": download_kraken,
    }
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_of = {pool.submit(function): name for name, function in jobs.items()}
        for future in as_completed(future_of):
            name = future_of[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"{name} FAILED: {exc}", flush=True)
    if errors:
        raise RuntimeError(" ; ".join(errors))


if __name__ == "__main__":
    download_all()
    summary = run_fit(ROOT)
    selected = summary["selected"]
    print("selected", selected["window"], "delta", selected["delta_seconds"], flush=True)
    print("alpha", selected["alpha"], "base", selected["base_usd"], flush=True)
    print("train", selected["train"], flush=True)
    print("test", selected["test"], flush=True)
    print("train_sign", selected["train_sign"], flush=True)
    print("test_sign", selected["test_sign"], flush=True)
