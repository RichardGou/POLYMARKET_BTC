"""HTTP session for Polymarket.

The home resolver answers polymarket.com with the ANJ block page
(certificate CN=*.anj.fr, plus ::1). Google public DNS still returns the
real Cloudflare addresses. This session resolves those hosts through
DNS-over-HTTPS and connects to that address, while TLS keeps checking the
certificate against the real hostname.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
from typing import Dict, List
from urllib.parse import urlsplit

import requests
import urllib3.util.connection as urllib3_connection

_DOH = "https://dns.google/resolve?name={host}&type=A"
_PATCH_LOCK = threading.Lock()
_PATCHED = False
_ORIGINAL_CREATE_CONNECTION = urllib3_connection.create_connection
_CACHE: Dict[str, List[str]] = {}


def _resolve_a(host: str) -> List[str]:
    cached = _CACHE.get(host)
    if cached:
        return cached
    url = _DOH.format(host=host)
    request = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    answers = [
        item["data"]
        for item in payload.get("Answer", [])
        if item.get("type") == 1 and item.get("data")
    ]
    if not answers:
        raise OSError(f"no A record for {host}")
    _CACHE[host] = answers
    return answers


def _create_connection(address, *args, **kwargs):
    host, port = address
    if host == "polymarket.com" or host.endswith(".polymarket.com"):
        last_error: Exception | None = None
        for ip in _resolve_a(host):
            try:
                return _ORIGINAL_CREATE_CONNECTION((ip, port), *args, **kwargs)
            except OSError as exc:
                last_error = exc
        raise OSError(f"unable to connect to {host}") from last_error
    return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)


def install_dns_bypass() -> None:
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        urllib3_connection.create_connection = _create_connection
        socket.create_connection = _create_connection
        _PATCHED = True


def session() -> requests.Session:
    install_dns_bypass()
    client = requests.Session()
    client.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "polymarket-oracle-dataset/0.1",
        }
    )
    return client


def host_of(url: str) -> str:
    return urlsplit(url).hostname or ""
