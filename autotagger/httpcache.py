"""Rate-limited, on-disk-cached HTTP for the metadata providers.

Both Apple and MusicBrainz will throttle or ban a client that hammers them, and
a tagging run over a library re-asks the same questions constantly (every track
on an album produces near-identical album lookups). One SQLite-backed cache plus
a token bucket per host solves both problems.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_cache_dir

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(user_cache_dir("autotagger")) / "http-cache.sqlite"
DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 days; catalogue metadata barely moves


class RateLimiter:
    """Simple token bucket, one per host."""

    def __init__(self, calls_per_second: float) -> None:
        self.interval = 1.0 / calls_per_second if calls_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self.interval


class HttpCache:
    def __init__(self, path: Path = DEFAULT_CACHE, ttl: int = DEFAULT_TTL, enabled: bool = True):
        self.enabled = enabled
        self.ttl = ttl
        self.path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                " key TEXT PRIMARY KEY, body TEXT NOT NULL, stored_at REAL NOT NULL)"
            )
            self._conn.commit()

    @staticmethod
    def _key(url: str, params: dict[str, Any] | None) -> str:
        blob = url + "?" + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, url: str, params: dict[str, Any] | None) -> Any | None:
        if not self._conn:
            return None
        key = self._key(url, params)
        with self._lock:
            row = self._conn.execute(
                "SELECT body, stored_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        body, stored_at = row
        if time.time() - stored_at > self.ttl:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    def put(self, url: str, params: dict[str, Any] | None, value: Any) -> None:
        if not self._conn:
            return
        key = self._key(url, params)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, body, stored_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            self._conn.commit()

    def clear(self) -> int:
        if not self._conn:
            return 0
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
        return n


class Fetcher:
    """`get_json` with caching, rate limiting and bounded retry."""

    def __init__(
        self,
        cache: HttpCache | None = None,
        user_agent: str = "autotagger/0.1 (+https://github.com/local/autotagger)",
        timeout: float = 20.0,
    ) -> None:
        self.cache = cache or HttpCache()
        self.limiters: dict[str, RateLimiter] = {}
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def set_rate(self, host: str, calls_per_second: float) -> None:
        self.limiters[host] = RateLimiter(calls_per_second)

    def get_json(
        self, url: str, params: dict[str, Any] | None = None, *, retries: int = 3
    ) -> Any | None:
        cached = self.cache.get(url, params)
        if cached is not None:
            return cached

        host = httpx.URL(url).host or ""
        limiter = self.limiters.get(host)
        backoff = 1.0
        for attempt in range(retries):
            if limiter:
                limiter.wait()
            try:
                resp = self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                log.debug("request failed (%s/%s): %s", attempt + 1, retries, exc)
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    log.debug("non-JSON response from %s", url)
                    return None
                self.cache.put(url, params, data)
                return data

            # Apple answers a throttled client with 403, not 429 — retry both.
            if resp.status_code in (403, 429, 503):
                wait = float(resp.headers.get("Retry-After", backoff))
                log.warning("throttled by %s, sleeping %.1fs", host, wait)
                time.sleep(wait)
                backoff *= 2
                continue

            if 400 <= resp.status_code < 500:
                log.debug("%s returned %s for %s", host, resp.status_code, url)
                return None

            time.sleep(backoff)
            backoff *= 2
        return None

    def get_bytes(self, url: str, *, retries: int = 3) -> bytes | None:
        backoff = 1.0
        for _ in range(retries):
            try:
                resp = self.client.get(url)
            except httpx.HTTPError:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                return None
            time.sleep(backoff)
            backoff *= 2
        return None

    def close(self) -> None:
        self.client.close()
