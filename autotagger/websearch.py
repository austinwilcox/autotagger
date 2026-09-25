"""Web search via Ollama's hosted search API, for files no catalogue contains.

    POST https://ollama.com/api/web_search   {"query": ..., "max_results": n}
    POST https://ollama.com/api/web_fetch    {"url": ...}

Needs a free key from https://ollama.com/settings/keys (`ollama signin`, then
export OLLAMA_API_KEY). Off unless --web-search is passed.

Two things this deliberately does NOT do:

  * It does not let the model browse freely. One search, one structured
    extraction, done.
  * It does not let web text become a tag value unchallenged. The preferred
    outcome is a *better catalogue query* — the web reveals the correct artist
    spelling or the compilation a track appears on, that query is re-run against
    Deezer and Apple, and the result is verified by duration like anything else.
    A candidate synthesized straight from web text is marked `source="web"`,
    carries no duration to verify against, and is capped so it cannot auto-apply
    without `--web-trust`.

Page content is untrusted input. The extraction prompt says so explicitly, and
nothing here executes, follows, or fetches anything a page asks for.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .httpcache import HttpCache
from .llm import LLMClient, LLMError
from .models import AudioFile, Candidate
from .prompts import WEB_SCHEMA, web_messages, web_query_for

log = logging.getLogger(__name__)

SEARCH_URL = "https://ollama.com/api/web_search"
FETCH_URL = "https://ollama.com/api/web_fetch"


class WebSearchError(Exception):
    pass


@dataclass
class WebIdentification:
    """What the web says this file is. Advisory until something verifies it."""

    found: bool
    confidence: float
    reasoning: str = ""
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    label: str | None = None
    year: str | None = None
    genre: str | None = None
    duration: float | None = None
    source_urls: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.found:
            return "web search found nothing conclusive"
        bits = [f"{self.artist or '?'} — {self.title or '?'}"]
        if self.album:
            bits.append(f"[{self.album}]")
        if self.label:
            bits.append(f"on {self.label}")
        if self.year:
            bits.append(f"({self.year})")
        bits.append(f"{self.confidence:.0%} confident")
        if self.source_urls:
            bits.append(f"— {self.source_urls[0]}")
        return " ".join(bits)

    def to_candidate(self) -> Candidate | None:
        """Synthesize a candidate from web text. Unverified by construction."""
        if not self.found or not self.artist or not self.title:
            return None
        return Candidate(
            source="web",
            source_id=self.source_urls[0] if self.source_urls else f"{self.artist}:{self.title}",
            title=self.title,
            artist=self.artist,
            album=self.album,
            album_artist=self.artist,
            year=self.year,
            genre=self.genre,
            duration=self.duration,
            label=self.label,
            extra={"source_urls": self.source_urls, "web_confidence": self.confidence},
        )


class OllamaWebSearch:
    def __init__(self, api_key: str | None = None, cache: HttpCache | None = None,
                 max_results: int = 5, timeout: float = 45.0):
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.cache = cache
        self.max_results = max(1, min(max_results, 10))  # API caps at 10
        self.client = httpx.Client(timeout=timeout)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def setup_hint(self) -> str:
        return (
            "Web search needs an Ollama API key:\n"
            "    ollama signin\n"
            "    # then create a key at https://ollama.com/settings/keys\n"
            "    export OLLAMA_API_KEY=...\n"
            "or pass --ollama-key."
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        if not self.configured:
            raise WebSearchError("no OLLAMA_API_KEY set")
        payload = {"query": query, "max_results": self.max_results}

        if self.cache:
            cached = self.cache.get(SEARCH_URL, payload)
            if cached is not None:
                return cached.get("results", [])
        try:
            resp = self.client.post(
                SEARCH_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise WebSearchError(f"web search request failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise WebSearchError(f"Ollama rejected the API key ({resp.status_code})")
        if resp.status_code == 429:
            raise WebSearchError("Ollama web search rate limit reached")
        if resp.status_code != 200:
            raise WebSearchError(f"web search returned HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise WebSearchError("web search returned non-JSON") from exc
        if self.cache:
            self.cache.put(SEARCH_URL, payload, data)
        return data.get("results", [])

    def close(self) -> None:
        self.client.close()


class WebIdentifier:
    """Search the web for one file, then have the LLM extract structured metadata."""

    def __init__(self, search_client: OllamaWebSearch, llm: LLMClient):
        self.search = search_client
        self.llm = llm

    def identify(self, af: AudioFile) -> WebIdentification | None:
        query = web_query_for(af)
        try:
            results = self.search.search(query)
        except WebSearchError as exc:
            log.debug("web search failed for %s: %s", af.path.name, exc)
            return None
        if not results:
            return WebIdentification(found=False, confidence=0.0,
                                     reasoning="web search returned no results")
        try:
            data = self.llm.complete_json(web_messages(af, results), WEB_SCHEMA)
        except LLMError as exc:
            log.debug("web extraction failed for %s: %s", af.path.name, exc)
            return None

        duration = data.get("duration_seconds")
        return WebIdentification(
            found=bool(data.get("found")),
            confidence=_clamp(data.get("confidence")),
            reasoning=str(data.get("reasoning") or "").strip(),
            artist=_text(data.get("artist")),
            title=_text(data.get("title")),
            album=_text(data.get("album")),
            label=_text(data.get("label")),
            year=_year(data.get("year")),
            genre=_text(data.get("genre")),
            duration=float(duration) if isinstance(duration, (int, float)) and duration else None,
            source_urls=[u for u in (data.get("source_urls") or []) if isinstance(u, str)][:5],
            search_queries=[
                q for q in (data.get("search_queries") or [])
                if isinstance(q, str) and len(q.strip()) >= 3
            ][:5],
        )


def _text(v: Any) -> str | None:
    return v.strip() or None if isinstance(v, str) else None


def _year(v: Any) -> str | None:
    s = _text(v)
    return s[:4] if s and s[:4].isdigit() else None


def _clamp(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
