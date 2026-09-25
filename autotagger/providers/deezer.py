"""Deezer provider.

Free, no API key, and markedly better than Apple for dance/electronic — which is
where a hardstyle or hardcore release is most likely to exist at all. It also
returns more per result than Apple does:

  * `duration` in whole seconds, so the strongest verification signal survives
  * `isrc` directly in the search response (Apple makes you go to MusicBrainz)
  * `label` and `upc` from the album endpoint
  * `bpm` from the track endpoint — useful metadata Apple simply does not have

Rate limit is 50 requests per 5 seconds per IP; the default here is far below it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..httpcache import Fetcher
from ..models import Candidate
from ..normalize import _dedupe_names, split_credit
from .base import Provider

log = logging.getLogger(__name__)

BASE = "https://api.deezer.com"

# Deezer covers are served from a path whose dimensions are editable, the same
# trick the iTunes provider uses. 1800 is the largest reliably available.
ARTWORK_SIZES = (1800, 1400, 1000)
_COVER_SIZE = re.compile(r"/\d+x\d+(-[\d-]+)?\.jpg$")


class DeezerProvider(Provider):
    name = "deezer"

    def __init__(self, fetcher: Fetcher, rate: float = 5.0):
        self.fetcher = fetcher
        self.fetcher.set_rate("api.deezer.com", rate)

    def search(self, query: str, limit: int = 25) -> list[Candidate]:
        if not query or not query.strip():
            return []
        data = self.fetcher.get_json(
            f"{BASE}/search", {"q": query.strip(), "limit": str(min(limit, 100))}
        )
        if not data or not data.get("data"):
            return []
        return [c for c in (self._to_candidate(t) for t in data["data"]) if c]

    def search_structured(self, artist: str | None, track: str | None, album: str | None = None,
                          limit: int = 25) -> list[Candidate]:
        """Deezer's field-scoped query syntax — far more precise than free text."""
        parts = []
        if artist:
            parts.append(f'artist:"{artist}"')
        if track:
            parts.append(f'track:"{track}"')
        if album:
            parts.append(f'album:"{album}"')
        return self.search(" ".join(parts), limit=limit) if parts else []

    def enrich(self, candidate: Candidate) -> Candidate:
        """Pull track position, disc, release date and label from the detail endpoints.

        Two extra requests, so this only runs on the candidate that actually won.
        """
        track_id = candidate.extra.get("deezer_track_id")
        if not track_id:
            return candidate

        track = self.fetcher.get_json(f"{BASE}/track/{track_id}") or {}

        # Deezer credits each act separately, which removes all guesswork: a
        # single entry means one act however many ampersands its name contains.
        contributors = [
            c.get("name") for c in (track.get("contributors") or []) if c.get("name")
        ]
        if contributors:
            candidate.artists = _dedupe_names(contributors)
        elif not candidate.artists:
            candidate.artists = split_credit(candidate.artist)
        candidate.track_number = candidate.track_number or track.get("track_position")
        candidate.disc_number = candidate.disc_number or track.get("disk_number")
        candidate.isrc = candidate.isrc or track.get("isrc")
        if track.get("release_date"):
            candidate.release_date = track["release_date"]
            candidate.year = track["release_date"][:4]
        # Deezer reports 0 for "unknown BPM" rather than omitting the field.
        if track.get("bpm"):
            candidate.extra["bpm"] = round(float(track["bpm"]))

        album_id = (track.get("album") or {}).get("id") or candidate.extra.get("deezer_album_id")
        if album_id:
            album = self.fetcher.get_json(f"{BASE}/album/{album_id}") or {}
            candidate.track_total = candidate.track_total or album.get("nb_tracks")
            candidate.label = candidate.label or album.get("label")
            candidate.barcode = candidate.barcode or album.get("upc")
            genres = ((album.get("genres") or {}).get("data") or [])
            if genres and not candidate.genre:
                candidate.genre = genres[0].get("name")
            if album.get("release_date") and not candidate.release_date:
                candidate.release_date = album["release_date"]
                candidate.year = album["release_date"][:4]
            if album.get("artist", {}).get("name"):
                candidate.album_artist = candidate.album_artist or album["artist"]["name"]
        return candidate

    def _to_candidate(self, t: dict[str, Any]) -> Candidate | None:
        title = t.get("title") or t.get("title_short")
        artist = (t.get("artist") or {}).get("name")
        if not title or not artist:
            return None
        album = t.get("album") or {}
        duration = t.get("duration")
        return Candidate(
            source=self.name,
            source_id=str(t.get("id") or title),
            title=title,
            artist=artist,
            album=album.get("title"),
            duration=float(duration) if duration else None,
            explicit=t.get("explicit_lyrics"),
            isrc=t.get("isrc") or None,
            artwork_url=_upscale_cover(album.get("cover_xl") or album.get("cover_big")),
            preview_url=t.get("preview"),
            extra={
                "deezer_track_id": t.get("id"),
                "deezer_album_id": album.get("id"),
                "link": t.get("link"),
                "rank": t.get("rank"),
            },
        )


def _upscale_cover(url: str | None) -> str | None:
    if not url:
        return None
    return _COVER_SIZE.sub(f"/{ARTWORK_SIZES[0]}x{ARTWORK_SIZES[0]}-000000-80-0-0.jpg", url)


def cover_url_ladder(url: str | None) -> list[str]:
    if not url:
        return []
    out = [_COVER_SIZE.sub(f"/{s}x{s}-000000-80-0-0.jpg", url) for s in ARTWORK_SIZES]
    if url not in out:
        out.append(url)
    seen: set[str] = set()
    return [u for u in out if not (u in seen or seen.add(u))]
