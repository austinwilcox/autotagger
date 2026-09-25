"""Apple / iTunes Search API provider.

Docs: https://performance-partners.apple.com/search-api
No key required. The public endpoint is soft-limited to roughly 20 calls per
minute per IP and answers a throttled client with HTTP 403, so the shared
Fetcher rate-limits and retries on our behalf.

Two endpoints are used:
  /search  — free text -> songs
  /lookup  — collectionId -> the full track list of an album, which is how we
             recover reliable track/disc totals and confirm an album match.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..httpcache import Fetcher
from ..models import Candidate
from ..normalize import split_credit
from .base import Provider

log = logging.getLogger(__name__)

SEARCH_URL = "https://itunes.apple.com/search"
LOOKUP_URL = "https://itunes.apple.com/lookup"

# Artwork URLs come back as ".../100x100bb.jpg". Apple will serve far larger
# renditions from the same path; try biggest first and fall back.
ARTWORK_SIZES = (3000, 1400, 1200, 600)
_SIZE_IN_URL = re.compile(r"/\d+x\d+(bb|-\d+)?\.(jpg|png)$")


class ITunesProvider(Provider):
    name = "itunes"

    def __init__(self, fetcher: Fetcher, country: str = "US", rate: float = 1.0):
        self.fetcher = fetcher
        self.country = country
        self.fetcher.set_rate("itunes.apple.com", rate)

    # -- search ------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[Candidate]:
        if not query or not query.strip():
            return []
        data = self.fetcher.get_json(
            SEARCH_URL,
            {
                "term": query.strip(),
                "media": "music",
                "entity": "song",
                "limit": str(min(limit, 200)),
                "country": self.country,
            },
        )
        if not data:
            return []
        return [
            c
            for c in (self._to_candidate(r) for r in data.get("results", []))
            if c is not None
        ]

    def lookup_album(self, collection_id: str | int, limit: int = 200) -> list[Candidate]:
        """All songs on a collection. Used to pin down track/disc totals."""
        data = self.fetcher.get_json(
            LOOKUP_URL,
            {"id": str(collection_id), "entity": "song", "limit": str(limit), "country": self.country},
        )
        if not data:
            return []
        return [
            c
            for c in (self._to_candidate(r) for r in data.get("results", []))
            if c is not None
        ]

    def lookup_artist(self, artist_id: str | int) -> str | None:
        """The name Apple files this artist under.

        This is what makes splitting a multi-artist credit safe. For a
        collaboration the track's `artistName` is "Fox Stevenson & Yue" while
        the artist entity is "Fox Stevenson" — so the remainder is a guest. For
        a band whose name merely contains an ampersand, the entity is the whole
        thing ("Simon & Garfunkel"), and nothing gets split.
        """
        data = self.fetcher.get_json(LOOKUP_URL, {"id": str(artist_id), "country": self.country})
        for result in (data or {}).get("results", []):
            if result.get("wrapperType") == "artist" and result.get("artistName"):
                return result["artistName"]
        return None

    def enrich(self, candidate: Candidate) -> Candidate:
        """Fill in trackCount/discCount from the album listing.

        The search endpoint usually supplies these, but singles and some
        territories omit them, and a wrong track total is the kind of thing that
        breaks album views in every player.
        """
        if not candidate.artists:
            artist_id = candidate.extra.get("artistId")
            primary = self.lookup_artist(artist_id) if artist_id else None
            candidate.artists = split_credit(candidate.artist, primary)

        collection_id = candidate.extra.get("collectionId")
        if not collection_id or (candidate.track_total and candidate.disc_total):
            return candidate
        siblings = self.lookup_album(collection_id)
        for sib in siblings:
            if sib.source_id == candidate.source_id:
                candidate.track_total = candidate.track_total or sib.track_total
                candidate.disc_total = candidate.disc_total or sib.disc_total
                break
        if not candidate.disc_total and siblings:
            candidate.disc_total = max((s.disc_number or 1) for s in siblings)
        return candidate

    # -- normalization -----------------------------------------------------

    def _to_candidate(self, r: dict[str, Any]) -> Candidate | None:
        if r.get("wrapperType") not in (None, "track") or r.get("kind") not in (None, "song"):
            return None
        if not r.get("trackName") or not r.get("artistName"):
            return None

        release_date = r.get("releaseDate") or ""
        explicitness = r.get("trackExplicitness")
        artwork = _upscale_artwork(r.get("artworkUrl100") or r.get("artworkUrl60"))

        return Candidate(
            source=self.name,
            source_id=str(r.get("trackId") or r.get("trackViewUrl") or r["trackName"]),
            title=r["trackName"],
            artist=r["artistName"],
            album=r.get("collectionName"),
            album_artist=r.get("collectionArtistName") or r.get("artistName"),
            track_number=r.get("trackNumber"),
            track_total=r.get("trackCount"),
            disc_number=r.get("discNumber"),
            disc_total=r.get("discCount"),
            year=release_date[:4] or None,
            release_date=release_date[:10] or None,
            genre=r.get("primaryGenreName"),
            duration=(r["trackTimeMillis"] / 1000.0) if r.get("trackTimeMillis") else None,
            explicit=(explicitness == "explicit") if explicitness else None,
            is_compilation=bool(r.get("collectionArtistName") == "Various Artists"),
            artwork_url=artwork,
            preview_url=r.get("previewUrl"),
            copyright=r.get("copyright"),
            extra={
                "collectionId": r.get("collectionId"),
                "artistId": r.get("artistId"),
                "trackViewUrl": r.get("trackViewUrl"),
                "country": r.get("country"),
                "artworkUrl100": r.get("artworkUrl100"),
            },
        )


def _upscale_artwork(url: str | None) -> str | None:
    """Rewrite a 100x100 artwork URL to the largest rendition Apple will serve."""
    if not url:
        return None
    return _SIZE_IN_URL.sub(f"/{ARTWORK_SIZES[0]}x{ARTWORK_SIZES[0]}bb.jpg", url)


def artwork_url_ladder(url: str | None) -> list[str]:
    """Every size to try, largest first, ending with the original URL."""
    if not url:
        return []
    out = [_SIZE_IN_URL.sub(f"/{s}x{s}bb.jpg", url) for s in ARTWORK_SIZES]
    if url not in out:
        out.append(url)
    seen: set[str] = set()
    return [u for u in out if not (u in seen or seen.add(u))]
