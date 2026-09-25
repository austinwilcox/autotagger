"""MusicBrainz provider — secondary source for the fields Apple does not expose.

Apple is excellent for artwork, genre and release dates, and terrible for ISRCs,
labels, catalogue numbers and stable identifiers. MusicBrainz is the reverse.
So this runs as an *enricher* by default: the Apple match decides what the track
is, then MusicBrainz is asked for the IDs that make the library portable to
Picard / Plex / Jellyfin / beets.

Their rate limit is a hard 1 request/second with a required identifying
User-Agent. Both are honoured by the shared Fetcher.
"""

from __future__ import annotations

import logging
from typing import Any

from ..httpcache import Fetcher
from ..models import Candidate
from ..normalize import normalize
from .base import Provider

log = logging.getLogger(__name__)

BASE = "https://musicbrainz.org/ws/2"


class MusicBrainzProvider(Provider):
    name = "musicbrainz"

    def __init__(self, fetcher: Fetcher, rate: float = 1.0):
        self.fetcher = fetcher
        self.fetcher.set_rate("musicbrainz.org", rate)

    def search(self, query: str, limit: int = 20) -> list[Candidate]:
        if not query or not query.strip():
            return []
        data = self.fetcher.get_json(
            f"{BASE}/recording",
            {"query": query.strip(), "limit": str(limit), "fmt": "json"},
        )
        if not data:
            return []
        out: list[Candidate] = []
        for rec in data.get("recordings", []):
            cand = self._to_candidate(rec)
            if cand:
                out.append(cand)
        return out

    def lookup_isrc(self, recording_id: str) -> str | None:
        data = self.fetcher.get_json(
            f"{BASE}/recording/{recording_id}", {"inc": "isrcs", "fmt": "json"}
        )
        isrcs = (data or {}).get("isrcs") or []
        return isrcs[0] if isrcs else None

    def annotate(self, chosen: Candidate) -> Candidate:
        """Attach MB identifiers + ISRC/label to an already-decided candidate.

        Matching here is deliberately strict: an ambiguous MusicBrainz hit is
        worse than no MusicBrainz data, because a wrong MBID silently corrupts
        every downstream tool that trusts it.
        """
        query = f'recording:"{chosen.title}" AND artist:"{chosen.artist}"'
        if chosen.album:
            query += f' AND release:"{chosen.album}"'
        results = self.search(query, limit=10)
        best = _pick_strict(chosen, results)
        if not best:
            return chosen
        chosen.musicbrainz_track_id = best.musicbrainz_track_id
        chosen.musicbrainz_album_id = best.musicbrainz_album_id
        chosen.musicbrainz_artist_id = best.musicbrainz_artist_id
        chosen.label = chosen.label or best.label
        chosen.barcode = chosen.barcode or best.barcode
        if best.musicbrainz_track_id:
            chosen.isrc = chosen.isrc or self.lookup_isrc(best.musicbrainz_track_id)
        return chosen

    def _to_candidate(self, rec: dict[str, Any]) -> Candidate | None:
        title = rec.get("title")
        credits = rec.get("artist-credit") or []
        if not title or not credits:
            return None
        artist = "".join(
            (c.get("name") or c.get("artist", {}).get("name") or "") + (c.get("joinphrase") or "")
            for c in credits
        ).strip()
        artist_id = (credits[0].get("artist") or {}).get("id")

        releases = rec.get("releases") or []
        release = releases[0] if releases else {}
        media = (release.get("media") or [{}])[0]
        track = (media.get("track") or [{}])[0]
        label_info = (release.get("label-info") or [{}])[0]

        length = rec.get("length")
        return Candidate(
            source=self.name,
            source_id=rec.get("id", ""),
            title=title,
            artist=artist,
            album=release.get("title"),
            album_artist=artist,
            track_number=track.get("number") and _int(track.get("number")),
            track_total=media.get("track-count"),
            disc_number=media.get("position"),
            year=(release.get("date") or "")[:4] or None,
            release_date=(release.get("date") or None),
            duration=(length / 1000.0) if length else None,
            musicbrainz_track_id=rec.get("id"),
            musicbrainz_album_id=release.get("id"),
            musicbrainz_artist_id=artist_id,
            label=(label_info.get("label") or {}).get("name"),
            barcode=release.get("barcode"),
            extra={"mb_score": rec.get("score")},
        )


def _int(v: Any) -> int | None:
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def _pick_strict(chosen: Candidate, results: list[Candidate]) -> Candidate | None:
    """Accept a MusicBrainz result only on an exact normalized title+artist hit."""
    want_title = normalize(chosen.title)
    want_artist = normalize(chosen.artist)
    for r in results:
        if normalize(r.title) != want_title:
            continue
        if normalize(r.artist) != want_artist:
            continue
        if chosen.duration and r.duration and abs(chosen.duration - r.duration) > 5:
            continue
        return r
    return None
