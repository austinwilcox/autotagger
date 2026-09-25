"""Acoustic fingerprinting via Chromaprint + AcoustID (optional).

This is the strongest identification signal available, because it listens to the
audio instead of reading what someone typed about it. It is optional because it
needs two extra things:

    brew install chromaprint          # provides the `fpcalc` binary
    export ACOUSTID_API_KEY=...       # free key from https://acoustid.org/new-application

When both are present, `--acoustid` turns a mangled, untagged file into a
MusicBrainz recording ID, which then seeds an exact catalogue search. When they
are absent the pipeline carries on without it — nothing here is load-bearing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .httpcache import Fetcher
from .models import AudioFile, Candidate

log = logging.getLogger(__name__)

ACOUSTID_URL = "https://api.acoustid.org/v2/lookup"


def fpcalc_available() -> bool:
    return shutil.which("fpcalc") is not None


def install_hint() -> str:
    return (
        "fpcalc not found. Install Chromaprint to enable acoustic fingerprinting:\n"
        "    brew install chromaprint        (macOS)\n"
        "    apt install libchromaprint-tools (Debian/Ubuntu)\n"
        "Then set ACOUSTID_API_KEY from https://acoustid.org/new-application"
    )


def fingerprint(path: Path, timeout: float = 60.0) -> tuple[str, int] | None:
    """Return (fingerprint, duration_seconds) for `path`, or None."""
    if not fpcalc_available():
        return None
    try:
        proc = subprocess.run(
            ["fpcalc", "-json", "-length", "120", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.debug("fpcalc failed for %s: %s", path.name, exc)
        return None
    if proc.returncode != 0:
        log.debug("fpcalc exit %s for %s: %s", proc.returncode, path.name, proc.stderr.strip())
        return None
    try:
        data = json.loads(proc.stdout)
        return data["fingerprint"], int(data["duration"])
    except (json.JSONDecodeError, KeyError):
        return None


def identify(
    fetcher: Fetcher, af: AudioFile, api_key: str | None = None, *, max_results: int = 5
) -> list[Candidate]:
    """Look the file up on AcoustID. Returns candidates carrying MusicBrainz IDs."""
    api_key = api_key or os.environ.get("ACOUSTID_API_KEY")
    if not api_key:
        log.debug("no ACOUSTID_API_KEY set; skipping fingerprint lookup")
        return []

    result = fingerprint(af.path)
    if not result:
        return []
    fp, duration = result
    af.acoustid_fingerprint = fp

    fetcher.set_rate("api.acoustid.org", 3.0)
    data = fetcher.get_json(
        ACOUSTID_URL,
        {
            "client": api_key,
            "format": "json",
            "duration": str(duration),
            "fingerprint": fp,
            "meta": "recordings+releasegroups+compress",
        },
    )
    if not data or data.get("status") != "ok":
        return []

    out: list[Candidate] = []
    for res in sorted(data.get("results", []), key=lambda r: r.get("score", 0), reverse=True):
        for rec in res.get("recordings", [])[:max_results]:
            rec_id = rec.get("id")
            if rec_id:
                af.acoustid_recording_ids.append(rec_id)
            artists = rec.get("artists") or []
            artist = ", ".join(a.get("name", "") for a in artists if a.get("name"))
            title = rec.get("title")
            if not title or not artist:
                continue
            groups = rec.get("releasegroups") or []
            album = groups[0].get("title") if groups else None
            out.append(
                Candidate(
                    source="acoustid",
                    source_id=rec_id or title,
                    title=title,
                    artist=artist,
                    album=album,
                    duration=rec.get("duration"),
                    musicbrainz_track_id=rec_id,
                    musicbrainz_album_id=groups[0].get("id") if groups else None,
                    extra={"acoustid_score": res.get("score")},
                )
            )
        if len(out) >= max_results:
            break
    return out[:max_results]


def seed_queries(candidates: list[Candidate]) -> list[str]:
    """Turn AcoustID hits into precise catalogue search strings."""
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        q = f"{c.artist} {c.title}"
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out
