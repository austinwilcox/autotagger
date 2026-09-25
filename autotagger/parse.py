"""Infer metadata from the file path when tags are missing or garbage.

A surprising amount of signal lives in the directory layout. The common rips
look like:

    Artist/Album (Year)/03 - Title.mp3
    Artist - Album/03 Title.flac
    Music/Various Artists/Now 42/1-05 Artist - Title.m4a
    downloads/Artist - Title (Official Video).mp3

`enrich_from_path` fills only the `guessed_*` fields — it never overwrites real
tags, so a file with good tags and a bad path is unharmed.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import AudioFile
from .normalize import strip_leading_tracknum

# Junk that scrapers and YouTube rippers bolt onto filenames.
_FILENAME_JUNK = re.compile(
    r"""\s*[\(\[]?\s*\b(
        official(\s+(music\s+)?video|\s+audio|\s+lyric\s+video)?
      | lyrics?(\s+video)?
      | audio
      | hq | hd | 4k | 320\s*kbps | \d{3}kbps
      | free\s+download
      | visuali[sz]er
      | mv
      | full\s+album
      | youtube | yt | soundcloud | spotify
    )\b\s*[\)\]]?""",
    re.IGNORECASE | re.VERBOSE,
)

# "Artist - Title" — the single most common filename convention.
_DASH_SPLIT = re.compile(r"\s+[-–—]\s+")

_YEAR_IN_FOLDER = re.compile(r"[\(\[\-\s](19\d{2}|20\d{2})[\)\]\s]?")

_VARIOUS = re.compile(r"^(various(\s+artists)?|va|compilation|soundtrack|ost)$", re.IGNORECASE)

_NONMUSIC_DIRS = {
    "music", "musik", "musique", "media", "audio", "songs", "itunes",
    "itunes media", "library", "downloads", "download", "downloaded",
    "desktop", "documents", "volumes", "users", "home", "mnt", "srv",
    "incoming", "new", "unsorted", "untagged", "to tag", "temp", "tmp",
}


def enrich_from_path(af: AudioFile, library_root: Path | None = None) -> AudioFile:
    """Populate `guessed_*` fields on an AudioFile from its path. Mutates and returns it."""
    stem = _clean_stem(af.path.stem)
    rest, track_no, disc_no = strip_leading_tracknum(stem)
    af.guessed_track_number = track_no
    af.guessed_disc_number = disc_no

    artist, title = _split_artist_title(rest)
    af.guessed_title = title or rest or None
    af.guessed_artist = artist

    parents = _meaningful_parents(af.path, library_root)
    if parents:
        album_dir = parents[0]
        album, year = _strip_year(album_dir)
        af.guessed_album = album or None
        if year and not af.year:
            af.path_hints.append(f"folder year: {year}")
    if len(parents) > 1 and not af.guessed_artist:
        parent_artist = parents[1]
        if not _VARIOUS.match(parent_artist):
            af.guessed_artist = parent_artist

    # "Artist - Album" as a single folder name.
    if af.guessed_album and not af.guessed_artist:
        bits = _DASH_SPLIT.split(af.guessed_album, maxsplit=1)
        if len(bits) == 2:
            af.guessed_artist, af.guessed_album = bits[0].strip(), bits[1].strip()

    af.path_hints.extend(f"dir: {p}" for p in parents[:3])
    if af.guessed_disc_number is None:
        disc = _disc_from_folder(parents)
        if disc:
            af.guessed_disc_number = disc
            af.path_hints.append(f"disc folder: {disc}")
    return af


def _clean_stem(stem: str) -> str:
    s = stem.replace("_", " ")
    s = _FILENAME_JUNK.sub(" ", s)
    s = re.sub(r"\s*[\(\[]\s*[\)\]]", " ", s)      # empty brackets left behind
    s = re.sub(r"\s{2,}", " ", s).strip(" -–—.")
    return s


def _split_artist_title(text: str) -> tuple[str | None, str | None]:
    """Resolve "Artist - Title". Returns (artist, title); artist may be None."""
    if not text:
        return None, None
    parts = _DASH_SPLIT.split(text)
    if len(parts) == 2:
        left, right = (p.strip() for p in parts)
        if left and right:
            return left, right
    if len(parts) > 2:
        # "Artist - Album - Title" — assume the last chunk is the title.
        return parts[0].strip(), parts[-1].strip()
    return None, text.strip()


def _meaningful_parents(path: Path, library_root: Path | None) -> list[str]:
    """Directory names from closest to furthest, skipping generic containers."""
    parents = []
    for parent in path.parents:
        if library_root and parent == library_root:
            break
        name = parent.name
        if not name or name in ("/", ""):
            break
        if name.lower().strip() in _NONMUSIC_DIRS:
            break
        parents.append(name)
        if len(parents) >= 4:
            break
    return parents


def _strip_year(folder: str) -> tuple[str, str | None]:
    m = _YEAR_IN_FOLDER.search(folder)
    if not m:
        return folder.strip(), None
    cleaned = (folder[: m.start()] + " " + folder[m.end():]).strip(" -–—()[]")
    return re.sub(r"\s{2,}", " ", cleaned), m.group(1)


def _disc_from_folder(parents: list[str]) -> int | None:
    for name in parents[:2]:
        m = re.fullmatch(r"(?:cd|disc|disk)\s*[-_]?\s*(\d{1,2})", name.strip(), re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def looks_like_various_artists(album_artist: str | None) -> bool:
    return bool(album_artist and _VARIOUS.match(album_artist.strip()))
