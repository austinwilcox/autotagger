"""Reading and writing tags across the formats people actually have on disk.

mutagen exposes a different object model per container (ID3 frames, MP4 atoms,
Vorbis comments). Rather than sprinkle `isinstance` checks through the pipeline,
everything funnels through two functions here:

    read_audio_file(path) -> AudioFile      (uniform evidence bundle)
    write_tags(path, values, artwork=...)   (uniform write)

`values` is always the same flat dict of canonical field names, listed in
`CANONICAL_FIELDS`. Each format adapter knows how to spell those.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import mutagen
from mutagen.aiff import AIFF
from mutagen.flac import FLAC, Picture
from mutagen.id3 import (
    APIC,
    ID3,
    ID3NoHeaderError,
    TALB,
    TCMP,
    TCOM,
    TCON,
    TCOP,
    TDRC,
    TIT2,
    TPE1,
    TPE2,
    TPOS,
    TRCK,
    TSRC,
    TXXX,
)
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from .models import AudioFile

log = logging.getLogger(__name__)

CANONICAL_FIELDS = (
    "title",
    "artist",
    "album",
    "album_artist",
    "track_number",
    "track_total",
    "disc_number",
    "disc_total",
    "year",
    "date",
    "genre",
    "composer",
    "compilation",
    "isrc",
    "copyright",
    "explicit",
    "musicbrainz_track_id",
    "musicbrainz_album_id",
    "musicbrainz_artist_id",
)

AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".mp4", ".m4b", ".flac", ".ogg", ".oga",
    ".opus", ".wav", ".aiff", ".aif", ".aifc",
}

_ID3_EXTS = {".mp3", ".wav", ".aiff", ".aif", ".aifc"}
_MP4_EXTS = {".m4a", ".mp4", ".m4b"}
_VORBIS_EXTS = {".flac", ".ogg", ".oga", ".opus"}


class UnsupportedFormat(Exception):
    pass


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load(path: Path):
    ext = path.suffix.lower()
    if ext == ".mp3":
        return MP3(path)
    if ext in _MP4_EXTS:
        return MP4(path)
    if ext == ".flac":
        return FLAC(path)
    if ext in (".ogg", ".oga"):
        return OggVorbis(path)
    if ext == ".opus":
        return OggOpus(path)
    if ext == ".wav":
        return WAVE(path)
    if ext in (".aiff", ".aif", ".aifc"):
        return AIFF(path)
    obj = mutagen.File(path)
    if obj is None:
        raise UnsupportedFormat(f"mutagen cannot open {path.name}")
    return obj


def _all(value: Any) -> list[str]:
    """Every value of a possibly multi-valued tag, as clean strings."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    out: list[str] = []
    for item in value:
        if isinstance(item, bytes):
            item = item.decode("utf-8", "replace")
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _first(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _parse_pair(value: str | None) -> tuple[int | None, int | None]:
    """Parse "3/12" (and bare "3") into (3, 12)."""
    if not value:
        return None, None
    s = str(value).strip()
    if "/" in s:
        a, _, b = s.partition("/")
        return _to_int(a), _to_int(b)
    return _to_int(s), None


def _to_int(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_audio_file(path: Path) -> AudioFile:
    """Read tags + stream properties into the uniform evidence bundle."""
    audio = _load(path)
    af = AudioFile(path=path, ext=path.suffix.lower())

    info = getattr(audio, "info", None)
    if info is not None:
        af.duration = getattr(info, "length", None)
        af.bitrate = getattr(info, "bitrate", None)
        af.sample_rate = getattr(info, "sample_rate", None)
        af.channels = getattr(info, "channels", None)

    ext = path.suffix.lower()
    if ext in _MP4_EXTS:
        _read_mp4(audio, af)
    elif ext in _VORBIS_EXTS:
        _read_vorbis(audio, af)
    else:
        _read_id3(audio, af, path)
    return af


def _read_id3(audio, af: AudioFile, path: Path) -> None:
    tags = getattr(audio, "tags", None)
    if tags is None:
        try:
            tags = ID3(path)
        except (ID3NoHeaderError, Exception):  # noqa: BLE001 - untagged file is fine
            return
    g: Callable[[str], str | None] = lambda k: _first(tags.get(k).text) if tags.get(k) else None
    af.title = g("TIT2")
    af.artist = g("TPE1")
    tpe1 = tags.get("TPE1")
    af.artists = _all(tpe1.text) if tpe1 else []
    for frame in tags.getall("TXXX"):
        if str(frame.desc).upper() == "ARTISTS":
            af.artists = _all(frame.text) or af.artists
    af.album = g("TALB")
    af.album_artist = g("TPE2")
    af.genre = g("TCON")
    af.composer = g("TCOM")
    af.date = g("TDRC") or g("TYER")
    af.year = (af.date or "")[:4] or None
    af.isrc = g("TSRC")
    af.copyright = g("TCOP")
    tcmp = g("TCMP")
    af.compilation = True if tcmp in ("1", "true", "True") else None
    af.track_number, af.track_total = _parse_pair(g("TRCK"))
    af.disc_number, af.disc_total = _parse_pair(g("TPOS"))
    for frame in tags.getall("TXXX"):
        if frame.desc == "MusicBrainz Release Track Id":
            af.musicbrainz_track_id = _first(frame.text)
    af.has_artwork = bool(tags.getall("APIC"))
    _collect_extra_id3(tags, af)


def _read_mp4(audio: MP4, af: AudioFile) -> None:
    t = audio.tags or {}
    af.title = _first(t.get("\xa9nam"))
    af.artist = _first(t.get("\xa9ART"))
    af.artists = _all(t.get("----:com.apple.iTunes:ARTISTS")) or _all(t.get("\xa9ART"))
    af.album = _first(t.get("\xa9alb"))
    af.album_artist = _first(t.get("aART"))
    af.genre = _first(t.get("\xa9gen"))
    af.composer = _first(t.get("\xa9wrt"))
    af.date = _first(t.get("\xa9day"))
    af.year = (af.date or "")[:4] or None
    af.isrc = _first(t.get("----:com.apple.iTunes:ISRC"))
    af.copyright = _first(t.get("cprt"))
    af.compilation = True if t.get("cpil") else None
    rtng = t.get("rtng")
    if rtng:
        # Apple's rating atom: 1 = explicit, 2 = clean, 0 = none.
        af.explicit = {1: True, 2: False}.get(int(rtng[0]))
    if t.get("trkn"):
        af.track_number, af.track_total = (list(t["trkn"][0]) + [None, None])[:2]
    if t.get("disk"):
        af.disc_number, af.disc_total = (list(t["disk"][0]) + [None, None])[:2]
    af.musicbrainz_track_id = _first(t.get("----:com.apple.iTunes:MusicBrainz Track Id"))
    af.has_artwork = bool(t.get("covr"))
    _collect_extra_mp4(t, af)


def _read_vorbis(audio, af: AudioFile) -> None:
    t = audio.tags or {}
    get = lambda k: _first(t.get(k))  # noqa: E731
    af.title = get("title")
    af.artist = get("artist")
    af.artists = _all(t.get("artists")) or _all(t.get("artist"))
    af.album = get("album")
    af.album_artist = get("albumartist")
    af.genre = get("genre")
    af.composer = get("composer")
    af.date = get("date") or get("year")
    af.year = (af.date or "")[:4] or None
    af.isrc = get("isrc")
    af.copyright = get("copyright")
    af.compilation = True if get("compilation") in ("1", "true", "True") else None
    af.track_number, af.track_total = _parse_pair(get("tracknumber"))
    af.track_total = af.track_total or _to_int(get("tracktotal"))
    af.disc_number, af.disc_total = _parse_pair(get("discnumber"))
    af.disc_total = af.disc_total or _to_int(get("disctotal"))
    af.musicbrainz_track_id = get("musicbrainz_trackid")
    if isinstance(audio, FLAC):
        af.has_artwork = bool(audio.pictures)
    else:
        af.has_artwork = bool(t.get("metadata_block_picture"))
    _collect_extra_vorbis(t, af)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_tags(
    path: Path,
    values: dict[str, Any],
    artwork: bytes | None = None,
    artwork_mime: str = "image/jpeg",
    replace_artwork: bool = True,
    clear_artwork: bool = False,
) -> None:
    """Write canonical `values` (and optionally cover art) into `path`.

    Only keys present in `values` are touched; a key mapped to None clears that
    tag. Untouched tags survive, which matters for anything the user hand-curated
    (ratings, play counts, custom comments).

    `clear_artwork` removes embedded covers — needed by `undo`, which must put a
    file back to having no artwork if it started that way.
    """
    ext = path.suffix.lower()
    args = (path, values, artwork, artwork_mime, replace_artwork, clear_artwork)
    if ext in _MP4_EXTS:
        _write_mp4(*args)
    elif ext in _VORBIS_EXTS:
        _write_vorbis(*args)
    elif ext in _ID3_EXTS:
        _write_id3(*args)
    else:
        raise UnsupportedFormat(f"no tag writer for {ext}")


def _pair(num: Any, total: Any) -> str | None:
    if num in (None, ""):
        return None
    return f"{int(num)}/{int(total)}" if total else str(int(num))


def _open_id3(path: Path):
    """Return (id3_tags, save_callable) for an ID3-carrying container.

    MP3 stores ID3 as a bare prefix, so `ID3(path).save(path)` is correct there.
    AIFF and WAV store it *inside* an IFF/RIFF chunk — writing a bare ID3 header
    over the top of those corrupts the file, so they have to go through their
    container object.
    """
    ext = path.suffix.lower()
    if ext == ".mp3":
        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        return tags, lambda version=3: tags.save(path, v2_version=version)

    audio = WAVE(path) if ext == ".wav" else AIFF(path)
    if audio.tags is None:
        audio.add_tags()

    def save(version: int = 3):
        try:
            return audio.save(v2_version=version)
        except TypeError:
            return audio.save()

    return audio.tags, save


def _write_id3(path, values, artwork, mime, replace_artwork, clear_artwork=False) -> None:
    tags, save = _open_id3(Path(path))

    # ID3v2.3 has no way to separate multiple values in a text frame — mutagen
    # joins them with "/", which any artist named like "AC/DC" then breaks. v2.4
    # separates with NUL, so a genuine list forces the newer version.
    artist_values = _all(values.get("artist")) if "artist" in values else []
    version = 4 if len(artist_values) > 1 else 3

    simple = {
        "title": TIT2, "artist": TPE1, "album": TALB, "album_artist": TPE2,
        "genre": TCON, "composer": TCOM, "isrc": TSRC, "copyright": TCOP,
    }
    for key, frame_cls in simple.items():
        if key not in values:
            continue
        texts = _all(values[key])
        tags.delall(frame_cls.__name__)
        if texts:
            tags.add(frame_cls(encoding=3, text=texts))

    if "year" in values or "date" in values:
        tags.delall("TDRC")
        tags.delall("TYER")
        stamp = values.get("date") or values.get("year")
        if stamp:
            tags.add(TDRC(encoding=3, text=[str(stamp)]))

    if "track_number" in values or "track_total" in values:
        tags.delall("TRCK")
        pair = _pair(values.get("track_number"), values.get("track_total"))
        if pair:
            tags.add(TRCK(encoding=3, text=[pair]))

    if "disc_number" in values or "disc_total" in values:
        tags.delall("TPOS")
        pair = _pair(values.get("disc_number"), values.get("disc_total"))
        if pair:
            tags.add(TPOS(encoding=3, text=[pair]))

    if "compilation" in values:
        tags.delall("TCMP")
        if values["compilation"]:
            tags.add(TCMP(encoding=3, text=["1"]))

    _write_id3_txxx(tags, values)

    # Picard's convention, which Jellyfin/Navidrome/MusicBee also read.
    if "artist" in values:
        tags.delall("TXXX:ARTISTS")
        if len(artist_values) > 1:
            tags.add(TXXX(encoding=3, desc="ARTISTS", text=artist_values))

    if clear_artwork:
        tags.delall("APIC")
    elif artwork:
        if replace_artwork:
            tags.delall("APIC")
        tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=artwork))

    save(version)


def _write_id3_txxx(tags: ID3, values: dict[str, Any]) -> None:
    """MusicBrainz IDs live in TXXX frames under exact, Picard-compatible names."""
    mb = {
        "musicbrainz_track_id": "MusicBrainz Release Track Id",
        "musicbrainz_album_id": "MusicBrainz Album Id",
        "musicbrainz_artist_id": "MusicBrainz Artist Id",
    }
    for key, desc in mb.items():
        if key not in values:
            continue
        for frame in list(tags.getall("TXXX")):
            if frame.desc == desc:
                tags.delall(f"TXXX:{desc}")
        if values[key]:
            tags.add(TXXX(encoding=3, desc=desc, text=[str(values[key])]))


def _write_mp4(path, values, artwork, mime, replace_artwork, clear_artwork=False) -> None:
    audio = MP4(path)
    if audio.tags is None:
        audio.add_tags()
    t = audio.tags

    simple = {
        "title": "\xa9nam", "artist": "\xa9ART", "album": "\xa9alb",
        "album_artist": "aART", "genre": "\xa9gen", "composer": "\xa9wrt",
        "copyright": "cprt",
    }
    for key, atom in simple.items():
        if key not in values:
            continue
        texts = _all(values[key])
        if texts:
            t[atom] = texts
        else:
            t.pop(atom, None)

    if "artist" in values:
        artist_values = _all(values["artist"])
        atom = "----:com.apple.iTunes:ARTISTS"
        if len(artist_values) > 1:
            t[atom] = [a.encode("utf-8") for a in artist_values]
        else:
            t.pop(atom, None)

    if "year" in values or "date" in values:
        stamp = values.get("date") or values.get("year")
        if stamp:
            t["\xa9day"] = [str(stamp)]
        else:
            t.pop("\xa9day", None)

    if "track_number" in values or "track_total" in values:
        n, total = values.get("track_number"), values.get("track_total")
        if n:
            t["trkn"] = [(int(n), int(total or 0))]
        else:
            t.pop("trkn", None)

    if "disc_number" in values or "disc_total" in values:
        n, total = values.get("disc_number"), values.get("disc_total")
        if n:
            t["disk"] = [(int(n), int(total or 0))]
        else:
            t.pop("disk", None)

    if "compilation" in values:
        t["cpil"] = bool(values["compilation"])

    if "explicit" in values and values["explicit"] is not None:
        # Apple's rating atom: 1 = explicit, 2 = clean, 0 = none.
        t["rtng"] = [1 if values["explicit"] else 2]

    freeform = {
        "isrc": "----:com.apple.iTunes:ISRC",
        "musicbrainz_track_id": "----:com.apple.iTunes:MusicBrainz Track Id",
        "musicbrainz_album_id": "----:com.apple.iTunes:MusicBrainz Album Id",
        "musicbrainz_artist_id": "----:com.apple.iTunes:MusicBrainz Artist Id",
    }
    for key, atom in freeform.items():
        if key not in values:
            continue
        if values[key]:
            t[atom] = [str(values[key]).encode("utf-8")]
        else:
            t.pop(atom, None)

    if clear_artwork:
        t.pop("covr", None)
    elif artwork:
        fmt = MP4Cover.FORMAT_PNG if mime == "image/png" else MP4Cover.FORMAT_JPEG
        covers = [] if replace_artwork else list(t.get("covr", []))
        covers.append(MP4Cover(artwork, imageformat=fmt))
        t["covr"] = covers

    audio.save()


def _write_vorbis(path, values, artwork, mime, replace_artwork, clear_artwork=False) -> None:
    audio = _load(path)
    if audio.tags is None:
        audio.add_tags()

    simple = {
        "title": "title", "artist": "artist", "album": "album",
        "album_artist": "albumartist", "genre": "genre", "composer": "composer",
        "isrc": "isrc", "copyright": "copyright",
        "musicbrainz_track_id": "musicbrainz_trackid",
        "musicbrainz_album_id": "musicbrainz_albumid",
        "musicbrainz_artist_id": "musicbrainz_artistid",
    }
    for key, name in simple.items():
        if key not in values:
            continue
        texts = _all(values[key])
        if texts:
            audio[name] = texts
        else:
            audio.pop(name, None)

    if "artist" in values:
        artist_values = _all(values["artist"])
        if len(artist_values) > 1:
            audio["artists"] = artist_values
        else:
            audio.pop("artists", None)

    if "year" in values or "date" in values:
        stamp = values.get("date") or values.get("year")
        if stamp:
            audio["date"] = [str(stamp)]
        else:
            audio.pop("date", None)

    for num_key, total_key, num_name, total_name in (
        ("track_number", "track_total", "tracknumber", "tracktotal"),
        ("disc_number", "disc_total", "discnumber", "disctotal"),
    ):
        if num_key in values:
            if values[num_key]:
                audio[num_name] = [str(int(values[num_key]))]
            else:
                audio.pop(num_name, None)
        if total_key in values:
            if values[total_key]:
                audio[total_name] = [str(int(values[total_key]))]
            else:
                audio.pop(total_name, None)

    if "compilation" in values:
        if values["compilation"]:
            audio["compilation"] = ["1"]
        else:
            audio.pop("compilation", None)

    if clear_artwork:
        if isinstance(audio, FLAC):
            audio.clear_pictures()
        else:
            audio.pop("metadata_block_picture", None)
    elif artwork:
        _attach_vorbis_picture(audio, artwork, mime, replace_artwork)

    audio.save()


def _attach_vorbis_picture(audio, data: bytes, mime: str, replace: bool) -> None:
    pic = Picture()
    pic.data = data
    pic.type = 3  # front cover
    pic.mime = mime
    pic.desc = "Cover"
    if isinstance(audio, FLAC):
        if replace:
            audio.clear_pictures()
        audio.add_picture(pic)
        return
    # Ogg/Opus carry the picture as a base64 Vorbis comment.
    import base64

    encoded = base64.b64encode(pic.write()).decode("ascii")
    existing = [] if replace else list(audio.get("metadata_block_picture", []))
    audio["metadata_block_picture"] = existing + [encoded]


def read_embedded_artwork(path: Path) -> bytes | None:
    """Return the existing front-cover bytes, if any."""
    try:
        audio = _load(path)
    except Exception:  # noqa: BLE001
        return None
    ext = path.suffix.lower()
    if ext in _MP4_EXTS:
        covers = (audio.tags or {}).get("covr") or []
        return bytes(covers[0]) if covers else None
    if isinstance(audio, FLAC):
        return audio.pictures[0].data if audio.pictures else None
    if ext in _VORBIS_EXTS:
        import base64

        blocks = (audio.tags or {}).get("metadata_block_picture") or []
        if not blocks:
            return None
        return Picture(base64.b64decode(blocks[0])).data
    tags = getattr(audio, "tags", None)
    if tags is None:
        return None
    apics = tags.getall("APIC")
    return apics[0].data if apics else None


# --------------------------------------------------------------------------
# Extended metadata
# --------------------------------------------------------------------------
# Deliberately read-only. These fields are frequently the only thing that
# identifies an independent or label release — a hardstyle track from a label
# compilation will often name the label in `publisher` even when title and
# artist are blank.

_ID3_EXTRA = {
    "TENC": "encoder", "TSSE": "encoder_settings", "TPUB": "publisher",
    "TBPM": "bpm", "TKEY": "initial_key", "TIT1": "grouping",
    "TCOP": "copyright", "TSRC": "isrc", "TLAN": "language",
    "TMED": "media", "TOAL": "original_album", "TOPE": "original_artist",
    "TDOR": "original_release_date", "TSOP": "artist_sort",
}


def _collect_extra_id3(tags, af: AudioFile) -> None:
    for frame_id, name in _ID3_EXTRA.items():
        frame = tags.get(frame_id)
        value = _first(getattr(frame, "text", None)) if frame else None
        if value:
            af.extra_tags[name] = value
    for comm in tags.getall("COMM"):
        value = _first(comm.text)
        if value:
            af.extra_tags[f"comment:{comm.desc or 'default'}"[:40]] = value
    for wxxx in tags.getall("WXXX"):
        if getattr(wxxx, "url", None):
            af.extra_tags[f"url:{wxxx.desc or 'default'}"[:40]] = wxxx.url
    for txxx in tags.getall("TXXX"):
        value = _first(txxx.text)
        if value and txxx.desc and not txxx.desc.startswith("MusicBrainz"):
            af.extra_tags[str(txxx.desc)[:40]] = value


_MP4_EXTRA = {
    "\xa9too": "encoder", "\xa9cmt": "comment", "\xa9grp": "grouping",
    "tmpo": "bpm", "cprt": "copyright", "\xa9lyr": "lyrics_present",
    "----:com.apple.iTunes:ISRC": "isrc",
    "----:com.apple.iTunes:LABEL": "publisher",
    "----:com.apple.iTunes:initialkey": "initial_key",
}


def _collect_extra_mp4(t, af: AudioFile) -> None:
    for atom, name in _MP4_EXTRA.items():
        value = t.get(atom)
        if not value:
            continue
        raw = value[0] if isinstance(value, list) else value
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        text = str(raw).strip()
        if text:
            af.extra_tags[name] = text[:300]


_VORBIS_EXTRA = (
    "encoder", "encoded_by", "comment", "description", "publisher", "label",
    "organization", "bpm", "initialkey", "key", "grouping", "copyright",
    "isrc", "contact", "www", "url", "license", "originaldate",
)


def _collect_extra_vorbis(t, af: AudioFile) -> None:
    for name in _VORBIS_EXTRA:
        value = _first(t.get(name))
        if value:
            af.extra_tags[name] = value[:300]
