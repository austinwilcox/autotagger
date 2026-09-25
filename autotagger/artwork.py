"""Album art: fetch the largest rendition Apple will serve, sanity-check, embed.

Apple's search results hand back a 100x100 thumbnail URL, but the same path
serves much larger renditions — `.../100x100bb.jpg` -> `.../3000x3000bb.jpg`.
Not every release has a 3000px master, so `fetch_artwork` walks a ladder from
largest to smallest and keeps the first one that is a real image.

Pillow is optional. With it installed you get downscaling and format conversion;
without it, artwork is embedded exactly as downloaded.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

from .httpcache import Fetcher
from .providers.deezer import cover_url_ladder
from .providers.itunes import artwork_url_ladder

log = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    from PIL import Image  # type: ignore
    import io

    HAVE_PILLOW = True
except ImportError:  # pragma: no cover
    HAVE_PILLOW = False

# Anything smaller than this is not worth embedding — it will look worse than
# whatever the player generates on its own.
MIN_ACCEPTABLE_PX = 250


@dataclass
class Artwork:
    data: bytes
    mime: str
    width: int
    height: int
    source_url: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]

    def describe(self) -> str:
        return f"{self.width}x{self.height} {self.mime.split('/')[-1]} ({len(self.data) // 1024}KB)"


def fetch_artwork(
    fetcher: Fetcher,
    url: str | None,
    *,
    max_px: int | None = 1400,
    min_px: int = MIN_ACCEPTABLE_PX,
) -> Artwork | None:
    """Download the best available rendition of `url`, optionally downscaled."""
    if not url:
        return None
    for candidate_url in _ladder_for(url):
        data = fetcher.get_bytes(candidate_url)
        if not data or len(data) < 1024:
            continue
        dims = image_dimensions(data)
        if not dims:
            log.debug("not a recognizable image: %s", candidate_url)
            continue
        w, h = dims
        if max(w, h) < min_px:
            log.debug("rendition too small (%dx%d): %s", w, h, candidate_url)
            continue
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        art = Artwork(data=data, mime=mime, width=w, height=h, source_url=candidate_url)
        if max_px and max(w, h) > max_px:
            art = downscale(art, max_px) or art
        return art
    return None


def _ladder_for(url: str) -> list[str]:
    """Pick the right size ladder for whichever CDN this URL belongs to.

    Apple and Deezer both encode the requested dimensions in the path, but with
    different filename grammars, so the rewrite has to match the host.
    """
    if "dzcdn.net" in url or "deezer" in url:
        return cover_url_ladder(url)
    return artwork_url_ladder(url)


def downscale(art: Artwork, max_px: int) -> Artwork | None:
    """Shrink to `max_px` on the long edge and re-encode as JPEG. No-op without Pillow."""
    if not HAVE_PILLOW:
        return None
    try:
        img = Image.open(io.BytesIO(art.data))
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True, progressive=True)
        return Artwork(
            data=buf.getvalue(),
            mime="image/jpeg",
            width=img.width,
            height=img.height,
            source_url=art.source_url,
        )
    except Exception as exc:  # noqa: BLE001 - a failed resize must not fail the tag write
        log.debug("downscale failed: %s", exc)
        return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read width/height straight from the file header — no Pillow required."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_dimensions(data)
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    i = 2
    n = len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0-SOF15, excluding the non-frame markers DHT/JPG/DAC.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return int(w), int(h)
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        try:
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        except struct.error:
            return None
        i += 2 + seg_len
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    if chunk == b"VP8 ":
        w = int.from_bytes(data[26:28], "little") & 0x3FFF
        h = int.from_bytes(data[28:30], "little") & 0x3FFF
        return w, h
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def should_replace(existing: bytes | None, new: Artwork, *, only_if_missing: bool) -> tuple[bool, str]:
    """Decide whether to overwrite embedded art. Returns (replace?, reason)."""
    if existing is None:
        return True, f"added {new.describe()}"
    if only_if_missing:
        return False, "kept existing artwork (--artwork-if-missing)"
    if hashlib.sha256(existing).digest() == hashlib.sha256(new.data).digest():
        return False, "artwork already identical"
    old_dims = image_dimensions(existing)
    if old_dims and max(old_dims) >= max(new.width, new.height):
        return False, f"kept existing {old_dims[0]}x{old_dims[1]} (not smaller than new)"
    old_desc = f"{old_dims[0]}x{old_dims[1]}" if old_dims else "unknown size"
    return True, f"upgraded {old_desc} -> {new.describe()}"


def save_folder_image(art: Artwork, directory: Path, filename: str = "cover.jpg") -> Path | None:
    """Drop a cover file next to the album, for players that prefer folder art."""
    try:
        target = directory / filename
        target.write_bytes(art.data)
        return target
    except OSError as exc:
        log.debug("could not write %s: %s", directory / filename, exc)
        return None
