"""Tag round-trip across every container we claim to support.

Generated with ffmpeg rather than committed as binary fixtures. The whole point
is to catch a per-format writer regression — a field that silently does not
persist in MP4 or Ogg is exactly the bug that survives unit tests of the
matcher.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autotagger.audio import read_audio_file, read_embedded_artwork, write_tags

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is needed to synthesize fixtures"
)

# 1x1 red PNG — small enough to inline, real enough for every embedder.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000c4944415408d763f8cfc000000301010018dd8db00000000049454e44ae426082"
)

FORMATS = [
    ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    ("m4a", ["-c:a", "aac", "-b:a", "64k"]),
    ("flac", ["-c:a", "flac"]),
    ("ogg", ["-c:a", "libvorbis"]),
    ("opus", ["-c:a", "libopus"]),
    ("aiff", ["-c:a", "pcm_s16be"]),
]

VALUES = {
    "title": "Exit Music (For a Film)",
    "artist": "Radiohead",
    "album": "OK Computer",
    "album_artist": "Radiohead",
    "track_number": 4,
    "track_total": 12,
    "disc_number": 1,
    "disc_total": 1,
    "year": "1997",
    "genre": "Alternative",
    "composer": "Thom Yorke",
}


def synthesize(path: Path, codec_args: list[str], seconds: float = 2.0) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds),
         *codec_args, str(path)],
        capture_output=True,
    )
    return proc.returncode == 0 and path.exists()


@pytest.mark.parametrize("ext,codec_args", FORMATS, ids=[f[0] for f in FORMATS])
def test_tag_roundtrip(tmp_path: Path, ext: str, codec_args: list[str]):
    path = tmp_path / f"sample.{ext}"
    if not synthesize(path, codec_args):
        pytest.skip(f"ffmpeg build cannot encode {ext}")

    write_tags(path, VALUES, artwork=PNG, artwork_mime="image/png")
    tags = read_audio_file(path)

    assert tags.title == VALUES["title"]
    assert tags.artist == VALUES["artist"]
    assert tags.album == VALUES["album"]
    assert tags.album_artist == VALUES["album_artist"]
    assert tags.track_number == 4
    assert tags.track_total == 12
    assert tags.disc_number == 1
    assert tags.year == "1997"
    assert tags.genre == VALUES["genre"]
    assert tags.composer == VALUES["composer"]
    assert tags.duration and 1.5 < tags.duration < 2.5

    assert tags.has_artwork
    assert read_embedded_artwork(path) == PNG


@pytest.mark.parametrize("ext,codec_args", FORMATS, ids=[f[0] for f in FORMATS])
def test_clear_artwork(tmp_path: Path, ext: str, codec_args: list[str]):
    path = tmp_path / f"sample.{ext}"
    if not synthesize(path, codec_args):
        pytest.skip(f"ffmpeg build cannot encode {ext}")

    write_tags(path, VALUES, artwork=PNG, artwork_mime="image/png")
    assert read_audio_file(path).has_artwork

    write_tags(path, {}, clear_artwork=True)
    assert not read_audio_file(path).has_artwork
    assert read_embedded_artwork(path) is None


@pytest.mark.parametrize("ext,codec_args", FORMATS, ids=[f[0] for f in FORMATS])
def test_partial_write_leaves_other_tags_alone(tmp_path: Path, ext: str, codec_args: list[str]):
    path = tmp_path / f"sample.{ext}"
    if not synthesize(path, codec_args):
        pytest.skip(f"ffmpeg build cannot encode {ext}")

    write_tags(path, VALUES)
    write_tags(path, {"genre": "Rock"})
    tags = read_audio_file(path)
    assert tags.genre == "Rock"
    assert tags.title == VALUES["title"]      # untouched
    assert tags.track_number == 4             # untouched


def test_extended_metadata_is_read_as_evidence(tmp_path: Path):
    """Encoder / publisher / comment identify releases the core fields cannot."""
    path = tmp_path / "sample.mp3"
    if not synthesize(path, ["-c:a", "libmp3lame", "-b:a", "64k"]):
        pytest.skip("ffmpeg build cannot encode mp3")

    from mutagen.id3 import COMM, ID3, TENC, TPUB

    tags = ID3(path) if path.stat().st_size else ID3()
    tags.add(TENC(encoding=3, text=["LAME in FL Studio 2026"]))
    tags.add(TPUB(encoding=3, text=["Dirty Workz"]))
    tags.add(COMM(encoding=3, lang="eng", desc="", text=["copyright free"]))
    tags.save(path, v2_version=3)

    extra = read_audio_file(path).extra_tags
    assert extra.get("encoder") == "LAME in FL Studio 2026"
    assert extra.get("publisher") == "Dirty Workz"
    assert any("copyright free" in v for v in extra.values())
