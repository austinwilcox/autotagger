"""Multi-artist credits.

The failure this prevents: a library that lists "Fox Stevenson",
"Fox Stevenson & Cruk" and "Fox Stevenson & Cruk & Priority One" as three
unrelated artists, because each track's credit was written as one joined string.

The opposite failure is worse, and is why splitting is only ever done on
authoritative data: "Simon & Garfunkel" must never become "Simon".
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autotagger.audio import read_audio_file, write_tags
from autotagger.normalize import join_credit, split_credit

# --- splitting --------------------------------------------------------------

ONE_ACT = [
    "Simon & Garfunkel",
    "Nick Cave & The Bad Seeds",
    "Earth, Wind & Fire",
    "Tyler, The Creator",
    "Crosby, Stills & Nash",
    "Hall & Oates",
    "Florence and the Machine",
    "AC/DC",
]


@pytest.mark.parametrize("name", ONE_ACT)
def test_a_band_whose_name_contains_a_separator_is_never_split(name):
    # With the provider's entity confirming it is one act.
    assert split_credit(name, name) == [name]
    # And with no entity at all, the safe default is to leave it alone.
    assert split_credit(name) == [name]


def test_collaboration_splits_on_the_artist_entity():
    assert split_credit("Fox Stevenson & Yue", "Fox Stevenson") == ["Fox Stevenson", "Yue"]
    assert split_credit("Fox Stevenson & Cruk & Priority One", "Fox Stevenson") == [
        "Fox Stevenson", "Cruk", "Priority One",
    ]


def test_feat_markers_split_without_an_entity():
    """"feat." is unambiguous, so it is safe even with no provider data."""
    assert split_credit("Daft Punk feat. Pharrell Williams") == [
        "Daft Punk", "Pharrell Williams",
    ]
    assert split_credit("Drake feat. Future & Young Thug") == [
        "Drake", "Future", "Young Thug",
    ]
    assert split_credit("Calvin Harris ft. Rihanna") == ["Calvin Harris", "Rihanna"]


def test_a_prefix_that_is_not_a_separator_does_not_split():
    """"Foxes" must not be read as "Fox" plus a remainder."""
    assert split_credit("Foxes", "Fox") == ["Foxes"]


def test_duplicates_are_collapsed():
    assert split_credit("Drake feat. Drake", "Drake") == ["Drake"]


def test_join_credit_round_trip():
    assert join_credit(["A"]) == "A"
    assert join_credit(["A", "B"]) == "A & B"
    assert join_credit(["A", "B", "C"]) == "A, B & C"


# --- writing ----------------------------------------------------------------

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is needed to synthesize fixtures"
)

FORMATS = [("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
           ("m4a", ["-c:a", "aac", "-b:a", "64k"]),
           ("flac", ["-c:a", "flac"])]


def make(path: Path, codec: list[str]) -> bool:
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", *codec, str(path)],
        capture_output=True,
    ).returncode == 0


@pytestmark_ffmpeg
@pytest.mark.parametrize("ext,codec", FORMATS, ids=[f[0] for f in FORMATS])
def test_multiple_artists_round_trip_as_separate_values(tmp_path: Path, ext, codec):
    path = tmp_path / f"a.{ext}"
    if not make(path, codec):
        pytest.skip(f"ffmpeg cannot encode {ext}")

    write_tags(path, {"artist": ["Fox Stevenson", "Yue"], "album_artist": "Fox Stevenson"})
    tags = read_audio_file(path)
    assert tags.artists == ["Fox Stevenson", "Yue"]
    assert tags.album_artist == "Fox Stevenson"


@pytestmark_ffmpeg
def test_mp3_multi_value_forces_id3v24(tmp_path: Path):
    """v2.3 joins values with "/", which corrupts a name like AC/DC."""
    import mutagen.id3

    path = tmp_path / "a.mp3"
    if not make(path, ["-c:a", "libmp3lame", "-b:a", "64k"]):
        pytest.skip("ffmpeg cannot encode mp3")

    write_tags(path, {"artist": ["AC/DC", "Guest"]})
    assert mutagen.id3.ID3(path).version[1] == 4
    assert read_audio_file(path).artists == ["AC/DC", "Guest"]


@pytestmark_ffmpeg
def test_single_artist_stays_a_plain_single_value(tmp_path: Path):
    import mutagen.id3

    path = tmp_path / "a.mp3"
    if not make(path, ["-c:a", "libmp3lame", "-b:a", "64k"]):
        pytest.skip("ffmpeg cannot encode mp3")

    write_tags(path, {"artist": "Simon & Garfunkel"})
    tags = read_audio_file(path)
    assert tags.artist == "Simon & Garfunkel"
    assert tags.artists == ["Simon & Garfunkel"]
    # No reason to leave v2.3 for a single value.
    assert mutagen.id3.ID3(path).version[1] == 3


@pytestmark_ffmpeg
def test_artist_list_is_idempotent(tmp_path: Path):
    """A multi-value artist must compare equal on re-read, or every later plan
    would propose the same change again."""
    from autotagger.models import TagChange

    path = tmp_path / "a.flac"
    if not make(path, ["-c:a", "flac"]):
        pytest.skip("ffmpeg cannot encode flac")

    artists = ["Fox Stevenson", "Cruk", "Priority One"]
    write_tags(path, {"artist": artists})
    current = read_audio_file(path).artists
    assert TagChange(field="artist", old=current, new=artists).is_noop
