from pathlib import Path

from autotagger.models import AudioFile
from autotagger.parse import enrich_from_path


def parse(path: str) -> AudioFile:
    p = Path(path)
    return enrich_from_path(AudioFile(path=p, ext=p.suffix))


def test_artist_album_track_from_folder_layout():
    af = parse("/Music/Radiohead/OK Computer (1997)/03 - Subterranean Homesick Alien.flac")
    assert af.guessed_title == "Subterranean Homesick Alien"
    assert af.guessed_track_number == 3
    assert af.guessed_album == "OK Computer"
    assert af.guessed_artist == "Radiohead"


def test_artist_dash_title_filename():
    af = parse("/downloads/Daft Punk - One More Time.mp3")
    assert af.guessed_artist == "Daft Punk"
    assert af.guessed_title == "One More Time"


def test_strips_scraper_junk():
    af = parse("/downloads/Tame Impala - The Less I Know The Better (Official Video).mp3")
    assert af.guessed_title == "The Less I Know The Better"
    assert af.guessed_artist == "Tame Impala"


def test_disc_track_prefix_and_disc_folder():
    af = parse("/Music/Various Artists/Now 42/CD2/1-05 Artist - Title.m4a")
    assert af.guessed_track_number == 5
    assert af.guessed_disc_number == 1


def test_artist_dash_album_single_folder():
    af = parse("/Music/Portishead - Dummy/02 Sour Times.mp3")
    assert af.guessed_artist == "Portishead"
    assert af.guessed_album == "Dummy"
    assert af.guessed_title == "Sour Times"


def test_generic_folders_are_not_treated_as_artists():
    af = parse("/Users/me/Music/song.mp3")
    assert af.guessed_artist is None
