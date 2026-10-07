from autotagger.normalize import (
    is_junk_candidate,
    normalize,
    split_artists,
    split_featured,
    strip_leading_tracknum,
    version_tags,
)


def test_normalize_collapses_cosmetic_noise():
    assert normalize("Come Together (Remastered 2009)") == normalize("Come Together")
    assert normalize("Hey Jude - 2015 Remaster") == normalize("Hey Jude")
    assert normalize("Björk") == "bjork"
    assert normalize("Simon & Garfunkel") == normalize("Simon and Garfunkel")


def test_normalize_keeps_distinct_titles_distinct():
    assert normalize("Blue Monday") != normalize("Blue Monday 88")


def test_version_tags_are_identity_bearing():
    assert version_tags("About a Girl (Live)") == {"live"}
    assert version_tags("Song (Acoustic Version)") == {"acoustic"}
    # Remaster is packaging, not a version marker.
    assert version_tags("Song (Remastered 2011)") == set()


def test_split_featured_handles_every_spelling():
    for raw in (
        "Stronger (feat. Kanye West)",
        "Stronger ft. Kanye West",
        "Stronger featuring Kanye West",
        "Stronger - feat. Kanye West",
    ):
        base, guests = split_featured(raw)
        assert base == "Stronger", raw
        assert guests == ["Kanye West"], raw


def test_split_featured_multiple_guests():
    base, guests = split_featured("Track (feat. A$AP Rocky & Tyler, The Creator)")
    assert base == "Track"
    assert len(guests) >= 2


def test_split_artists_flattens_primaries_and_guests():
    names = split_artists("Drake & Future feat. Young Thug")
    assert "Drake" in names and "Future" in names and "Young Thug" in names


def test_strip_leading_tracknum():
    assert strip_leading_tracknum("03 - Title")[:2] == ("Title", 3)
    assert strip_leading_tracknum("1-05 Title") == ("Title", 5, 1)
    assert strip_leading_tracknum("[04] Title")[:2] == ("Title", 4)
    # A bare number is a title, not a track number.
    assert strip_leading_tracknum("1979")[1] is None


def test_is_junk_candidate():
    assert is_junk_candidate("Wonderwall", "Karaoke Stars", "Karaoke Hits Vol. 3")
    assert is_junk_candidate("Wonderwall (In the Style of Oasis)", None, None)
    assert not is_junk_candidate("Wonderwall", "Oasis", "(What's the Story) Morning Glory?")


def test_strip_packaging_removes_release_noise_only():
    from autotagger.normalize import strip_packaging

    assert strip_packaging("Exit Music (For a Film) [Remastered]") == "Exit Music (For a Film)"
    assert strip_packaging("Hey Jude - 2015 Remaster") == "Hey Jude"
    assert strip_packaging("Song (Bonus Track)") == "Song"
    assert strip_packaging("Song [1997 2017]") == "Song"
    # Genuine version markers and stylized titles survive untouched.
    assert strip_packaging("About a Girl (Live)") == "About a Girl (Live)"
    assert strip_packaging("Stronger (feat. Kanye West)") == "Stronger (feat. Kanye West)"
    assert strip_packaging("DAMN.") == "DAMN."


def test_strip_search_noise_drops_label_and_scene_brackets():
    from autotagger.normalize import strip_search_noise

    # A label suffix no word list can recognize, but fatal to a search.
    assert strip_search_noise("Rush [NCS Release]") == "Rush"
    assert strip_search_noise("Gone [Monstercat Release]") == "Gone"
    # Parentheses carry identity and are left alone.
    assert strip_search_noise("Stronger (feat. Kanye West)") == "Stronger (feat. Kanye West)"
    assert strip_search_noise("About a Girl (Live)") == "About a Girl (Live)"
    # Never reduce a title to nothing.
    assert strip_search_noise("[NCS Release]") == "[NCS Release]"
    assert strip_search_noise(None) is None
