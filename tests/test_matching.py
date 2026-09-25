from pathlib import Path

from autotagger.matching import Thresholds, classify, dedupe, rank, score_candidate
from autotagger.models import AudioFile, Candidate


def make_file(**kw) -> AudioFile:
    defaults = dict(path=Path("/music/test.mp3"), ext=".mp3")
    defaults.update(kw)
    return AudioFile(**defaults)


def make_candidate(**kw) -> Candidate:
    defaults = dict(source="itunes", source_id="1", title="T", artist="A",
                    artwork_url="https://example.invalid/100x100bb.jpg")
    defaults.update(kw)
    return Candidate(**defaults)


def test_exact_match_scores_high():
    af = make_file(title="One More Time", artist="Daft Punk", album="Discovery", duration=320.0)
    cand = make_candidate(title="One More Time", artist="Daft Punk", album="Discovery", duration=320.3)
    assert score_candidate(af, cand).score > 0.95


def test_duration_mismatch_sinks_an_otherwise_perfect_match():
    af = make_file(title="About a Girl", artist="Nirvana", duration=217.0)
    studio = make_candidate(title="About a Girl", artist="Nirvana", album="Bleach", duration=337.0)
    live = make_candidate(source_id="2", title="About a Girl (Live)", artist="Nirvana",
                          album="MTV Unplugged in New York", duration=217.0)
    assert score_candidate(af, live).score > score_candidate(af, studio).score


def test_live_vs_studio_penalty():
    af = make_file(title="Song", artist="Band", duration=200.0)
    studio = make_candidate(title="Song", artist="Band", duration=200.0)
    live = make_candidate(source_id="2", title="Song (Live)", artist="Band", duration=200.0)
    assert score_candidate(af, studio).score > score_candidate(af, live).score


def test_karaoke_is_buried():
    af = make_file(title="Wonderwall", artist="Oasis", duration=258.0)
    real = make_candidate(title="Wonderwall", artist="Oasis", duration=258.0)
    junk = make_candidate(source_id="2", title="Wonderwall (Karaoke Version)",
                          artist="Karaoke Stars", duration=258.0)
    assert score_candidate(af, junk).score < 0.4
    assert score_candidate(af, real).score > 0.9


def test_featured_artist_position_does_not_matter():
    af = make_file(title="Stronger (feat. Kanye West)", artist="Daft Punk", duration=200.0)
    cand = make_candidate(title="Stronger", artist="Daft Punk feat. Kanye West", duration=200.0)
    assert score_candidate(af, cand).score > 0.9


def test_missing_signals_are_not_penalised():
    """A file with only a title should still match on title + duration."""
    af = make_file(title="Paranoid Android", duration=384.0)
    cand = make_candidate(title="Paranoid Android", artist="Radiohead",
                          album="OK Computer", duration=384.0)
    assert score_candidate(af, cand).score > 0.9


def test_classify_bands():
    af = make_file(title="Song", artist="Band", duration=200.0)
    good = make_candidate(title="Song", artist="Band", duration=200.0)
    bad = make_candidate(source_id="2", title="Totally Different", artist="Someone", duration=99.0)
    t = Thresholds()

    verdict, _ = classify(rank(af, [good, bad]), t)
    assert verdict == "accept"

    verdict, _ = classify(rank(af, [bad]), t)
    assert verdict == "reject"


def test_ambiguous_when_two_candidates_tie():
    af = make_file(title="Song", artist="Band")
    a = make_candidate(title="Song", artist="Band", album="Album One")
    b = make_candidate(source_id="2", title="Song", artist="Band", album="Album Two")
    verdict, _ = classify(rank(af, [a, b]), Thresholds())
    assert verdict in ("ambiguous", "accept")


def test_dedupe_collapses_reissues():
    a = make_candidate(title="Song", artist="Band", album="Original", duration=200.0)
    b = make_candidate(source_id="2", title="Song", artist="Band",
                       album="Greatest Hits", duration=200.4)
    assert len(dedupe([a, b])) == 1


def test_original_album_beats_reissue_on_a_tie():
    af = make_file(title="Exit Music (For a Film)", artist="Radiohead", duration=264.0)
    original = make_candidate(title="Exit Music (For a Film)", artist="Radiohead",
                              album="OK Computer", duration=264.5)
    reissue = make_candidate(source_id="2", title="Exit Music (For a Film) [Remastered]",
                             artist="Radiohead", album="OK Computer OKNOTOK 1997 2017",
                             duration=265.0)
    assert score_candidate(af, original).score > score_candidate(af, reissue).score


def test_hard_duration_miss_outweighs_a_perfect_title():
    """Regression: a 2-minute duration gap must sink a title/artist-perfect hit."""
    af = make_file(title="About a Girl", artist="Nirvana", duration=217.0)
    wrong_length = make_candidate(title="About a Girl", artist="Nirvana",
                                  album="Bleach", duration=337.0)
    assert score_candidate(af, wrong_length).score < 0.4


def test_wrong_artist_is_disqualifying_even_with_perfect_title_and_duration():
    """Regression: the "Kazan / One Last Time" failure.

    A common title plus a coincidental duration match must not carry a
    recording by a completely unrelated artist. rapidfuzz scores
    WRatio("kazan", "ariana grande") at 54, which used to buy half credit on the
    artist component and pushed the total over the acceptance bar.
    """
    af = make_file(guessed_title="One Last Time", guessed_artist="Kazan", duration=192.8)
    for wrong_artist, dur in [("LP", 193.0), ("Ariana Grande", 197.0), ("Anna Clendening", 197.0)]:
        cand = make_candidate(title="One Last Time", artist=wrong_artist,
                              album="One Last Time - Single", duration=dur)
        score = score_candidate(af, cand).score
        assert score < Thresholds().consider, f"{wrong_artist} scored {score:.2f}"

    verdict, _ = classify(rank(af, [
        make_candidate(title="One Last Time", artist="LP", duration=193.0),
        make_candidate(source_id="2", title="One Last Time", artist="Ariana Grande", duration=197.0),
    ]), Thresholds())
    assert verdict == "reject"


def test_artist_similarity_gives_no_partial_credit_to_unrelated_names():
    from autotagger.matching import _set_similarity

    assert _set_similarity({"kazan"}, {"ariana grande"}) == 0.0
    assert _set_similarity({"kazan"}, {"lp"}) == 0.0
    # Genuine variations still match.
    assert _set_similarity({"the beatles"}, {"beatles"}) > 0.8


def test_missing_guest_credit_is_still_forgiven():
    """The gate must not fire on a legitimately incomplete artist credit."""
    af = make_file(title="Song", artist="Drake", duration=200.0)
    cand = make_candidate(title="Song", artist="Drake, Future", duration=200.0)
    assert score_candidate(af, cand).score > 0.85


def test_no_artist_on_file_means_no_gate():
    """A file with no artist evidence is scored on what it does have."""
    af = make_file(title="Paranoid Android", duration=384.0)
    cand = make_candidate(title="Paranoid Android", artist="Radiohead", duration=384.0)
    assert score_candidate(af, cand).score > 0.9


def test_dedupe_merges_across_providers_and_records_corroboration():
    apple = make_candidate(source="itunes", title="Can't Stop", artist="Red Hot Chili Peppers",
                           album="By the Way", duration=268.0)
    deezer = make_candidate(source="deezer", source_id="d1", title="Can't Stop",
                            artist="Red Hot Chili Peppers", album="By the Way",
                            duration=269.0, isrc="USWB10201694", label="Warner")
    merged = dedupe([apple, deezer])
    assert len(merged) == 1
    assert merged[0].extra["sources"] == ["itunes", "deezer"]
    # Blanks on the keeper are filled from the duplicate.
    assert merged[0].isrc == "USWB10201694"
    assert merged[0].label == "Warner"


def test_dedupe_keeps_originals_and_reissues_apart():
    """A remaster normalizes identically to the original but is a different release."""
    original = make_candidate(title="Exit Music (For a Film)", artist="Radiohead",
                              album="OK Computer", duration=264.5)
    reissue = make_candidate(source="deezer", source_id="d2",
                             title="Exit Music (For a Film) [Remastered]", artist="Radiohead",
                             album="OK Computer OKNOTOK 1997 2017", duration=265.0)
    assert len(dedupe([reissue, original])) == 2

    af = make_file(title="Exit Music (For a Film)", guessed_artist="Radiohead", duration=264.0)
    best = rank(af, dedupe([reissue, original]))[0]
    assert best.candidate.album == "OK Computer"


def test_dedupe_truncates_only_after_merging():
    """A second provider's results must survive the first filling the quota."""
    apple = [make_candidate(source="itunes", source_id=str(i), title=f"Song {i}",
                            artist="Band", duration=200.0 + i) for i in range(25)]
    deezer = [make_candidate(source="deezer", source_id="d0", title="Song 0",
                             artist="Band", duration=200.0, isrc="AAA")]
    merged = dedupe(apple + deezer, limit=25)
    assert merged[0].isrc == "AAA"
    assert merged[0].extra["sources"] == ["itunes", "deezer"]


def test_same_release_of_same_recording_is_not_ambiguous():
    """Standard vs deluxe edition of one performance should resolve, not refuse."""
    af = make_file(guessed_title="Can't Stop", duration=269.0)
    standard = make_candidate(title="Can't Stop", artist="Red Hot Chili Peppers",
                              album="By the Way", duration=269.0)
    deluxe = make_candidate(source_id="2", title="Can't Stop", artist="Red Hot Chili Peppers",
                            album="By the Way (Deluxe Edition)", duration=268.0)
    verdict, best = classify(rank(af, [deluxe, standard]), Thresholds())
    assert verdict == "accept"
    assert best.candidate.album == "By the Way"


def test_differently_labelled_same_performance_resolves():
    from autotagger.matching import same_recording

    a = make_candidate(title="About a Girl (Live Acoustic)", artist="Nirvana", duration=217.0)
    b = make_candidate(source_id="2", title="About A Girl (Live)", artist="Nirvana", duration=217.5)
    assert same_recording(a, b)


def test_genuinely_different_recordings_stay_ambiguous():
    from autotagger.matching import same_recording

    a = make_candidate(title="Song", artist="Band A", duration=200.0)
    b = make_candidate(source_id="2", title="Song", artist="Band B", duration=200.0)
    assert not same_recording(a, b)
    # Same artist and title, but two minutes apart — different recordings.
    c = make_candidate(source_id="3", title="Song", artist="Band A", duration=320.0)
    assert not same_recording(a, c)
