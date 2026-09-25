"""The deterministic matcher — the part that has to be right without an LLM.

Scoring is a weighted sum of independent signals, then a set of hard penalties
for the specific ways music matching goes wrong in practice. The weights are
tuned around one principle: **duration is the most trustworthy signal we have.**
A title can be spelled five ways; 3 minutes 47 seconds is 3 minutes 47 seconds.

Confidence bands (see `Thresholds`) decide what happens next:
    >= accept     -> write it, no questions asked
    >= consider   -> ambiguous: hand the shortlist to the LLM or the user
    <  consider   -> no match; leave the file alone
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import AudioFile, Candidate, ScoredCandidate
from .normalize import (
    is_junk_candidate,
    normalize,
    split_artists,
    split_featured,
    version_tags,
)

# Weights sum to 1.0. Signals that are absent get redistributed proportionally
# (see `_weighted`), so a file with no album tag isn't punished for it.
# Artist outweighs title deliberately. A title is a weak identifier — hundreds
# of unrelated songs are called "One Last Time" — whereas artist + duration
# together are close to unique.
WEIGHTS = {
    "artist": 0.30,
    "title": 0.28,
    "duration": 0.22,
    "album": 0.13,
    "track_number": 0.07,
}

# Below this, two artist names are not the same act and deserve no partial
# credit. rapidfuzz's WRatio scores "kazan" against "ariana grande" at 0.54 on
# incidental letter overlap; treating that as "half right" is what lets a
# title+duration coincidence carry a completely wrong match.
ARTIST_NAME_FLOOR = 0.62

# An artist this far off is disqualifying, not merely a low component score.
ARTIST_MISMATCH_GATE = 0.50

# How far apart two catalogues may report the same recording's length.
DUPLICATE_DURATION_S = 2.5

# Ceiling for candidates synthesized from web text rather than a catalogue.
# Sits just under the default `Thresholds.accept` of 0.88 on purpose.
WEB_SOURCE_CEILING = 0.80

# Duration tolerance. Encoder differences and gapless trimming account for a
# couple of seconds; beyond ~12s it is a different recording (edit, live, remix).
DURATION_EXACT_S = 2.0
DURATION_ZERO_S = 12.0


@dataclass
class Thresholds:
    accept: float = 0.88
    consider: float = 0.62
    # Accept outright if the runner-up is this far behind, even below `accept`.
    decisive_gap: float = 0.15


def score_candidate(af: AudioFile, cand: Candidate, trust_web: bool = False) -> ScoredCandidate:
    parts: dict[str, float] = {}
    penalties: list[str] = []

    # --- title ---------------------------------------------------------
    file_title, file_guests = split_featured(af.best_title)
    cand_title, cand_guests = split_featured(cand.title)
    if af.best_title:
        parts["title"] = _similarity(file_title, cand_title)

    # --- artist --------------------------------------------------------
    # Guest artists migrate between the title and artist fields constantly, so
    # both sides are compared as a flat set of everyone credited.
    if af.best_artist:
        file_artists = set(map(normalize, split_artists(af.best_artist) + file_guests))
        cand_artists = set(map(normalize, split_artists(cand.artist) + cand_guests))
        if cand.album_artist:
            cand_artists |= set(map(normalize, split_artists(cand.album_artist)))
        parts["artist"] = _set_similarity(file_artists, cand_artists)

    # --- duration ------------------------------------------------------
    if af.duration and cand.duration:
        delta = abs(af.duration - cand.duration)
        if delta <= DURATION_EXACT_S:
            parts["duration"] = 1.0
        elif delta >= DURATION_ZERO_S:
            parts["duration"] = 0.0
            penalties.append(f"duration off by {delta:.0f}s")
        else:
            span = DURATION_ZERO_S - DURATION_EXACT_S
            parts["duration"] = 1.0 - (delta - DURATION_EXACT_S) / span

    # --- album ---------------------------------------------------------
    if af.best_album and cand.album:
        parts["album"] = _similarity(af.best_album, cand.album)

    # --- track number --------------------------------------------------
    file_track = af.track_number or af.guessed_track_number
    if file_track and cand.track_number:
        parts["track_number"] = 1.0 if file_track == cand.track_number else 0.0

    score = _weighted(parts)

    # --- hard penalties -------------------------------------------------
    # A wrong artist is disqualifying. Without this, a perfect title plus a
    # coincidental duration match (both trivially common — three-minute songs
    # share titles constantly) adds up to a passing score for a recording by a
    # completely different act. This is the single most damaging failure mode,
    # because the result looks entirely plausible in a diff.
    if "artist" in parts and parts["artist"] < ARTIST_MISMATCH_GATE:
        score *= 0.20
        penalties.append(
            f"artist mismatch: {af.best_artist!r} vs {cand.artist!r}"
        )

    # A duration miss beyond tolerance is not merely a zero on one weighted
    # component — it is positive evidence of a *different recording*. Without
    # this multiplier a candidate with a perfect title and artist still scores
    # ~0.73 while being two minutes too long, which is how taggers end up
    # writing the studio album's metadata onto a live recording.
    duration_delta = (
        abs(af.duration - cand.duration) if (af.duration and cand.duration) else None
    )
    if duration_delta is not None and duration_delta >= DURATION_ZERO_S:
        score *= 0.25 if duration_delta >= 30 else 0.40

    # Version mismatch is the other classic silent failure. How hard it bites
    # depends on how much the file's own title is worth: a marker missing from
    # a real tag is meaningful, a marker missing from a scraped filename is not.
    file_versions = version_tags(af.best_title) | version_tags(af.best_album)
    cand_versions = version_tags(cand.title) | version_tags(cand.album)
    mismatched = file_versions.symmetric_difference(cand_versions)
    # `cover`/`karaoke`/`tribute` are handled separately below.
    mismatched -= {"cover", "karaoke", "tribute"}
    if mismatched:
        duration_exact = parts.get("duration") == 1.0
        file_title_trusted = bool(af.title)
        for tag in mismatched:
            if tag in cand_versions and not file_versions:
                # The candidate claims a version the file never mentions. When
                # the duration matches to the second, the candidate is probably
                # right and the file is simply under-labelled.
                if duration_exact:
                    score *= 0.70 if file_title_trusted else 0.85
                else:
                    score *= 0.45
            else:
                # The file claims a version this candidate does not have. That
                # is a straight contradiction regardless of duration.
                score *= 0.45
        penalties.append("version mismatch: " + ", ".join(sorted(mismatched)))

    if is_junk_candidate(cand.title, cand.artist, cand.album, cand.album_artist):
        if not is_junk_candidate(af.best_title, af.best_artist, af.best_album):
            score *= 0.15
            penalties.append("soundalike/karaoke release")

    # A candidate with no artwork is still a valid metadata match, but when a
    # near-equal alternative has art, prefer the one that can fill the cover.
    if not cand.artwork_url:
        score *= 0.97
        penalties.append("no artwork available")

    # Singles masquerading as the album: Apple lists "Song - Single" collections
    # alongside the real album. Slight demotion unless the file says otherwise.
    if cand.album and cand.album.strip().endswith("- Single"):
        if not (af.best_album and af.best_album.strip().endswith("- Single")):
            score *= 0.94
            penalties.append("single release rather than album")

    # Reissues, anniversary boxes and "Deluxe Edition"s are the same recording
    # on a different release. They score identically to the original after
    # normalization, so break the tie toward the original album — that is what
    # the user's track numbers and album grouping will agree with.
    file_blob = " ".join(filter(None, (af.best_album, af.best_title)))
    if _is_reissue(cand.album, cand.title) and not _is_reissue(file_blob, None):
        score *= 0.90
        penalties.append("reissue/alternate edition")

    # Independent catalogues agreeing is evidence in its own right.
    sources = cand.extra.get("sources") or [cand.source]
    if len(sources) > 1:
        score = min(1.0, score * 1.06)
        penalties.append("corroborated by " + ", ".join(sources))

    # A candidate assembled from web text has not been checked against a
    # catalogue and usually carries no duration, so the strongest verification
    # signal is unavailable. Hold it below the acceptance bar so it surfaces for
    # confirmation instead of being written silently.
    if cand.source == "web" and not trust_web:
        score = min(score, WEB_SOURCE_CEILING)
        penalties.append("unverified web source — confirm before applying")

    return ScoredCandidate(candidate=cand, score=min(score, 1.0), breakdown=parts, penalties=penalties)


def _weighted(parts: dict[str, float]) -> float:
    """Weighted mean over only the signals we actually have."""
    total_weight = sum(WEIGHTS[k] for k in parts)
    if total_weight <= 0:
        return 0.0
    return sum(parts[k] * WEIGHTS[k] for k in parts) / total_weight


def _similarity(a: str | None, b: str | None) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # token_set_ratio handles reordering and extra words ("The Beatles" vs
    # "Beatles"); WRatio catches typos and truncation. Take the better.
    return max(fuzz.token_set_ratio(na, nb), fuzz.WRatio(na, nb)) / 100.0


def _set_similarity(a: set[str], b: set[str]) -> float:
    """Best-match similarity between two sets of artist names.

    Not plain Jaccard: "Drake" vs "Drake, Future" should score well, because a
    missing guest credit on one side is normal, not a contradiction.
    """
    a = {x for x in a if x}
    b = {x for x in b if x}
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Every name on the shorter side should find a home on the longer side.
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    scores = []
    for x in small:
        best = max(_similarity(x, y) for y in large)
        # No partial credit for a name that simply is not in the other set.
        scores.append(best if best >= ARTIST_NAME_FLOOR else 0.0)
    base = sum(scores) / len(scores)
    # Mild penalty for extra unmatched names on the other side.
    extra = len(large) - len(small)
    return base * (0.96 ** extra)


def rank(af: AudioFile, candidates: list[Candidate], trust_web: bool = False) -> list[ScoredCandidate]:
    scored = [score_candidate(af, c, trust_web=trust_web) for c in candidates]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def classify(
    scored: list[ScoredCandidate], thresholds: Thresholds
) -> tuple[str, ScoredCandidate | None]:
    """Return ("accept" | "ambiguous" | "reject", best).

    "ambiguous" means *which recording is this?* is genuinely unresolved — the
    case where a human, or an LLM with good instructions, adds value. It does
    NOT mean "which release of the same recording?". Two catalogues offering the
    same performance on a standard and a deluxe edition look like a tie to a
    bare score comparison, but refusing there is worse for the user than simply
    preferring the original, which the reissue penalty already does.
    """
    if not scored:
        return "reject", None
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    gap = best.score - (runner.score if runner else 0.0)

    if best.score < thresholds.consider:
        return "reject", best
    if best.score >= thresholds.accept and gap >= 0.02:
        return "accept", best
    if gap >= thresholds.decisive_gap:
        return "accept", best
    if runner and same_recording(best.candidate, runner.candidate):
        return "accept", best
    return "ambiguous", best


def same_recording(a: Candidate, b: Candidate) -> bool:
    """Are these two entries the same performance, differing only by release?

    Duration does the work. Two catalogue entries by the same artist that run to
    the same length within a couple of seconds are the same master, however
    differently the releases label it — "About a Girl (Live)" and "About a Girl
    (Live Acoustic)" are one recording described twice.
    """
    if a.duration and b.duration:
        if abs(a.duration - b.duration) > DUPLICATE_DURATION_S:
            return False
        return (
            _similarity(a.artist, b.artist) >= 0.85
            and _similarity(a.title, b.title) >= 0.70
        )
    # Without durations there is nothing to verify with, so demand exact strings.
    return normalize(a.title) == normalize(b.title) and normalize(a.artist) == normalize(b.artist)


def dedupe(candidates: list[Candidate], limit: int = 40) -> list[Candidate]:
    """Collapse the same recording appearing more than once.

    Two jobs, and the second is why this is not a simple set operation:

    1. One provider listing a track on several releases (album, greatest hits,
       reissue). Keeps the first occurrence, which — because providers return by
       relevance — is usually the canonical album.

    2. *Different* providers describing the same recording. Apple and Deezer
       report durations a second apart and often name the release differently
       ("By the Way (Deluxe Edition)" vs "By the Way"), so an exact key leaves
       both in the running, where they split the vote and turn a file that used
       to match into an ambiguous one.

    Merging them is not merely tidier — two independent catalogues agreeing is
    real evidence, recorded in `extra["sources"]` and rewarded in scoring.
    """
    kept: list[Candidate] = []
    for cand in candidates:
        match = _find_duplicate(kept, cand)
        if match is not None:
            _merge_into(match, cand)
            continue
        cand.extra.setdefault("sources", [cand.source])
        kept.append(cand)
    # Truncate only at the end. Truncating during the loop would drop the
    # second provider's results entirely whenever the first one filled the
    # quota on its own, losing exactly the cross-provider corroboration this
    # function exists to find.
    return kept[:limit]


def _find_duplicate(kept: list[Candidate], cand: Candidate) -> Candidate | None:
    title, artist = normalize(cand.title), normalize(cand.artist)
    is_reissue = _is_reissue(cand.album, cand.title)
    for existing in kept:
        if normalize(existing.title) != title or normalize(existing.artist) != artist:
            continue
        # An original and its remaster normalize to the same strings and run to
        # the same length, but they are different releases. Keeping them apart
        # lets the reissue penalty decide between them, rather than whichever
        # provider happened to be queried first.
        if _is_reissue(existing.album, existing.title) != is_reissue:
            continue
        # Encoders and catalogues disagree by a second or two on the same master.
        if cand.duration and existing.duration:
            if abs(cand.duration - existing.duration) <= DUPLICATE_DURATION_S:
                return existing
            continue
        return existing
    return None


def _merge_into(keeper: Candidate, other: Candidate) -> None:
    """Fold a duplicate's data into the candidate we are keeping.

    Only fills blanks — the keeper ranked higher, so its own values win. This is
    how a Deezer result contributes its ISRC and label to an Apple match, and
    vice versa.
    """
    for field_name in (
        "album", "album_artist", "track_number", "track_total", "disc_number",
        "disc_total", "year", "release_date", "genre", "composer", "duration",
        "artwork_url", "isrc", "label", "barcode", "copyright", "preview_url",
        "musicbrainz_track_id", "musicbrainz_album_id", "musicbrainz_artist_id",
    ):
        if getattr(keeper, field_name, None) in (None, "") and getattr(other, field_name, None):
            setattr(keeper, field_name, getattr(other, field_name))

    sources = keeper.extra.setdefault("sources", [keeper.source])
    if other.source not in sources:
        sources.append(other.source)


_REISSUE = re.compile(
    r"\b(re-?master(ed|ing)?|deluxe|anniversary|expanded|okonotok|oknotok|"
    r"super\s*deluxe|legacy\s*edition|collector'?s\s*edition|"
    r"\d+th\s*anniversary|reissue)\b",
    re.IGNORECASE,
)


def _is_reissue(*fields: str | None) -> bool:
    blob = " ".join(f for f in fields if f)
    return bool(_REISSUE.search(blob))
