"""Text normalization for music metadata matching.

Everything in here exists to answer one question well: *are these two strings
naming the same recording?* Music metadata is adversarially messy — the same
track shows up as "Song (Remastered 2011)", "Song - 2011 Remaster",
"Song (feat. X)", "SONG", and "Sóng". The matcher needs those to collapse to
the same key, while "Song (Live)" must NOT collapse into the studio version.

So normalization is split in two:
  * `normalize()`  — aggressive squashing of noise that never changes identity.
  * `version_tags()` — extraction of the noise that DOES change identity
                       (live / remix / acoustic / ...), kept as a separate set.
"""

from __future__ import annotations

import re
import unicodedata

# --- Version markers -------------------------------------------------------
# Each entry: canonical tag -> regex matching how it appears in the wild.
# A mismatch on any of these between two titles is a strong "different
# recording" signal, so they are pulled out before fuzzy comparison.
VERSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "live": re.compile(r"\b(live|unplugged|in concert|concert version)\b"),
    "remix": re.compile(r"\b(remix|rmx|re-?edit|bootleg|flip|vip mix)\b"),
    "acoustic": re.compile(r"\bacoustic\b"),
    "instrumental": re.compile(r"\b(instrumental|karaoke backing)\b"),
    "demo": re.compile(r"\b(demo|rough mix|work tape)\b"),
    "radio_edit": re.compile(r"\b(radio edit|radio version|single edit)\b"),
    "extended": re.compile(r"\b(extended|club mix|12\"? mix|long version)\b"),
    "reprise": re.compile(r"\breprise\b"),
    "mono": re.compile(r"\bmono( version| mix)?\b"),
    "orchestral": re.compile(r"\b(orchestral|symphonic|with orchestra)\b"),
    "sped_up": re.compile(r"\b(sped up|speed up|nightcore)\b"),
    "slowed": re.compile(r"\b(slowed|slowed down|reverb version|daycore)\b"),
    "cover": re.compile(r"\b(cover version|covers of|as made famous by|in the style of)\b"),
    "karaoke": re.compile(r"\bkaraoke\b"),
    "tribute": re.compile(r"\b(tribute|tribute to|performed by the .* players)\b"),
}

# Markers that describe *packaging*, not the recording. Safe to delete: a
# remaster of a track is still the same track for tagging purposes.
_COSMETIC = re.compile(
    r"""
    \b(
        (?:\d{4}\s*)?re-?master(?:ed|ing)?(?:\s*(?:version|edition))?(?:\s*\d{4})?
      | deluxe(?:\s*(?:edition|version))?
      | expanded(?:\s*edition)?
      | anniversary(?:\s*edition)?
      | special\s*edition
      | bonus\s*track
      | album\s*version
      | original\s*(?:mix|version|motion\s*picture\s*soundtrack)
      | explicit(?:\s*version)?
      | clean(?:\s*version)?
      | digital\s*remaster(?:ed)?
      | hd\s*remaster(?:ed)?
      | single\s*version
      | stereo(?:\s*(?:mix|version))?
      | \d{1,2}\s*bit
      | \d{2,3}\s*khz
    )\b
    """,
    re.VERBOSE,
)

# "feat." in all its disguises. Captures everything after it as the guest list.
_FEAT = re.compile(
    r"""
    [\s\(\[\-–—]*                      # optional opening bracket / dash
    \b(?:feat|ft|featuring|with|w/)\b\.?\s*
    (?P<guests>[^)\]]+?)
    \s*[\)\]]?\s*$                     # to end of string / closing bracket
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ARTIST_SPLIT = re.compile(r"\s*(?:,|&|\+|/|\bx\b|\band\b|\bvs\.?\b|;)\s*", re.IGNORECASE)

_BRACKETS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# Leading track numbers a ripper stuck on the filename: "03 - Title", "03. Title",
# "1-05 Title" (disc-track), "[04] Title".
_LEADING_TRACKNO = re.compile(
    r"^\s*[\[\(]?(?:(?P<disc>\d{1,2})\s*[-–]\s*)?(?P<track>\d{1,3})[\]\)]?\s*(?:[-–—.)_]\s*|\s+)"
)


def strip_accents(text: str) -> str:
    """Fold "Beyoncé" -> "Beyonce" so accent-stripped rips still match."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str | None, *, drop_brackets: bool = True) -> str:
    """Squash a string down to its identity-bearing core.

    Deliberately lossy. Use only for comparison, never for what gets written
    into a tag.
    """
    if not text:
        return ""
    s = strip_accents(text).lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("&", " and ")
    s = _COSMETIC.sub(" ", s)
    if drop_brackets:
        s = _BRACKETS.sub(" ", s)
    s = re.sub(r"\s+-\s+.*$", "", s) if _looks_like_trailing_noise(s) else s
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _looks_like_trailing_noise(s: str) -> bool:
    """True when a " - suffix" is packaging chatter rather than part of the title.

    "Song - 2011 Remaster" -> noise.  "Song - Part II" -> not noise.
    Checked after `_COSMETIC` has already blanked known packaging words, so an
    empty-ish tail is the tell.
    """
    parts = re.split(r"\s+-\s+", s)
    if len(parts) < 2:
        return False
    tail = _PUNCT.sub("", parts[-1]).strip()
    return tail == "" or bool(re.fullmatch(r"\d{4}|\d{4}\s*\w{0,10}", tail))


def version_tags(text: str | None) -> set[str]:
    """Extract identity-changing version markers ("live", "remix", ...)."""
    if not text:
        return set()
    s = strip_accents(text).lower()
    return {tag for tag, pat in VERSION_PATTERNS.items() if pat.search(s)}


def split_featured(title_or_artist: str | None) -> tuple[str, list[str]]:
    """Separate "Title (feat. A & B)" into ("Title", ["A", "B"]).

    Guest credits move around constantly between the title field and the artist
    field, so both are parsed the same way and compared as a set.
    """
    if not title_or_artist:
        return "", []
    m = _FEAT.search(title_or_artist)
    if not m:
        return title_or_artist.strip(), []
    base = title_or_artist[: m.start()].strip(" ([-–—\t")
    guests = [g.strip() for g in _ARTIST_SPLIT.split(m.group("guests")) if g.strip()]
    return base.strip(), guests


def split_artists(artist: str | None) -> list[str]:
    """Split a credited-artist string into individual artist names."""
    if not artist:
        return []
    base, guests = split_featured(artist)
    primaries = [a.strip() for a in _ARTIST_SPLIT.split(base) if a.strip()]
    return primaries + guests


def strip_leading_tracknum(name: str) -> tuple[str, int | None, int | None]:
    """Pull a leading "03 - " / "1-05 " off a filename stem.

    Returns (remainder, track_number, disc_number).
    """
    m = _LEADING_TRACKNO.match(name)
    if not m:
        return name.strip(), None, None
    rest = name[m.end():].strip()
    if not rest:  # the whole name was a number — keep it as the title
        return name.strip(), None, None
    disc = int(m.group("disc")) if m.group("disc") else None
    return rest, int(m.group("track")), disc


def is_junk_candidate(*fields: str | None) -> bool:
    """Detect karaoke/tribute/"made famous by" pollution in search results.

    The iTunes catalogue is full of soundalike albums that score well on pure
    string similarity. They are never what the user wants.
    """
    blob = " ".join(f.lower() for f in fields if f)
    return bool(
        re.search(
            r"\b(karaoke|made famous by|in the style of|tribute to|"
            r"originally performed by|8-?bit|lullaby versions?|"
            r"music box versions?|rockabye baby)\b",
            blob,
        )
    )


# Packaging words that, when they are the *entire* content of a bracket, make
# the whole bracket noise: "Exit Music (For a Film) [Remastered]".
_TRAILING_PACKAGING = re.compile(
    r"\s*[-–—]\s*(?:\d{4}\s*)?(?:digital\s+|hd\s+)?re-?master(?:ed|ing)?"
    r"(?:\s*\d{4})?\s*$",
    re.IGNORECASE,
)


def strip_packaging(title: str | None) -> str | None:
    """Remove reissue/packaging noise from a title that will be WRITTEN to a tag.

    Distinct from `normalize()`: the result is user-facing text, so casing,
    punctuation and genuine version markers are all preserved. Only the parts
    that describe the *release* rather than the *recording* come out.

        "Exit Music (For a Film) [Remastered]"  -> "Exit Music (For a Film)"
        "Hey Jude - 2015 Remaster"              -> "Hey Jude"
        "About a Girl (Live)"                   -> unchanged
    """
    if not title:
        return title

    def _maybe_drop(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        remainder = _COSMETIC.sub("", inner.lower())
        remainder = re.sub(r"[^\w]", "", remainder)
        # Bare years inside a bracket are packaging too: "(2017)", "[1997 2017]".
        remainder = re.sub(r"(?:19|20)\d{2}", "", remainder)
        return "" if not remainder.strip() else match.group(0)

    out = _BRACKETS.sub(_maybe_drop, title)
    out = _TRAILING_PACKAGING.sub("", out)
    return _WS.sub(" ", out).strip(" -–—") or title


# --- Splitting a credit into individual acts -------------------------------
# This is only ever safe with authoritative data. "Simon & Garfunkel",
# "Nick Cave & The Bad Seeds", "Earth, Wind & Fire", "Tyler, The Creator" and
# "Crosby, Stills & Nash" are all single acts whose names contain the very
# separators a naive split would break on — and splitting one of those is far
# worse than leaving a joined credit alone, because it silently renames an
# artist. So: split on the provider's artist entity, or on an explicit
# "feat."-style marker, and otherwise leave the string whole.

_FEAT_MARKER = re.compile(
    r"\s*(?:[\(\[]\s*)?\b(?:feat|ft|featuring|w/)\b\.?\s+", re.IGNORECASE
)
_CREDIT_SEP = re.compile(r"\s*(?:,|&|\+|/|;|\bx\b|\bvs\.?\b|\band\b)\s*", re.IGNORECASE)


def split_credit(credit: str | None, primary: str | None = None) -> list[str]:
    """Split a credit string into individual acts.

    `primary` is the provider's own artist entity — the name the store files the
    track under. It is what makes this safe:

        credit "Fox Stevenson & Yue",     primary "Fox Stevenson"
            -> ["Fox Stevenson", "Yue"]           (a collaboration)

        credit "Simon & Garfunkel",       primary "Simon & Garfunkel"
            -> ["Simon & Garfunkel"]              (one act, left alone)

    Without a `primary` that actually prefixes the credit, only explicit
    "feat."-style markers are split on. A bare "&" is left intact.
    """
    if not credit or not credit.strip():
        return []
    credit = credit.strip()

    head, guests = _split_on_feat(credit)

    if primary and primary.strip():
        primary = primary.strip()
        if _casefold(head) == _casefold(primary):
            return _dedupe_names([primary, *guests])
        if _casefold(head).startswith(_casefold(primary)):
            remainder = head[len(primary):]
            # Only a real separator marks a collaboration; "Foxes" must not be
            # read as "Fox" plus "es".
            if _CREDIT_SEP.match(remainder) or not remainder.strip():
                rest = [p for p in _CREDIT_SEP.split(remainder) if p.strip()]
                return _dedupe_names([primary, *rest, *guests])

    return _dedupe_names([head, *guests])


def _split_on_feat(credit: str) -> tuple[str, list[str]]:
    """"A feat. B & C" -> ("A", ["B", "C"]). Unambiguous, so always safe."""
    m = _FEAT_MARKER.search(credit)
    if not m:
        return credit.strip(), []
    head = credit[: m.start()].strip(" ([-–—,")
    tail = credit[m.end():].strip(" )]")
    guests = [g.strip() for g in _CREDIT_SEP.split(tail) if g.strip()]
    return head or credit.strip(), guests


def _casefold(value: str) -> str:
    return strip_accents(value).casefold().replace("’", "'").strip()


def _dedupe_names(names: list[str]) -> list[str]:
    """Drop blanks and case-insensitive repeats, preserving credit order."""
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        cleaned = name.strip(" ,&+/;-–—")
        if not cleaned:
            continue
        key = _casefold(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def join_credit(artists: list[str]) -> str:
    """Render a list of acts back into a display string: "A, B & C"."""
    names = _dedupe_names(artists)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} & {names[-1]}"
