"""Core data structures passed between the scanner, providers, matcher and writer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .normalize import strip_search_noise


@dataclass
class AudioFile:
    """Everything we know about a file *before* consulting any provider.

    This is the evidence bundle. Both the deterministic matcher and the LLM see
    exactly this — no hidden inputs — which keeps their decisions comparable.
    """

    path: Path
    ext: str
    duration: float | None = None          # seconds, from the decoder
    bitrate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None

    # Existing tags, already read. Missing/blank values are None, never "".
    title: str | None = None
    artist: str | None = None
    artists: list[str] = field(default_factory=list)   # one entry per credited act
    album: str | None = None
    album_artist: str | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    year: str | None = None
    date: str | None = None            # full release date when the tag carries one
    genre: str | None = None
    composer: str | None = None
    isrc: str | None = None
    copyright: str | None = None
    compilation: bool | None = None
    explicit: bool | None = None
    musicbrainz_track_id: str | None = None
    has_artwork: bool = False

    # Tags beyond the core set: encoder, comment, publisher/label, URL, BPM,
    # initial key, grouping, copyright. Never written back — they exist to be
    # *evidence*. A label name in `publisher` identifies a release the core
    # fields cannot, and an encoder string like "LAME in FL Studio 2026"
    # says the file is an independent production, not a catalogue release.
    extra_tags: dict[str, str] = field(default_factory=dict)

    # Derived from the path when tags are absent or untrustworthy.
    guessed_title: str | None = None
    guessed_artist: str | None = None
    guessed_album: str | None = None
    guessed_track_number: int | None = None
    guessed_disc_number: int | None = None
    path_hints: list[str] = field(default_factory=list)

    acoustid_fingerprint: str | None = None
    acoustid_recording_ids: list[str] = field(default_factory=list)

    @property
    def best_title(self) -> str | None:
        return self.title or self.guessed_title

    @property
    def best_artist(self) -> str | None:
        return self.artist or self.album_artist or self.guessed_artist

    @property
    def best_album(self) -> str | None:
        return self.album or self.guessed_album

    def search_terms(self) -> list[str]:
        """Query strings to try against a provider, most specific first.

        Providers are rate-limited, so order matters: the first query that
        yields a confident match short-circuits the rest.
        """
        terms: list[str] = []
        t, a, al = self.best_title, self.best_artist, self.best_album
        if t and a:
            terms.append(f"{a} {t}")
        # A label or scene suffix in the title is fatal to a catalogue search —
        # iTunes returns nothing at all for "iFeature Rush [NCS Release]" — so a
        # de-bracketed variant is tried early, right after the literal one.
        t_clean, a_clean = strip_search_noise(t), strip_search_noise(a)
        if t_clean and a_clean and (t_clean, a_clean) != (t, a):
            terms.append(f"{a_clean} {t_clean}")
        if t and a and al:
            terms.append(f"{a} {al} {t}")
        if t and al and not a:
            terms.append(f"{al} {t}")
        if t:
            terms.append(t)
        if t_clean and t_clean != t:
            terms.append(t_clean)
        if a and al:
            terms.append(f"{a} {al}")
        # Last resort: the raw filename stem, minus a leading track number.
        stem = self.path.stem.replace("_", " ").replace("-", " ")
        if stem and stem not in terms:
            terms.append(stem)
        seen: set[str] = set()
        return [x for x in terms if not (x.lower() in seen or seen.add(x.lower()))]


@dataclass
class Candidate:
    """One normalized result from a metadata provider."""

    source: str                       # "itunes" | "musicbrainz" | "llm"
    source_id: str
    title: str
    artist: str
    # Individually credited acts, from the provider's own structured data.
    # Empty when the provider only gave a joined display string.
    artists: list[str] = field(default_factory=list)
    album: str | None = None
    album_artist: str | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    year: str | None = None
    release_date: str | None = None
    genre: str | None = None
    composer: str | None = None
    duration: float | None = None      # seconds
    explicit: bool | None = None
    is_compilation: bool = False
    artwork_url: str | None = None     # highest-resolution URL we can construct
    isrc: str | None = None
    label: str | None = None
    barcode: str | None = None
    musicbrainz_track_id: str | None = None
    musicbrainz_album_id: str | None = None
    musicbrainz_artist_id: str | None = None
    copyright: str | None = None
    preview_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def label_line(self) -> str:
        bits = [f"{self.artist} — {self.title}"]
        if self.album:
            bits.append(f"[{self.album}]")
        if self.year:
            bits.append(f"({self.year})")
        if self.duration:
            bits.append(f"{int(self.duration)//60}:{int(self.duration)%60:02d}")
        return " ".join(bits)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float                       # 0.0–1.0 overall confidence
    breakdown: dict[str, float] = field(default_factory=dict)
    penalties: list[str] = field(default_factory=list)

    def __lt__(self, other: "ScoredCandidate") -> bool:
        return self.score < other.score


@dataclass
class Decision:
    """The outcome of matching one file."""

    file: AudioFile
    chosen: Candidate | None
    confidence: float
    method: str                        # "deterministic" | "llm" | "manual" | "none"
    reasoning: str = ""
    runners_up: list[ScoredCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.chosen is not None


@dataclass
class TagChange:
    field: str
    old: Any
    new: Any

    @property
    def is_noop(self) -> bool:
        return _norm(self.old) == _norm(self.new)


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, (list, tuple)):
        # A one-element list and the bare value are the same tag.
        items = [_norm(x) for x in v]
        items = [x for x in items if x is not None]
        if not items:
            return None
        return items[0] if len(items) == 1 else tuple(items)
    return v


@dataclass
class FileResult:
    path: Path
    status: str                        # applied | dry-run | skipped | failed | no-match
    decision: Decision | None = None
    changes: list[TagChange] = field(default_factory=list)
    artwork: str | None = None         # description of what happened to artwork
    error: str | None = None
    # The resolved cover, kept in memory so `plan` can pin the exact bytes it
    # saw rather than re-fetching (and possibly getting something else) later.
    # Never serialized by `to_json`.
    artwork_data: Any = None
    rename_to: Path | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": str(self.path),
            "status": self.status,
            "changes": [asdict(c) for c in self.changes],
            "artwork": self.artwork,
            "error": self.error,
        }
        if self.decision:
            d["confidence"] = round(self.decision.confidence, 3)
            d["method"] = self.decision.method
            d["reasoning"] = self.decision.reasoning
            if self.decision.chosen:
                d["matched"] = asdict(self.decision.chosen)
        return d
