"""Per-file orchestration: evidence -> candidates -> decision -> tags on disk.

The escalation ladder, cheapest first. Each rung only runs when the one above it
came back ambiguous, which keeps a large library fast and keeps the slow/
rate-limited resources (Apple, the LLM, the user) for the files that need them.

    1. read tags + audio properties            (local, free)
    2. parse the filename and folder structure (local, free)
    3. acoustic fingerprint                    (optional, local CPU + one API call)
    4. catalogue search                        (rate-limited)
    5. deterministic scoring                   (local, free)  <- resolves most files
    6. LLM query repair, then re-search        (only when step 4 found nothing)
    7. LLM selection                           (only when step 5 was ambiguous)
    8. interactive prompt                      (only when still ambiguous)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from . import audio as audio_mod
from .artwork import Artwork, fetch_artwork, save_folder_image, should_replace
from .config import Config
from .httpcache import Fetcher, HttpCache
from .llm import LLMClient, LLMConfig, LLMError
from .matching import classify, dedupe, rank
from .models import AudioFile, Candidate, Decision, FileResult, ScoredCandidate, TagChange
from .normalize import join_credit, split_credit, strip_packaging
from .parse import enrich_from_path, looks_like_various_artists
from .prompts import QUERY_SCHEMA, SELECTION_SCHEMA, query_messages, selection_messages
from .providers import DeezerProvider, ITunesProvider, MusicBrainzProvider
from .websearch import OllamaWebSearch, WebIdentification, WebIdentifier

log = logging.getLogger(__name__)


class Tagger:
    def __init__(self, config: Config, interactive_chooser: Callable | None = None):
        self.config = config
        self.fetcher = Fetcher(cache=HttpCache(enabled=not config.no_cache))
        self.itunes = ITunesProvider(self.fetcher, country=config.country, rate=config.itunes_rate)
        # Deezer is free and keyless, so it is also used as an automatic
        # fallback when Apple returns nothing — better dance/electronic coverage
        # is exactly what Apple is weakest at.
        self.deezer = DeezerProvider(self.fetcher)
        self.musicbrainz = MusicBrainzProvider(self.fetcher) if config.use_musicbrainz else None
        self.web: WebIdentifier | None = None
        self.llm: LLMClient | None = None
        self.choose_interactively = interactive_chooser
        self._artwork_cache: dict[str, Artwork | None] = {}

        if config.llm:
            self.llm = LLMClient(
                LLMConfig(
                    url=config.llm_url,
                    model=config.llm_model,
                    api=config.llm_api,
                    api_key=config.llm_api_key,
                    max_tokens=config.llm_max_tokens,
                    think=config.llm_think,
                )
            )
            if config.web_search:
                search_client = OllamaWebSearch(
                    api_key=config.ollama_api_key,
                    cache=self.fetcher.cache,
                    max_results=config.web_results,
                )
                if search_client.configured:
                    self.web = WebIdentifier(search_client, self.llm)
                else:
                    log.debug("web search requested but no API key; skipping")

    # -- public ------------------------------------------------------------

    def process(self, path: Path) -> FileResult:
        try:
            return self._process(path)
        except audio_mod.UnsupportedFormat as exc:
            return FileResult(path=path, status="skipped", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            log.debug("unhandled error on %s", path, exc_info=True)
            return FileResult(path=path, status="failed", error=f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        self.fetcher.close()
        if self.llm:
            self.llm.close()

    # -- internals ---------------------------------------------------------

    def _process(self, path: Path) -> FileResult:
        af = audio_mod.read_audio_file(path)
        enrich_from_path(af, self.config.library_root)

        if self.config.skip_tagged and _looks_complete(af, self.config):
            return FileResult(path=path, status="skipped", error=None, artwork="already complete")

        decision = self._identify(af)
        if not decision.accepted:
            return FileResult(path=path, status="no-match", decision=decision)

        chosen = decision.chosen
        assert chosen is not None
        if chosen.source == "itunes":
            chosen = self.itunes.enrich(chosen)
        elif chosen.source == "deezer":
            chosen = self.deezer.enrich(chosen)
        if self.musicbrainz:
            try:
                chosen = self.musicbrainz.annotate(chosen)
            except Exception as exc:  # noqa: BLE001
                decision.warnings.append(f"musicbrainz enrichment failed: {exc}")
        decision.chosen = chosen

        values = self._build_values(af, chosen)
        changes = self._diff(af, values)

        art, art_note = self._resolve_artwork(af, chosen)
        if not changes and not art:
            return FileResult(path=path, status="skipped", decision=decision, artwork=art_note)

        if self.config.dry_run:
            return FileResult(
                path=path, status="dry-run", decision=decision, changes=changes,
                artwork=art_note, artwork_data=art,
                rename_to=rename_target(path, values, self.config.rename),
            )

        self._backup(af)
        write_values = {c.field: c.new for c in changes}
        audio_mod.write_tags(
            path,
            write_values,
            artwork=art.data if art else None,
            artwork_mime=art.mime if art else "image/jpeg",
        )
        if art and self.config.save_cover_file:
            save_folder_image(art, path.parent, self.config.cover_filename)

        final_path = path
        if self.config.rename:
            final_path = self._rename(path, values) or path

        return FileResult(
            path=final_path, status="applied", decision=decision, changes=changes,
            artwork=art_note, artwork_data=art,
        )

    # -- identification ----------------------------------------------------

    def _identify(self, af: AudioFile) -> Decision:
        candidates = self._gather(af)
        if not candidates:
            return Decision(file=af, chosen=None, confidence=0.0, method="none",
                            reasoning="no catalogue results for any search term")

        trust_web = self.config.web_trust
        scored = rank(af, dedupe(candidates, limit=self.config.search_limit), trust_web)
        verdict, best = classify(scored, self.config.thresholds)

        # Nothing in any catalogue matched. Ask the web what this recording is,
        # then retry the catalogues with what it tells us — a corrected artist
        # spelling or the compilation a track appears on often finds a release
        # the raw filename never could.
        web_note: WebIdentification | None = None
        if verdict == "reject" and self.web:
            web_note = self.web.identify(af)
            if web_note and web_note.found:
                extra: list[Candidate] = []
                for query in web_note.search_queries:
                    extra.extend(self.itunes.search(query, limit=self.config.search_limit))
                    extra.extend(self.deezer.search(query, limit=self.config.search_limit))
                if extra:
                    candidates = candidates + extra
                    scored = rank(af, dedupe(candidates, limit=self.config.search_limit), trust_web)
                    verdict, best = classify(scored, self.config.thresholds)
                # Still nothing verifiable: offer the web's own reading of the
                # file, capped so it cannot auto-apply without --web-trust.
                if verdict == "reject":
                    synthetic = web_note.to_candidate()
                    if synthetic:
                        candidates = candidates + [synthetic]
                        scored = rank(af, dedupe(candidates, limit=self.config.search_limit), trust_web)
                        verdict, best = classify(scored, self.config.thresholds)

        def finish(decision: Decision) -> Decision:
            if web_note:
                decision.warnings.append(f"web: {web_note.summary()}")
            return decision

        if verdict == "accept" and best and not self.config.llm_always:
            return finish(Decision(
                file=af, chosen=best.candidate, confidence=best.score, method="deterministic",
                reasoning=_explain(best), runners_up=scored[1:4],
            ))

        # Rejected files are sent to the LLM too — not to rescue the match, which
        # the guard in `_ask_llm` forbids, but so it can tell the user what the
        # recording actually is. On a file the catalogue simply does not contain,
        # "this looks like an independent hardstyle release" is the most useful
        # thing the tool can say.
        if self.llm:
            decision = self._ask_llm(af, scored)
            if decision:
                return finish(decision)

        if verdict == "accept" and best:
            return finish(Decision(
                file=af, chosen=best.candidate, confidence=best.score, method="deterministic",
                reasoning=_explain(best), runners_up=scored[1:4],
            ))

        if verdict == "ambiguous" and self.config.interactive and self.choose_interactively:
            picked = self.choose_interactively(af, scored[:8])
            if picked is not None:
                return finish(Decision(file=af, chosen=picked.candidate, confidence=1.0,
                                       method="manual", reasoning="chosen by user",
                                       runners_up=scored[1:4]))
            return finish(Decision(file=af, chosen=None, confidence=0.0, method="manual",
                                   reasoning="user skipped", runners_up=scored[:4]))

        return finish(Decision(
            file=af, chosen=None, confidence=best.score if best else 0.0, method="none",
            reasoning=_no_match_reason(verdict, best, scored, self.config),
            runners_up=scored[:4],
        ))

    def _gather(self, af: AudioFile) -> list[Candidate]:
        """Collect candidates, escalating only as far as needed."""
        candidates: list[Candidate] = []

        if self.config.use_acoustid:
            from .fingerprint import identify, seed_queries

            hits = identify(self.fetcher, af, self.config.acoustid_key)
            for query in seed_queries(hits):
                candidates.extend(self.itunes.search(query, limit=self.config.search_limit))

        if not candidates:
            candidates = self.itunes.search_for_file(af, limit=self.config.search_limit)

        if self.config.use_deezer or not candidates:
            candidates.extend(self.deezer.search_for_file(af, limit=self.config.search_limit))

        if not candidates and self.llm:
            for query in self._repair_queries(af):
                candidates.extend(self.itunes.search(query, limit=self.config.search_limit))
                if candidates:
                    break

        if not candidates and self.musicbrainz:
            candidates = self.musicbrainz.search_for_file(af, limit=self.config.search_limit)

        return candidates

    def _repair_queries(self, af: AudioFile) -> list[str]:
        """Ask the LLM what this mangled filename should be searched as."""
        assert self.llm is not None
        try:
            data = self.llm.complete_json(query_messages(af), QUERY_SCHEMA)
        except LLMError as exc:
            log.debug("query repair failed for %s: %s", af.path.name, exc)
            return []
        queries = [q for q in (data.get("queries") or []) if isinstance(q, str) and len(q) >= 3]
        # Adopt the LLM's parse as filename guesses so the matcher can score with it.
        af.guessed_artist = af.guessed_artist or data.get("probable_artist")
        af.guessed_title = af.guessed_title or data.get("probable_title")
        af.guessed_album = af.guessed_album or data.get("probable_album")
        if data.get("notes"):
            af.path_hints.append(f"llm: {data['notes']}")
        return queries[:5]

    def _ask_llm(self, af: AudioFile, scored: list[ScoredCandidate]) -> Decision | None:
        assert self.llm is not None
        try:
            data = self.llm.complete_json(
                selection_messages(af, scored, self.config.llm_max_candidates),
                SELECTION_SCHEMA,
            )
        except LLMError as exc:
            log.debug("llm selection failed for %s: %s", af.path.name, exc)
            return None

        idx = data.get("selected_index")
        reasoning = str(data.get("reasoning") or "").strip()
        notes = str(data.get("notes") or "").strip()
        confidence = _clamp(data.get("confidence"))
        identified = str(data.get("identified_as") or "").strip()
        warnings = [w for w in (notes, f"identified as: {identified}" if identified else "") if w]

        def refuse(why: str) -> Decision:
            return Decision(
                file=af, chosen=None, confidence=confidence, method="llm",
                reasoning=why, runners_up=scored[:4], warnings=warnings,
            )

        if idx is None:
            return refuse(reasoning or "LLM rejected all candidates")

        # The model judged this not to be a catalogue release. Trust the refusal
        # even if it contradicts itself by also naming an index — a false
        # positive here writes a wrong artist into the user's library.
        if data.get("catalogue_plausible") is False:
            return refuse(
                "LLM judged this not to be a commercial catalogue release. " + reasoning
            )

        if not isinstance(idx, int) or not 0 <= idx < min(len(scored), self.config.llm_max_candidates):
            log.debug("llm returned out-of-range index %r for %s", idx, af.path.name)
            return None

        if confidence < self.config.llm_min_confidence:
            return refuse(
                f"LLM confidence {confidence:.2f} below --llm-min-confidence. {reasoning}"
            )

        # The LLM breaks ties; it does not overrule a rejection. Its pick must
        # still clear the deterministic bar the scorer applies to everything
        # else. Without this, a confidently-argued but wrong selection — the
        # model asserting an artist match that is not there — sails straight
        # through into the file.
        picked_score = scored[idx].score
        if picked_score < self.config.thresholds.consider:
            return refuse(
                f"LLM chose a candidate the matcher scored {picked_score:.2f}, below the "
                f"{self.config.thresholds.consider:.2f} threshold "
                f"({'; '.join(scored[idx].penalties) or 'weak match'}). "
                f"Model's argument: {reasoning}"
            )

        chosen = _apply_llm_formatting(scored[idx].candidate, data)
        return Decision(
            file=af, chosen=chosen, confidence=confidence, method="llm",
            reasoning=reasoning, runners_up=[s for i, s in enumerate(scored[:5]) if i != idx],
            warnings=warnings,
        )

    # -- tag values --------------------------------------------------------

    def _build_values(self, af: AudioFile, c: Candidate) -> dict[str, Any]:
        # Artwork-only: the match still has to happen (that is where the cover
        # comes from), but nothing is written into the text tags.
        if self.config.artwork_only:
            return {}

        various = c.is_compilation or looks_like_various_artists(c.album_artist)
        # Providers ship packaging noise inside the title ("... [Remastered]").
        # It describes the release, not the recording, so it is stripped before
        # it reaches the tag — the release info is already in `album` and `date`.
        #
        # The ALBUM name is deliberately left alone. "By the Way (Deluxe
        # Edition)" has different track numbers and a different track total from
        # "By the Way"; renaming the album while keeping the deluxe numbering
        # would produce metadata that contradicts itself.
        clean = strip_packaging if self.config.clean_titles else (lambda x: x)
        title, artist_value, album_artist = self._artist_fields(af, c, clean(c.title), various)
        values: dict[str, Any] = {
            "title": title,
            "artist": artist_value,
            "album": c.album,
            "album_artist": album_artist,
            "track_number": c.track_number,
            "track_total": c.track_total,
            "disc_number": c.disc_number or 1,
            "disc_total": c.disc_total,
            "year": c.year,
            "date": c.release_date or c.year,
            "genre": c.genre,
            "composer": c.composer,
            # Only flag a compilation when it *is* one. Writing `False` has no
            # representation to read back from, so it would be re-proposed on
            # every subsequent plan — a permanent phantom diff.
            "compilation": various or None,
            "isrc": c.isrc,
            "copyright": c.copyright,
            # Explicitness only has a tag to live in on MP4/M4A.
            "explicit": c.explicit if af.ext in (".m4a", ".mp4", ".m4b") else None,
            "musicbrainz_track_id": c.musicbrainz_track_id,
            "musicbrainz_album_id": c.musicbrainz_album_id,
            "musicbrainz_artist_id": c.musicbrainz_artist_id,
        }
        return {k: v for k, v in values.items() if v is not None and self.config.may_write(k)}

    def _artist_fields(
        self, af: AudioFile, c: Candidate, title: str | None, various: bool
    ) -> tuple[str | None, Any, str | None]:
        """Decide what goes in `title`, `artist` and `album_artist`.

        The default ("list") writes each credited act as a separate tag value.
        A joined credit string is why a library ends up with "Fox Stevenson",
        "Fox Stevenson & Cruk" and "Fox Stevenson & Cruk & Priority One" as
        three unrelated artists; one value per act collapses them back to one
        entry per real artist.

        `album_artist` is always the single lead act, because that is the field
        most players group albums by — a joined string there reintroduces the
        exact problem at the album level.
        """
        style = self.config.artist_style
        if style == "keep":
            return title, c.artist, c.album_artist or ("Various Artists" if various else c.artist)

        artists = c.artists or split_credit(c.artist)
        if not artists:
            return title, c.artist, c.album_artist or c.artist

        lead = artists[0]
        album_artist = "Various Artists" if various else lead

        if style == "primary":
            # Credit the lead act only, and preserve the others in the title so
            # nothing is lost — that is where players expect guests anyway.
            others = artists[1:]
            if others and title and "feat" not in title.lower():
                title = f"{title} (feat. {join_credit(others)})"
            return title, lead, album_artist

        return title, artists if len(artists) > 1 else lead, album_artist

    def _diff(self, af: AudioFile, values: dict[str, Any]) -> list[TagChange]:
        """Compute the changes to write, honouring --overwrite."""
        # Every field we may write needs an entry here. A field that cannot be
        # read back always looks like a change, so it would be re-proposed on
        # every run and make `plan` -> `apply` -> `plan` never settle.
        current = {
            "title": af.title,
            "artist": af.artists if len(af.artists) > 1 else af.artist,
            "album": af.album,
            "album_artist": af.album_artist, "track_number": af.track_number,
            "track_total": af.track_total, "disc_number": af.disc_number,
            "disc_total": af.disc_total, "year": af.year, "date": af.date,
            "genre": af.genre, "composer": af.composer, "isrc": af.isrc,
            "copyright": af.copyright, "compilation": af.compilation,
            "explicit": af.explicit,
            "musicbrainz_track_id": af.musicbrainz_track_id,
        }
        changes: list[TagChange] = []
        for field_name, new in values.items():
            old = current.get(field_name)
            if old is not None and not self.config.overwrite_existing:
                continue
            change = TagChange(field=field_name, old=old, new=new)
            if not change.is_noop:
                changes.append(change)
        return changes

    # -- artwork -----------------------------------------------------------

    def _resolve_artwork(self, af: AudioFile, c: Candidate) -> tuple[Artwork | None, str | None]:
        if not self.config.artwork or not c.artwork_url:
            return None, None
        if af.has_artwork and self.config.artwork_if_missing:
            return None, "kept existing artwork (--artwork-if-missing)"

        key = c.artwork_url
        if key in self._artwork_cache:
            art = self._artwork_cache[key]
        else:
            art = fetch_artwork(self.fetcher, key, max_px=self.config.artwork_max_px)
            self._artwork_cache[key] = art
        if not art:
            return None, "no usable artwork found"

        existing = audio_mod.read_embedded_artwork(af.path) if af.has_artwork else None
        replace, reason = should_replace(existing, art, only_if_missing=self.config.artwork_if_missing)
        return (art if replace else None), reason

    # -- side effects ------------------------------------------------------

    def _backup(self, af: AudioFile) -> None:
        """Snapshot the prior tags so `autotagger undo` can put them back."""
        if not self.config.backup_dir:
            return
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        import hashlib

        key = hashlib.sha256(str(af.path.resolve()).encode()).hexdigest()[:20]
        payload = {
            "path": str(af.path.resolve()),
            "saved_at": time.time(),
            "tags": {
                "title": af.title,
            "artist": af.artists if len(af.artists) > 1 else af.artist,
            "album": af.album,
                "album_artist": af.album_artist, "track_number": af.track_number,
                "track_total": af.track_total, "disc_number": af.disc_number,
                "disc_total": af.disc_total, "year": af.year, "genre": af.genre,
                "composer": af.composer,
            },
            "had_artwork": af.has_artwork,
        }
        (self.config.backup_dir / f"{key}.json").write_text(json.dumps(payload, indent=2))

    def _rename(self, path: Path, values: dict[str, Any]) -> Path | None:
        target = rename_target(path, values, self.config.rename)
        if target is None or target == path:
            return target
        if target.exists():
            log.warning("rename target already exists, keeping original: %s", target.name)
            return None
        path.rename(target)
        return target


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_ILLEGAL = str.maketrans({c: "_" for c in '/\\:*?"<>|'})


def rename_target(path: Path, values: dict[str, Any], template: str | None) -> Path | None:
    """Resolve a rename template against the new tag values. Pure — touches no disk.

    Shared by the tagger and the planner so a plan records exactly the filename
    an apply will produce.
    """
    if not template:
        return None
    fields = {
        "title": values.get("title") or "Unknown Title",
        "artist": values.get("artist") or "Unknown Artist",
        "album": values.get("album") or "Unknown Album",
        "album_artist": values.get("album_artist") or values.get("artist") or "Unknown Artist",
        "track": values.get("track_number") or 0,
        "disc": values.get("disc_number") or 1,
        "year": values.get("year") or "",
        "genre": values.get("genre") or "",
    }
    try:
        stem = template.format(**fields)
    except (KeyError, ValueError) as exc:
        log.warning("bad --rename template: %s", exc)
        return None
    return path.with_name(_safe_filename(stem) + path.suffix.lower())


def _safe_filename(name: str) -> str:
    cleaned = name.translate(_ILLEGAL).strip().rstrip(".")
    return cleaned[:180] or "untitled"


def _clamp(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _explain(best: ScoredCandidate) -> str:
    parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(best.breakdown.items()))
    text = f"score {best.score:.2f} ({parts})"
    if best.penalties:
        text += " | " + "; ".join(best.penalties)
    return text


def _apply_llm_formatting(candidate: Candidate, data: dict[str, Any]) -> Candidate:
    """Let the LLM restyle text fields, but never invent facts.

    Only fields the model actually returned are taken, and only the presentation
    fields — anything identifying (IDs, durations, artwork, ISRC) stays exactly
    as the provider supplied it.
    """
    import copy

    out = copy.deepcopy(candidate)
    for field_name in ("title", "artist", "album", "album_artist", "composer", "genre"):
        value = data.get(field_name)
        if isinstance(value, str) and value.strip():
            setattr(out, field_name, value.strip())
    for field_name in ("track_number", "disc_number"):
        value = data.get(field_name)
        if isinstance(value, int) and value > 0:
            setattr(out, field_name, value)
    year = data.get("year")
    if isinstance(year, str) and year.strip().isdigit() and len(year.strip()) == 4:
        out.year = year.strip()
    if isinstance(data.get("is_compilation"), bool):
        out.is_compilation = data["is_compilation"]
    return out


def _looks_complete(af: AudioFile, config: Config) -> bool:
    """A file with all the core fields and (if wanted) artwork needs no work."""
    core = (af.title, af.artist, af.album, af.album_artist, af.track_number, af.year)
    if not all(core):
        return False
    return af.has_artwork or not config.artwork


def _no_match_reason(verdict: str, best, scored, config: Config) -> str:
    """Explain a non-match in terms the user can act on."""
    if best is None:
        return "no catalogue candidates"
    if verdict == "reject":
        return (
            f"best candidate scored {best.score:.2f}, below the "
            f"--consider threshold of {config.thresholds.consider:.2f}: "
            f"{best.candidate.label_line()}"
        )
    runner = scored[1].score if len(scored) > 1 else 0.0
    hint = "--llm" if not config.llm else "--interactive"
    return (
        f"ambiguous: top candidate {best.score:.2f} vs runner-up {runner:.2f} "
        f"(needs a gap of {config.thresholds.decisive_gap:.2f}, or a score of "
        f"{config.thresholds.accept:.2f}). Re-run with {hint} to resolve."
    )