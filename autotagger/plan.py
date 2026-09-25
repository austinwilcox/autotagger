"""Plan / apply — Terraform semantics for tagging.

The problem this solves: a dry run is only a *prediction*. Between looking at it
and running `--write`, the answer can change — a local LLM is not deterministic,
the catalogue is not frozen, and the HTTP cache eventually expires. So what you
reviewed is not necessarily what gets written.

A plan removes that gap. `plan` does all the resolution — catalogue lookups, LLM
calls, artwork fetching — and freezes the outcome into a file. `apply` makes
**no network calls and no LLM calls at all**: it reads the plan and writes
exactly what the plan says, including the exact image bytes seen at plan time.

    autotagger plan ~/Music --llm          # resolve, review, freeze
    autotagger show                        # re-read the plan later
    autotagger apply                       # execute it, verbatim

Two properties worth relying on:

* **The plan is plain JSON and is meant to be edited.** Delete entries you don't
  want, fix a title the model got slightly wrong, change a track number. Apply
  executes what is in the file, so hand-editing is a supported workflow rather
  than a hack.

* **Drift is detected.** Each entry records a fingerprint of the file as it was
  at plan time. If a file changed in between — you retagged it in another tool,
  replaced it, re-ripped it — apply refuses that entry rather than overwriting
  work it did not account for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .audio import CANONICAL_FIELDS, read_audio_file, write_tags
from .models import FileResult

log = logging.getLogger(__name__)

PLAN_VERSION = 1
DEFAULT_PLAN_PATH = Path("autotagger.plan.json")

# Carried inside every saved plan, because the file is meant to be opened and
# edited and the rules should travel with it.
PLAN_README = [
    "This plan is meant to be edited. `autotagger apply` writes exactly what is",
    "in this file and nothing else — it performs no catalogue lookups, no LLM",
    "calls and no network requests, so what you read here is what you get.",
    "",
    "To change what gets written:",
    "  - Edit any value under an entry's 'changes' to correct it.",
    "  - Delete a single field from 'changes' to leave that tag untouched.",
    "  - Set an entry's \"apply\" to false to skip that file entirely.",
    "  - Delete a whole entry to skip it (setting apply:false is easier to undo).",
    "  - Set an entry's \"artwork\" to null to keep its existing cover.",
    "  - Edit \"rename_to\", or set it to null to leave the filename alone.",
    "",
    "Paths are absolute, so a plan can be applied from any directory.",
    "",
    "'previous' records what each tag held when the plan was made. It is shown",
    "for context and is never written.",
    "",
    "'fingerprint' detects drift. If a file changes between plan and apply, that",
    "entry is refused rather than overwriting work the plan did not account for.",
    "Re-plan to pick up the new state, or apply --force to override.",
]


def artwork_dir_for(plan_path: Path) -> Path:
    """Sidecar directory holding the exact cover bytes this plan will write."""
    return plan_path.with_suffix(plan_path.suffix + ".artwork")


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------

def fingerprint(path: Path) -> str:
    """Identify a file's current state cheaply, for drift detection.

    Size plus the current tag values, not mtime — copying or syncing a file
    rewrites mtime without changing anything that matters here, and that would
    produce false drift on every entry.
    """
    try:
        size = path.stat().st_size
        af = read_audio_file(path)
    except Exception:  # noqa: BLE001 - unreadable file is drift enough
        return "unreadable"
    parts = [
        str(size),
        af.title or "", af.artist or "", af.album or "", af.album_artist or "",
        str(af.track_number or ""), str(af.disc_number or ""),
        af.year or "", af.genre or "", str(af.has_artwork),
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]


# --------------------------------------------------------------------------
# Plan model
# --------------------------------------------------------------------------

@dataclass
class PlannedArtwork:
    digest: str
    mime: str
    width: int
    height: int
    source_url: str | None = None
    note: str | None = None

    def filename(self) -> str:
        ext = "png" if self.mime == "image/png" else "jpg"
        return f"{self.digest}.{ext}"

    def to_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest, "mime": self.mime, "width": self.width,
            "height": self.height, "source_url": self.source_url, "note": self.note,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PlannedArtwork":
        return cls(
            digest=d["digest"], mime=d.get("mime", "image/jpeg"),
            width=int(d.get("width", 0)), height=int(d.get("height", 0)),
            source_url=d.get("source_url"), note=d.get("note"),
        )


@dataclass
class PlanEntry:
    path: Path
    fingerprint: str
    apply: bool = True
    changes: dict[str, Any] = field(default_factory=dict)
    previous: dict[str, Any] = field(default_factory=dict)
    artwork: PlannedArtwork | None = None
    rename_to: str | None = None
    match: str = ""
    confidence: float = 0.0
    method: str = ""
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.changes and self.artwork is None and not self.rename_to

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "apply": self.apply,
            "fingerprint": self.fingerprint,
            "changes": self.changes,
            "previous": self.previous,
            "artwork": self.artwork.to_json() if self.artwork else None,
            "rename_to": self.rename_to,
            "match": self.match,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PlanEntry":
        art = d.get("artwork")
        return cls(
            path=Path(d["path"]).expanduser().resolve(),
            apply=bool(d.get("apply", True)),
            fingerprint=d.get("fingerprint", ""),
            changes=d.get("changes") or {},
            previous=d.get("previous") or {},
            artwork=PlannedArtwork.from_json(art) if art else None,
            rename_to=d.get("rename_to"),
            match=d.get("match", ""),
            confidence=float(d.get("confidence") or 0.0),
            method=d.get("method", ""),
            reasoning=d.get("reasoning", ""),
            warnings=list(d.get("warnings") or []),
        )


@dataclass
class Plan:
    entries: list[PlanEntry] = field(default_factory=list)
    unmatched: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0
    tool_version: str = __version__
    version: int = PLAN_VERSION
    settings: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "_readme": PLAN_README,
            "version": self.version,
            "tool_version": self.tool_version,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.created_at)),
            "settings": self.settings,
            "entries": [e.to_json() for e in self.entries],
            "unmatched": self.unmatched,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Plan":
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise PlanError(f"no plan at {path}. Run `autotagger plan <paths>` first.") from None
        except json.JSONDecodeError as exc:
            raise PlanError(f"{path} is not valid JSON: {exc}") from exc

        if not isinstance(data, dict) or "entries" not in data:
            raise PlanError(f"{path} does not look like an autotagger plan")
        version = int(data.get("version") or 0)
        if version > PLAN_VERSION:
            raise PlanError(
                f"{path} was written by a newer autotagger (plan format v{version}, "
                f"this build understands v{PLAN_VERSION})"
            )
        return cls(
            entries=[PlanEntry.from_json(e) for e in data["entries"]],
            unmatched=list(data.get("unmatched") or []),
            created_at=float(data.get("created_at") or 0.0),
            tool_version=data.get("tool_version", "unknown"),
            version=version,
            settings=data.get("settings") or {},
        )


class PlanError(Exception):
    pass


_INT_FIELDS = {"track_number", "track_total", "disc_number", "disc_total"}
_BOOL_FIELDS = {"compilation", "explicit"}


def validate(plan: "Plan") -> list[str]:
    """Check a (possibly hand-edited) plan and return human-readable problems.

    Editing the plan is a supported workflow, so mistakes in it deserve a clear
    message naming the file and field — not a stack trace halfway through a run
    that has already written to half the library.
    """
    problems: list[str] = []
    for entry in plan.entries:
        where = entry.path.name
        for field_name, value in entry.changes.items():
            if field_name not in CANONICAL_FIELDS:
                problems.append(
                    f"{where}: unknown tag {field_name!r}. Valid tags: "
                    + ", ".join(sorted(CANONICAL_FIELDS))
                )
                continue
            if value is None:
                continue
            if field_name in _INT_FIELDS:
                try:
                    int(value)
                except (TypeError, ValueError):
                    problems.append(
                        f"{where}: {field_name} must be a whole number, got {value!r}"
                    )
            elif field_name in _BOOL_FIELDS:
                if not isinstance(value, bool):
                    problems.append(
                        f"{where}: {field_name} must be true or false, got {value!r}"
                    )
            elif not isinstance(value, (str, int, float)):
                problems.append(
                    f"{where}: {field_name} must be text, got {type(value).__name__}"
                )
        if entry.rename_to and ("/" in entry.rename_to or "\\" in entry.rename_to):
            problems.append(
                f"{where}: rename_to must be a filename, not a path ({entry.rename_to!r}). "
                "Renaming only ever happens within the file's own directory."
            )
    return problems


# --------------------------------------------------------------------------
# Building a plan from a dry run
# --------------------------------------------------------------------------

def build(results: list[FileResult], settings: dict[str, Any], plan_path: Path,
          now: float) -> Plan:
    """Freeze dry-run results into a plan, saving cover bytes alongside it."""
    art_dir = artwork_dir_for(plan_path)
    plan = Plan(created_at=now, settings=settings)

    for result in results:
        if result.status in ("failed", "no-match") or (
            result.status == "skipped" and not result.changes
        ):
            plan.unmatched.append({
                "path": str(result.path.resolve()),
                "status": result.status,
                "reason": result.error or (result.decision.reasoning if result.decision else ""),
                "warnings": list(result.decision.warnings) if result.decision else [],
            })
            continue

        art = _persist_artwork(result, art_dir)
        entry = PlanEntry(
            # Absolute, always. A plan names specific files on disk, so it has to
            # keep working when applied from a different directory than the one
            # it was created in — a relative path would silently resolve against
            # the wrong root and report every entry as missing.
            path=result.path.resolve(),
            fingerprint=fingerprint(result.path),
            changes={c.field: c.new for c in result.changes},
            previous={c.field: c.old for c in result.changes},
            artwork=art,
            rename_to=result.rename_to.name if result.rename_to and result.rename_to != result.path else None,
        )
        if result.decision:
            entry.confidence = result.decision.confidence
            entry.method = result.decision.method
            entry.reasoning = result.decision.reasoning
            entry.warnings = list(result.decision.warnings)
            if result.decision.chosen:
                entry.match = result.decision.chosen.label_line()
        if not entry.is_empty:
            plan.entries.append(entry)

    plan.save(plan_path)
    return plan


def _persist_artwork(result: FileResult, art_dir: Path) -> PlannedArtwork | None:
    """Write the resolved cover into the plan's sidecar directory, deduplicated."""
    art = result.artwork_data
    if art is None:
        return None
    planned = PlannedArtwork(
        digest=art.digest, mime=art.mime, width=art.width, height=art.height,
        source_url=art.source_url, note=result.artwork,
    )
    target = art_dir / planned.filename()
    if not target.exists():
        art_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(art.data)
    return planned


# --------------------------------------------------------------------------
# Applying a plan
# --------------------------------------------------------------------------

@dataclass
class ApplyOutcome:
    path: Path
    status: str          # applied | drift | missing | failed | skipped
    detail: str = ""


def apply(
    plan: Plan,
    plan_path: Path,
    *,
    force: bool = False,
    backup_dir: Path | None = None,
    only: list[Path] | None = None,
) -> list[ApplyOutcome]:
    """Execute a plan. Performs no network, provider or LLM calls whatsoever."""
    art_dir = artwork_dir_for(plan_path)
    wanted = [p.resolve() for p in (only or [])]
    out: list[ApplyOutcome] = []

    for entry in plan.entries:
        path = entry.path
        if not entry.apply:
            out.append(ApplyOutcome(path, "skipped", 'disabled in the plan ("apply": false)'))
            continue
        if entry.is_empty:
            out.append(ApplyOutcome(path, "skipped", "nothing left to write"))
            continue
        if wanted and not any(path.resolve() == w or w in path.resolve().parents for w in wanted):
            continue
        if not path.exists():
            out.append(ApplyOutcome(path, "missing", "file no longer exists"))
            continue

        current = fingerprint(path)
        if current != entry.fingerprint and not force:
            out.append(ApplyOutcome(
                path, "drift",
                "file changed since the plan was made — re-plan, or use --force to "
                "apply anyway",
            ))
            continue

        artwork_bytes, art_note = _load_artwork(entry, art_dir)
        try:
            if backup_dir:
                _backup(path, backup_dir)
            write_tags(
                path,
                dict(entry.changes),
                artwork=artwork_bytes,
                artwork_mime=entry.artwork.mime if entry.artwork else "image/jpeg",
            )
            final = _do_rename(path, entry)
            detail = f"{len(entry.changes)} tag(s)"
            if artwork_bytes:
                detail += f", artwork {entry.artwork.width}x{entry.artwork.height}"
            elif art_note:
                detail += f", {art_note}"
            if final != path:
                detail += f", renamed to {final.name}"
            out.append(ApplyOutcome(final, "applied", detail))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            out.append(ApplyOutcome(path, "failed", f"{type(exc).__name__}: {exc}"))

    return out


def _load_artwork(entry: PlanEntry, art_dir: Path) -> tuple[bytes | None, str]:
    if entry.artwork is None:
        return None, ""
    source = art_dir / entry.artwork.filename()
    if not source.exists():
        return None, f"artwork missing from {art_dir.name}/ — skipped"
    data = source.read_bytes()
    # The plan names the bytes by digest; verify rather than trust the filename.
    if hashlib.sha256(data).hexdigest()[:16] != entry.artwork.digest:
        return None, "artwork failed its digest check — skipped"
    return data, ""


def _do_rename(path: Path, entry: PlanEntry) -> Path:
    if not entry.rename_to:
        return path
    target = path.with_name(entry.rename_to)
    if target == path:
        return path
    if target.exists():
        log.warning("rename target exists, keeping original: %s", target.name)
        return path
    path.rename(target)
    return target


def _backup(path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    af = read_audio_file(path)
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
    (backup_dir / f"{key}.json").write_text(json.dumps({
        "path": str(path.resolve()),
        "saved_at": time.time(),
        "tags": {
            "title": af.title, "artist": af.artist, "album": af.album,
            "album_artist": af.album_artist, "track_number": af.track_number,
            "track_total": af.track_total, "disc_number": af.disc_number,
            "disc_total": af.disc_total, "year": af.year, "genre": af.genre,
            "composer": af.composer,
        },
        "had_artwork": af.has_artwork,
    }, indent=2))
