"""Plan / apply: the guarantee is that apply writes exactly what the plan says."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autotagger.audio import read_audio_file, read_embedded_artwork, write_tags
from autotagger.plan import (
    Plan,
    PlanEntry,
    PlanError,
    PlannedArtwork,
    apply,
    artwork_dir_for,
    fingerprint,
    validate,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is needed to synthesize fixtures"
)

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000c4944415408d763f8cfc000000301010018dd8db00000000049454e44ae426082"
)


def make_mp3(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds),
         "-c:a", "libmp3lame", "-b:a", "64k", str(path)],
        check=True, capture_output=True,
    )
    return path


def make_plan(tmp_path: Path, audio: Path, changes: dict, with_art: bool = False) -> Path:
    art = None
    if with_art:
        import hashlib

        digest = hashlib.sha256(PNG).hexdigest()[:16]
        art = PlannedArtwork(digest=digest, mime="image/png", width=1, height=1)
        art_dir = artwork_dir_for(tmp_path / "p.json")
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / art.filename()).write_bytes(PNG)

    plan = Plan(entries=[PlanEntry(
        path=audio, fingerprint=fingerprint(audio), changes=changes, artwork=art,
    )])
    plan_path = tmp_path / "p.json"
    plan.save(plan_path)
    return plan_path


def test_apply_writes_exactly_what_the_plan_says(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "Planned", "artist": "Planner"})

    outcomes = apply(Plan.load(plan_path), plan_path)
    assert [o.status for o in outcomes] == ["applied"]

    tags = read_audio_file(audio)
    assert tags.title == "Planned"
    assert tags.artist == "Planner"
    assert tags.album is None          # never mentioned by the plan


def test_editing_a_value_changes_what_is_written(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "Original", "genre": "Pop"})

    data = json.loads(plan_path.read_text())
    data["entries"][0]["changes"]["genre"] = "French House"
    plan_path.write_text(json.dumps(data))

    apply(Plan.load(plan_path), plan_path)
    assert read_audio_file(audio).genre == "French House"


def test_deleting_a_field_leaves_that_tag_untouched(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    write_tags(audio, {"genre": "Keep Me"})
    plan_path = make_plan(tmp_path, audio, {"title": "T", "genre": "Overwrite Me"})

    data = json.loads(plan_path.read_text())
    del data["entries"][0]["changes"]["genre"]
    # The file is unchanged, so its fingerprint still matches.
    plan_path.write_text(json.dumps(data))

    apply(Plan.load(plan_path), plan_path)
    tags = read_audio_file(audio)
    assert tags.title == "T"
    assert tags.genre == "Keep Me"


def test_apply_false_skips_the_entry(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "Nope"})

    data = json.loads(plan_path.read_text())
    data["entries"][0]["apply"] = False
    plan_path.write_text(json.dumps(data))

    outcomes = apply(Plan.load(plan_path), plan_path)
    assert outcomes[0].status == "skipped"
    assert read_audio_file(audio).title is None


def test_drift_is_refused_and_force_overrides(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "Planned"})

    # Someone else retags the file after the plan was made.
    write_tags(audio, {"artist": "Changed Elsewhere"})

    outcomes = apply(Plan.load(plan_path), plan_path)
    assert outcomes[0].status == "drift"
    assert read_audio_file(audio).title is None

    outcomes = apply(Plan.load(plan_path), plan_path, force=True)
    assert outcomes[0].status == "applied"
    assert read_audio_file(audio).title == "Planned"


def test_pinned_artwork_is_written_from_the_sidecar(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "T"}, with_art=True)

    apply(Plan.load(plan_path), plan_path)
    assert read_embedded_artwork(audio) == PNG


def test_corrupted_artwork_fails_its_digest_check(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "T"}, with_art=True)

    art_dir = artwork_dir_for(plan_path)
    for f in art_dir.iterdir():
        f.write_bytes(b"not the planned image")

    outcomes = apply(Plan.load(plan_path), plan_path)
    assert outcomes[0].status == "applied"
    assert "digest" in outcomes[0].detail
    assert read_embedded_artwork(audio) is None


def test_missing_file_is_reported_not_crashed(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "T"})
    audio.unlink()

    outcomes = apply(Plan.load(plan_path), plan_path)
    assert outcomes[0].status == "missing"


def test_validation_catches_hand_editing_mistakes(tmp_path: Path):
    plan = Plan(entries=[PlanEntry(
        path=Path("/x/a.mp3"), fingerprint="f",
        changes={"titel": "typo", "track_number": "not a number"},
        rename_to="nested/path.mp3",
    )])
    problems = validate(plan)
    assert any("unknown tag 'titel'" in p for p in problems)
    assert any("track_number must be a whole number" in p for p in problems)
    assert any("must be a filename, not a path" in p for p in problems)


def test_plan_roundtrips_and_rejects_future_versions(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "T", "track_number": 3})

    loaded = Plan.load(plan_path)
    assert loaded.entries[0].changes == {"title": "T", "track_number": 3}
    assert loaded.entries[0].apply is True

    data = json.loads(plan_path.read_text())
    data["version"] = 999
    plan_path.write_text(json.dumps(data))
    with pytest.raises(PlanError, match="newer autotagger"):
        Plan.load(plan_path)


def test_plan_file_carries_its_own_instructions(tmp_path: Path):
    audio = make_mp3(tmp_path / "a.mp3")
    plan_path = make_plan(tmp_path, audio, {"title": "T"})
    readme = "\n".join(json.loads(plan_path.read_text())["_readme"])
    assert "meant to be edited" in readme
    assert '"apply" to false' in readme


def test_plan_stores_absolute_paths_so_it_applies_from_anywhere(tmp_path: Path, monkeypatch):
    """Regression: a plan made in one directory must apply from another."""
    from autotagger.plan import build
    from autotagger.models import FileResult, TagChange

    work = tmp_path / "work"
    work.mkdir()
    audio = make_mp3(work / "a.mp3")

    monkeypatch.chdir(work)
    result = FileResult(
        path=Path("a.mp3"),  # relative, as a scan from inside `work` would produce
        status="dry-run",
        changes=[TagChange(field="title", old=None, new="Planned")],
    )
    plan_path = work / "named.plan.json"
    build([result], {}, plan_path, now=0.0)

    stored = json.loads(plan_path.read_text())["entries"][0]["path"]
    assert Path(stored).is_absolute()

    # Apply from a completely different working directory.
    monkeypatch.chdir(tmp_path)
    outcomes = apply(Plan.load(plan_path), plan_path)
    assert [o.status for o in outcomes] == ["applied"]
    assert read_audio_file(audio).title == "Planned"


def test_plan_is_idempotent_after_apply(tmp_path: Path):
    """plan -> apply -> plan must settle. A field that cannot be read back
    would be re-proposed forever, which is a permanent phantom diff."""
    from autotagger.audio import CANONICAL_FIELDS

    audio = make_mp3(tmp_path / "a.mp3")
    values = {
        "title": "One More Time", "artist": "Daft Punk", "album": "Discovery",
        "album_artist": "Daft Punk", "track_number": 1, "track_total": 14,
        "disc_number": 1, "disc_total": 1, "year": "2000", "date": "2000-11-30",
        "genre": "Dance", "isrc": "GBDUW0000053", "copyright": "(C) 2000",
        "compilation": True,
    }
    assert set(values) <= set(CANONICAL_FIELDS)
    write_tags(audio, values)

    after = read_audio_file(audio)
    # Everything written must come back, or `_diff` will propose it again.
    assert after.date == "2000-11-30"
    assert after.isrc == "GBDUW0000053"
    assert after.copyright == "(C) 2000"
    assert after.compilation is True


def test_validate_accepts_multi_value_artist(tmp_path):
    """--artist-style list writes several artist values; that is not a bad edit."""
    from autotagger.plan import Plan, PlanEntry, validate

    entry = PlanEntry(
        path=tmp_path / "song.mp3",
        fingerprint="x",
        changes={"artist": ["Facading", "Holly Terrens", "Jagsy"]},
    )
    assert validate(Plan(entries=[entry])) == []

    # A list of the wrong shape is still a problem, and still names the field.
    bad = PlanEntry(path=tmp_path / "song.mp3", fingerprint="x", changes={"artist": ["", "  "]})
    problems = validate(Plan(entries=[bad]))
    assert len(problems) == 1 and "artist" in problems[0]

    # Lists remain invalid for single-valued fields.
    album = PlanEntry(path=tmp_path / "song.mp3", fingerprint="x", changes={"album": ["A", "B"]})
    assert "album" in validate(Plan(entries=[album]))[0]
