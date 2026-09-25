"""Command-line interface.

    autotagger tag ~/Music                      # dry run, prints a diff
    autotagger tag ~/Music --write              # actually write
    autotagger tag ~/Music --llm --write        # with local-LLM disambiguation
    autotagger tag ~/Music --interactive        # ask when unsure
    autotagger doctor                           # check the environment
    autotagger undo --backup-dir ~/.autotagger-backups
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from .audio import AUDIO_EXTENSIONS
from .config import Config
from .matching import Thresholds
from .pipeline import Tagger
from .plan import (
    DEFAULT_PLAN_PATH,
    Plan,
    PlanError,
    apply as apply_plan,
    artwork_dir_for,
    build as build_plan,
    validate as validate_plan,
)
from .report import (
    console,
    interactive_chooser,
    render_apply,
    render_plan,
    render_plan_summary,
    render_result,
    render_summary,
    write_json_report,
)

DEFAULT_BACKUP_DIR = Path.home() / ".autotagger" / "backups"


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autotagger",
        description="Tag music files from the Apple/iTunes catalogue, with artwork "
                    "and optional local-LLM disambiguation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument("--version", action="version", version=f"autotagger {__version__}")
    sub = p.add_subparsers(dest="command")

    tag = sub.add_parser("tag", help="tag files or directories (default command)")
    _add_tag_args(tag)

    plan = sub.add_parser(
        "plan", help="resolve everything and freeze the result into an editable plan file")
    _add_tag_args(plan, for_plan=True)
    plan.add_argument("-o", "--out", type=Path, default=DEFAULT_PLAN_PATH,
                      help=f"where to write the plan (default {DEFAULT_PLAN_PATH})")

    apply_p = sub.add_parser(
        "apply", help="execute a plan file — no network, no LLM, writes exactly what it says")
    apply_p.add_argument("plan_file", nargs="?", type=Path, default=DEFAULT_PLAN_PATH)
    apply_p.add_argument("--only", nargs="*", type=Path, default=None,
                         help="limit the apply to these paths")
    apply_p.add_argument("--force", action="store_true",
                         help="apply entries whose files changed since the plan was made")
    apply_p.add_argument("--no-backup", action="store_true")
    apply_p.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    apply_p.add_argument("-v", "--verbose", action="store_true")

    show = sub.add_parser("show", help="re-render a saved plan as a diff")
    show.add_argument("plan_file", nargs="?", type=Path, default=DEFAULT_PLAN_PATH)
    show.add_argument("-v", "--verbose", action="store_true")

    undo = sub.add_parser("undo", help="restore tags from a backup directory")
    undo.add_argument("paths", nargs="*", type=Path, help="limit the undo to these paths")
    undo.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    undo.add_argument("-v", "--verbose", action="store_true")

    doctor = sub.add_parser("doctor", help="check dependencies, providers and the LLM endpoint")
    doctor.add_argument("--llm-url", default=os.environ.get("AUTOTAGGER_LLM_URL", "http://localhost:11434"))
    doctor.add_argument("--llm-model", default=os.environ.get("AUTOTAGGER_LLM_MODEL", "qwen3:latest"))
    doctor.add_argument("--llm-api", choices=["auto", "ollama", "openai"], default="auto")
    doctor.add_argument("--ollama-key", default=os.environ.get("OLLAMA_API_KEY"))

    cache = sub.add_parser("clear-cache", help="empty the HTTP response cache")
    cache.add_argument("-v", "--verbose", action="store_true")

    return p


def _add_tag_args(t: argparse.ArgumentParser, for_plan: bool = False) -> None:
    # `plan` never writes, so the write-time flags would only confuse its help.
    hide = argparse.SUPPRESS if for_plan else None
    t.add_argument("paths", nargs="+", type=Path, help="files or directories to tag")

    g = t.add_argument_group("what to write")
    g.add_argument("--write", action="store_true",
                   help=hide or "apply changes (default is a dry run that only prints a diff)")
    g.add_argument("--overwrite", action="store_true",
                   help="replace tag values that are already set (default: only fill blanks)")
    g.add_argument("--fields", type=str, default=None,
                   help="comma-separated whitelist of tags to write, e.g. 'title,artist,album'")
    g.add_argument("--rename", type=str, default=None,
                   help="rename files from a template, e.g. '{track:02d} - {title}'")
    g.add_argument("--artist-style", choices=["list", "primary", "keep"], default="list",
                   help="how to write a multi-artist credit. 'list' (default) writes each "
                        "act as a separate value so your library shows one entry per "
                        "artist; 'primary' credits the lead artist and moves the rest into "
                        "the title; 'keep' writes the joined string as the provider gave it")
    g.add_argument("--keep-packaging", action="store_true",
                   help="keep \"[Remastered]\"/\"(Deluxe Edition)\" noise in titles "
                        "(default: strip it)")
    g.add_argument("--no-backup", action="store_true",
                   help=hide or "do not snapshot prior tags")
    g.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR, help=hide)

    g = t.add_argument_group("what to scan")
    g.add_argument("--no-recursive", action="store_true")
    g.add_argument("--ext", type=str, default=None,
                   help="comma-separated extensions to include (default: all supported)")
    g.add_argument("--library-root", type=Path, default=None,
                   help="stop path-based inference at this directory")
    g.add_argument("--skip-tagged", action="store_true",
                   help="skip files that already have all core tags and artwork")

    g = t.add_argument_group("artwork")
    g.add_argument("--no-artwork", action="store_true")
    g.add_argument("--artwork-only", action="store_true",
                   help="embed cover art and nothing else — no text tags are touched")
    g.add_argument("--artwork-if-missing", action="store_true",
                   help="only add art when the file has none; never replace")
    g.add_argument("--artwork-size", type=int, default=1400,
                   help="max pixels on the long edge (default 1400; needs Pillow to downscale)")
    g.add_argument("--save-cover", action="store_true",
                   help="also write cover.jpg next to the file")

    g = t.add_argument_group("providers")
    g.add_argument("--country", default="US", help="iTunes storefront (default US)")
    g.add_argument("--rate", type=float, default=1.0,
                   help="max iTunes requests/second (Apple soft-limits around 20/min)")
    g.add_argument("--deezer", action="store_true",
                   help="search Deezer alongside Apple — better dance/electronic coverage, "
                        "returns ISRC, label and BPM (no key needed). Used automatically "
                        "as a fallback whenever Apple returns nothing")
    g.add_argument("--musicbrainz", action="store_true",
                   help="enrich matches with MusicBrainz IDs, ISRC and label")
    g.add_argument("--acoustid", action="store_true",
                   help="fingerprint audio first (needs fpcalc + ACOUSTID_API_KEY)")
    g.add_argument("--no-cache", action="store_true", help="bypass the HTTP cache")
    g.add_argument("--limit", type=int, default=25, help="candidates to consider per file")

    g = t.add_argument_group("matching")
    g.add_argument("--accept", type=float, default=0.88,
                   help="score at or above which a match is applied without asking")
    g.add_argument("--consider", type=float, default=0.62,
                   help="score below which a file is left alone entirely")

    g = t.add_argument_group("LLM")
    g.add_argument("--llm", action="store_true", help="use an LLM to resolve ambiguous files")
    g.add_argument("--llm-always", action="store_true",
                   help="consult the LLM on every file, including confident matches")
    g.add_argument("--llm-url", default=os.environ.get("AUTOTAGGER_LLM_URL", "http://localhost:11434"),
                   help="Ollama base URL, or an OpenAI-compatible URL ending in /v1")
    g.add_argument("--llm-model", default=os.environ.get("AUTOTAGGER_LLM_MODEL", "qwen3:latest"))
    g.add_argument("--llm-api", choices=["auto", "ollama", "openai"], default="auto")
    g.add_argument("--llm-key", default=os.environ.get("AUTOTAGGER_LLM_API_KEY"))
    g.add_argument("--llm-candidates", type=int, default=8,
                   help="how many candidates to show the model")
    g.add_argument("--llm-min-confidence", type=float, default=0.7,
                   help="ignore the model's pick below this self-reported confidence")
    g.add_argument("--llm-max-tokens", type=int, default=2048)
    g.add_argument("--llm-think", action="store_true",
                   help="let a hybrid reasoning model think before answering "
                        "(slower; disabled by default because the prompt already "
                        "specifies the procedure)")

    g = t.add_argument_group("web search")
    g.add_argument("--web-search", action="store_true",
                   help="for files no catalogue contains, search the web and have the LLM "
                        "identify them. Requires --llm and an Ollama API key")
    g.add_argument("--ollama-key", default=os.environ.get("OLLAMA_API_KEY"),
                   help="Ollama API key for web search (default: $OLLAMA_API_KEY)")
    g.add_argument("--web-results", type=int, default=5,
                   help="search results to feed the model (max 10)")
    g.add_argument("--web-trust", action="store_true",
                   help="allow web-derived metadata to be applied automatically. Off by "
                        "default: web text carries no duration to verify against, so such "
                        "matches are held below the acceptance bar for confirmation")

    g = t.add_argument_group("interaction and output")
    g.add_argument("-i", "--interactive", action="store_true",
                   help="prompt for a choice when the match is ambiguous")
    g.add_argument("--json", dest="json_report", type=Path, default=None,
                   help="write a machine-readable report here")
    g.add_argument("-j", "--workers", type=int, default=4,
                   help="parallel files (forced to 1 in interactive mode)")
    g.add_argument("-v", "--verbose", action="store_true")


EPILOG = """\
examples:
  autotagger tag ~/Music/Unsorted
      Dry run. Prints exactly what would change, writes nothing.

  autotagger tag ~/Music --write --overwrite --musicbrainz
      Full retag, replacing existing values, adding MusicBrainz IDs and ISRCs.

  autotagger tag ~/Music --llm --llm-model qwen3:latest --write
      Local LLM breaks ties the scorer can't. Requires `ollama serve`.

  autotagger tag ~/Music --deezer --write
      Also search Deezer. Better for dance/electronic; returns ISRC and label.

  autotagger tag ~/Downloads --llm --web-search
      For tracks no store carries: search the web and identify them.
      Needs OLLAMA_API_KEY.

  autotagger tag ~/Music -i --llm
      LLM first; anything it also can't settle gets handed to you.

  autotagger tag ~/Music --write --rename '{track:02d} - {title}'
      Tag, then rename each file from its new metadata.

  autotagger undo
      Put back the tags from the most recent run's backups.
"""


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_tag(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    results = _run(config, args)
    if results is None:
        return 1

    render_summary(results, config.dry_run)
    if config.json_report:
        write_json_report(results, config.json_report)

    failed = sum(1 for r in results if r.status == "failed")
    return 2 if failed else 0


def _run(config: Config, args: argparse.Namespace):
    """Scan, resolve and report. Returns the results, or None if nothing to do."""
    files = collect_files(config)
    if not files:
        console.print("[yellow]No audio files found.[/yellow]")
        return None
    console.print(f"[dim]{len(files)} audio files[/dim]")

    if config.web_search and not config.ollama_api_key:
        from .websearch import OllamaWebSearch

        console.print("[yellow]--web-search needs an Ollama API key.[/yellow]")
        console.print(f"[dim]{OllamaWebSearch().setup_hint()}[/dim]")
        console.print("[yellow]Continuing without web search.[/yellow]")

    tagger = Tagger(config, interactive_chooser=interactive_chooser if args.interactive else None)
    if config.llm:
        ok, message = tagger.llm.health() if tagger.llm else (False, "no client")
        style = "green" if ok else "red"
        console.print(f"[{style}]LLM: {message}[/{style}]")
        if not ok:
            console.print("[yellow]Continuing without the LLM.[/yellow]")
            tagger.llm = None

    results = []
    try:
        if config.workers > 1:
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                for result in pool.map(tagger.process, files):
                    results.append(result)
                    render_result(result, config.verbose)
        else:
            for path in files:
                result = tagger.process(path)
                results.append(result)
                render_result(result, config.verbose)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        tagger.close()

    return results


def _config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config from the shared `tag`/`plan` argument set."""
    extensions = None
    if args.ext:
        extensions = {e if e.startswith(".") else f".{e}" for e in
                      (x.strip().lower() for x in args.ext.split(",")) if e}

    return Config(
        paths=args.paths,
        recursive=not args.no_recursive,
        extensions=extensions,
        library_root=args.library_root,
        skip_tagged=args.skip_tagged,
        dry_run=not args.write,
        overwrite_existing=args.overwrite,
        fields={f.strip() for f in args.fields.split(",")} if args.fields else None,
        rename=args.rename,
        clean_titles=not args.keep_packaging,
        artist_style=args.artist_style,
        backup_dir=None if args.no_backup else args.backup_dir,
        country=args.country,
        itunes_rate=args.rate,
        use_deezer=args.deezer,
        use_musicbrainz=args.musicbrainz,
        use_acoustid=args.acoustid,
        search_limit=args.limit,
        no_cache=args.no_cache,
        artwork=not args.no_artwork,
        artwork_only=args.artwork_only,
        artwork_if_missing=args.artwork_if_missing,
        artwork_max_px=args.artwork_size,
        save_cover_file=args.save_cover,
        thresholds=Thresholds(accept=args.accept, consider=args.consider),
        llm=args.llm or args.llm_always or args.web_search,
        llm_always=args.llm_always,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        llm_api=args.llm_api,
        llm_api_key=args.llm_key,
        llm_max_candidates=args.llm_candidates,
        llm_min_confidence=args.llm_min_confidence,
        llm_max_tokens=args.llm_max_tokens,
        llm_think=args.llm_think,
        web_search=args.web_search,
        ollama_api_key=args.ollama_key,
        web_results=args.web_results,
        web_trust=args.web_trust,
        interactive=args.interactive,
        json_report=args.json_report,
        verbose=args.verbose,
        workers=1 if args.interactive else max(1, args.workers),
    )



def _settings_for(config: Config) -> dict:
    """The knobs that shaped a plan, recorded in it for reference."""
    return {
        "country": config.country,
        "overwrite": config.overwrite_existing,
        "clean_titles": config.clean_titles,
        "artist_style": config.artist_style,
        "artwork": config.artwork,
        "artwork_only": config.artwork_only,
        "artwork_max_px": config.artwork_max_px,
        "deezer": config.use_deezer,
        "musicbrainz": config.use_musicbrainz,
        "acoustid": config.use_acoustid,
        "llm": config.llm,
        "llm_model": config.llm_model if config.llm else None,
        "web_search": config.web_search,
        "rename": config.rename,
        "accept": config.thresholds.accept,
        "consider": config.thresholds.consider,
    }


def cmd_plan(args: argparse.Namespace) -> int:
    """Resolve everything, then freeze it. Never writes to the audio files."""
    config = _config_from_args(args)
    config.dry_run = True  # a plan resolves; only `apply` writes

    results = _run(config, args)
    if results is None:
        return 1

    plan = build_plan(results, _settings_for(config), args.out, time.time())
    render_plan_summary(plan, args.out, artwork_dir_for(args.out))
    if config.json_report:
        write_json_report(results, config.json_report)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        plan = Plan.load(args.plan_file)
    except PlanError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(
        f"[dim]plan {args.plan_file} — written {plan.to_json()['created_at_iso']} "
        f"by autotagger {plan.tool_version}[/dim]\n"
    )
    render_plan(plan, args.verbose)
    problems = validate_plan(plan)
    if problems:
        console.print("\n[red]Problems in this plan:[/red]")
        for problem in problems:
            console.print(f"  [red]•[/red] {problem}")
        return 1

    enabled = sum(1 for e in plan.entries if e.apply)
    console.print(
        f"\n[bold]{enabled}[/bold] of [bold]{len(plan.entries)}[/bold] entries enabled"
        + (f", {len(plan.unmatched)} unmatched" if plan.unmatched else "")
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    try:
        plan = Plan.load(args.plan_file)
    except PlanError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    problems = validate_plan(plan)
    if problems:
        console.print(f"[red]{args.plan_file} has problems — nothing was written:[/red]")
        for problem in problems:
            console.print(f"  [red]•[/red] {problem}")
        return 1

    if not plan.entries:
        console.print("[yellow]Plan contains no changes.[/yellow]")
        return 0

    outcomes = apply_plan(
        plan,
        args.plan_file,
        force=args.force,
        backup_dir=None if args.no_backup else args.backup_dir,
        only=args.only,
    )
    return render_apply(outcomes)


def cmd_undo(args: argparse.Namespace) -> int:
    from .audio import write_tags

    backup_dir: Path = args.backup_dir
    if not backup_dir.exists():
        console.print(f"[red]No backup directory at {backup_dir}[/red]")
        return 1

    wanted = [p.resolve() for p in (args.paths or [])]
    restored = failed = 0
    for entry in sorted(backup_dir.glob("*.json")):
        try:
            payload = json.loads(entry.read_text())
            path = Path(payload["path"])
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if wanted and not any(path == w or w in path.parents for w in wanted):
            continue
        if not path.exists():
            console.print(f"[dim]missing, skipped: {path}[/dim]")
            continue
        try:
            # A file that had no cover before the run must not keep the one
            # autotagger embedded.
            write_tags(path, payload["tags"], clear_artwork=not payload.get("had_artwork"))
            entry.unlink()
            restored += 1
            console.print(f"[green]restored[/green] {path.name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            console.print(f"[red]failed[/red] {path.name}: {exc}")

    console.print(f"\n[bold]{restored} restored[/bold]" + (f", {failed} failed" if failed else ""))
    return 2 if failed else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .artwork import HAVE_PILLOW
    from .fingerprint import fpcalc_available, install_hint
    from .httpcache import Fetcher, HttpCache
    from .llm import LLMClient, LLMConfig

    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "[green]✓[/green]" if good else "[yellow]○[/yellow]"
        console.print(f"  {mark} {label}" + (f" [dim]— {detail}[/dim]" if detail else ""))
        ok = ok and good

    console.print("[bold]autotagger doctor[/bold]\n")
    console.print("[bold]core[/bold]")
    import mutagen
    check("mutagen", True, f"v{mutagen.version_string}")
    check("Pillow (artwork downscaling)", HAVE_PILLOW,
          "optional — without it, art is embedded at source resolution")

    console.print("\n[bold]providers[/bold]")
    fetcher = Fetcher(cache=HttpCache(enabled=False))
    try:
        data = fetcher.get_json(
            "https://itunes.apple.com/search",
            {"term": "daft punk one more time", "media": "music", "entity": "song", "limit": "1"},
        )
        hit = (data or {}).get("results", [{}])
        check("iTunes Search API", bool(data and data.get("resultCount")),
              hit[0].get("trackName", "") if hit else "no results")
    finally:
        fetcher.close()

    fetcher2 = Fetcher(cache=HttpCache(enabled=False))
    try:
        dz = fetcher2.get_json("https://api.deezer.com/search",
                               {"q": "daft punk one more time", "limit": "1"})
        hits = (dz or {}).get("data") or []
        check("Deezer API", bool(hits),
              hits[0]["title"] if hits else "no results")
    finally:
        fetcher2.close()

    check("fpcalc (AcoustID fingerprinting)", fpcalc_available(),
          "optional" if fpcalc_available() else install_hint().splitlines()[1].strip())
    check("ACOUSTID_API_KEY", bool(os.environ.get("ACOUSTID_API_KEY")),
          "optional — needed only with --acoustid")

    console.print("\n[bold]web search[/bold]")
    from .websearch import OllamaWebSearch

    searcher = OllamaWebSearch(api_key=getattr(args, "ollama_key", None))
    if not searcher.configured:
        check("Ollama API key", False, "optional — needed only with --web-search")
        console.print(f"[dim]{searcher.setup_hint()}[/dim]")
    else:
        try:
            results = searcher.search("Kazan One Last Time Dirty Workz")
            check("Ollama web search", bool(results), f"{len(results)} results")
        except Exception as exc:  # noqa: BLE001
            check("Ollama web search", False, str(exc))
    searcher.close()

    console.print("\n[bold]LLM[/bold]")
    client = LLMClient(LLMConfig(url=args.llm_url, model=args.llm_model, api=args.llm_api))
    healthy, message = client.health()
    check(f"{client.api} endpoint", healthy, message)
    if healthy:
        try:
            probe = client.complete_json(
                [{"role": "user", "content": 'Reply with exactly {"ok": true}'}],
                {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            )
            check("structured JSON output", probe.get("ok") is True, str(probe))
        except Exception as exc:  # noqa: BLE001
            check("structured JSON output", False, str(exc))
    client.close()

    console.print()
    return 0 if ok else 0  # doctor reports, it does not fail the shell


def cmd_clear_cache(args: argparse.Namespace) -> int:
    from .httpcache import HttpCache

    cache = HttpCache()
    n = cache.clear()
    console.print(f"Cleared {n} cached responses from {cache.path}")
    return 0


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def collect_files(config: Config) -> list[Path]:
    extensions = config.extensions or AUDIO_EXTENSIONS
    out: list[Path] = []
    seen: set[Path] = set()
    for root in config.paths:
        root = root.expanduser()
        if root.is_file():
            if root.suffix.lower() in extensions:
                out.append(root)
            continue
        if not root.exists():
            console.print(f"[yellow]not found: {root}[/yellow]")
            continue
        walker = root.rglob("*") if config.recursive else root.glob("*")
        for path in walker:
            if not path.is_file() or path.name.startswith("._"):
                continue
            if path.suffix.lower() not in extensions:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `autotagger ~/Music` is shorthand for `autotagger tag ~/Music`.
    known = {"tag", "plan", "apply", "show", "undo", "doctor", "clear-cache"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv.insert(0, "tag")

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    handlers = {
        "tag": cmd_tag, "plan": cmd_plan, "apply": cmd_apply, "show": cmd_show,
        "undo": cmd_undo, "doctor": cmd_doctor, "clear-cache": cmd_clear_cache,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted.[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
