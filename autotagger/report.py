"""Terminal output: the dry-run diff, the interactive picker, the run summary."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .models import AudioFile, FileResult, ScoredCandidate

console = Console()

_STATUS_STYLE = {
    "applied": "bold green",
    "dry-run": "bold cyan",
    "skipped": "dim",
    "no-match": "bold yellow",
    "failed": "bold red",
}


def render_result(result: FileResult, verbose: bool = False) -> None:
    style = _STATUS_STYLE.get(result.status, "white")
    header = Text()
    header.append(f"{result.status:>8} ", style=style)
    header.append(str(result.path.name))

    if result.status in ("skipped", "failed") and not verbose:
        if result.error:
            header.append(f"  — {result.error}", style="dim")
        console.print(header)
        return

    decision = result.decision
    body = []
    if decision:
        conf = decision.confidence
        conf_style = "green" if conf >= 0.88 else "yellow" if conf >= 0.62 else "red"
        line = Text("  match: ", style="dim")
        line.append(
            decision.chosen.label_line() if decision.chosen else "(none)",
            style="bold" if decision.chosen else "dim",
        )
        line.append(f"   {conf:.0%} via {decision.method}", style=conf_style)
        body.append(line)
        if decision.reasoning and (verbose or decision.method in ("llm", "none")):
            body.append(Text(f"  why: {decision.reasoning}", style="dim italic"))
        for warning in decision.warnings:
            body.append(Text(f"  ! {warning}", style="yellow"))
        if verbose and decision.runners_up:
            for runner in decision.runners_up[:3]:
                body.append(
                    Text(f"    alt {runner.score:.2f}  {runner.candidate.label_line()}", style="dim")
                )

    if result.changes:
        table = Table(show_header=False, box=None, padding=(0, 1, 0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="red dim", overflow="fold")
        table.add_column(style="dim", no_wrap=True)
        table.add_column(style="green", overflow="fold")
        for change in result.changes:
            table.add_row(change.field, _fmt(change.old), "→", _fmt(change.new))
        body.append(table)

    if result.artwork:
        body.append(Text(f"  artwork: {result.artwork}", style="magenta"))
    if result.error:
        body.append(Text(f"  error: {result.error}", style="red"))

    console.print(header)
    for item in body:
        console.print(item)


def _fmt(value) -> str:
    if value is None or value == "":
        return "∅"
    return str(value)


def render_summary(results: list[FileResult], dry_run: bool) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    table = Table(title=None, box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(justify="right", style="bold")
    table.add_column()
    for status in ("applied", "dry-run", "skipped", "no-match", "failed"):
        if counts.get(status):
            table.add_row(str(counts[status]), Text(status, style=_STATUS_STYLE[status]))

    footer = Text()
    if dry_run:
        footer.append("\nDry run — nothing was written. Re-run with ", style="yellow")
        footer.append("--write", style="bold yellow")
        footer.append(" to apply.", style="yellow")

    console.print(Panel(Group(table, footer) if footer.plain else table,
                        title=f"[bold]{len(results)} files[/bold]", border_style="dim"))

    problems = [r for r in results if r.status in ("no-match", "failed")]
    if problems:
        console.print("\n[bold yellow]Needs attention[/bold yellow]")
        for r in problems[:20]:
            reason = r.error or (r.decision.reasoning if r.decision else "")
            console.print(f"  [dim]{r.path}[/dim]\n    {reason}")
        if len(problems) > 20:
            console.print(f"  [dim]… and {len(problems) - 20} more[/dim]")


def write_json_report(results: list[FileResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_json() for r in results], indent=2, default=str))
    console.print(f"[dim]JSON report written to {path}[/dim]")


def interactive_chooser(af: AudioFile, scored: list[ScoredCandidate]) -> ScoredCandidate | None:
    """Show the shortlist and let the user pick. Returns None to skip the file."""
    console.print()
    console.rule(f"[bold]{af.path.name}[/bold]", style="cyan")

    facts = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    facts.add_column(style="dim", justify="right")
    facts.add_column()
    if af.duration:
        facts.add_row("duration", f"{int(af.duration) // 60}:{int(af.duration) % 60:02d}")
    for label, value in (
        ("tag title", af.title), ("tag artist", af.artist), ("tag album", af.album),
        ("from path", af.guessed_title), ("path artist", af.guessed_artist),
        ("path album", af.guessed_album),
    ):
        if value:
            facts.add_row(label, str(value))
    console.print(facts)

    table = Table(box=None, padding=(0, 1, 0, 0))
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("score", width=6)
    table.add_column("candidate", overflow="fold")
    table.add_column("Δt", width=7, justify="right")
    for i, sc in enumerate(scored):
        c = sc.candidate
        delta = ""
        if af.duration and c.duration:
            delta = f"{c.duration - af.duration:+.0f}s"
        style = "green" if sc.score >= 0.88 else "yellow" if sc.score >= 0.62 else "red"
        table.add_row(str(i), Text(f"{sc.score:.2f}", style=style), c.label_line(), delta)
    console.print(table)

    answer = Prompt.ask(
        "  [bold]Pick a number[/bold], or [bold]s[/bold]kip, or [bold]q[/bold]uit",
        default="s",
        show_default=False,
    ).strip().lower()
    if answer in ("q", "quit"):
        raise KeyboardInterrupt
    if answer in ("", "s", "skip"):
        return None
    try:
        idx = int(answer)
    except ValueError:
        return None
    return scored[idx] if 0 <= idx < len(scored) else None


# --------------------------------------------------------------------------
# Plan / apply rendering
# --------------------------------------------------------------------------

def render_plan_summary(plan, plan_path, artwork_dir) -> None:
    """Terraform-style closing summary for `plan`."""
    changing = len(plan.entries)
    with_art = sum(1 for e in plan.entries if e.artwork)
    renames = sum(1 for e in plan.entries if e.rename_to)
    low = [e for e in plan.entries if e.confidence < 0.88]

    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_row(str(changing), Text("files to change", style="cyan"))
    if with_art:
        table.add_row(str(with_art), Text("covers pinned", style="magenta"))
    if renames:
        table.add_row(str(renames), Text("renames", style="cyan"))
    # "already correct" and "could not be identified" are very different
    # outcomes; lumping both under "no match" makes a clean library look broken.
    up_to_date = sum(1 for u in plan.unmatched if u.get("status") == "skipped")
    no_match = len(plan.unmatched) - up_to_date
    if up_to_date:
        table.add_row(str(up_to_date), Text("already up to date", style="green"))
    if no_match:
        table.add_row(str(no_match), Text("no match — untouched", style="yellow"))

    footer = Text()
    footer.append("\nPlan written to ", style="dim")
    footer.append(str(plan_path), style="bold")
    if with_art:
        footer.append(f"\nCover art pinned in {artwork_dir.name}/", style="dim")
    from .plan import DEFAULT_PLAN_PATH

    command = "autotagger apply"
    if plan_path != DEFAULT_PLAN_PATH:
        command += f" {plan_path}"
    footer.append("\nReview it, edit it if you like, then run:\n", style="dim")
    footer.append(f"  {command}\n", style="bold green")
    footer.append("Apply makes no network or LLM calls — it writes exactly this.", style="dim")

    console.print(Panel(Group(table, footer), title="[bold]Plan[/bold]", border_style="cyan"))

    if low:
        console.print(
            f"\n[yellow]{len(low)} entr{'y' if len(low) == 1 else 'ies'} below 0.88 confidence"
            "[/yellow] [dim]— worth a look before applying:[/dim]"
        )
        for e in low[:10]:
            console.print(f"  [dim]{e.confidence:.0%}[/dim] {e.path.name} → {e.match}")
        if len(low) > 10:
            console.print(f"  [dim]… and {len(low) - 10} more[/dim]")


def render_plan(plan, verbose: bool = False) -> None:
    """Re-render a saved plan as a diff, for `autotagger show`."""
    for entry in plan.entries:
        header = Text()
        header.append("  change ", style="bold cyan")
        header.append(entry.path.name)
        console.print(header)

        line = Text("  match: ", style="dim")
        line.append(entry.match or "(none)", style="bold")
        conf_style = "green" if entry.confidence >= 0.88 else "yellow"
        line.append(f"   {entry.confidence:.0%} via {entry.method}", style=conf_style)
        console.print(line)
        if verbose and entry.reasoning:
            console.print(Text(f"  why: {entry.reasoning}", style="dim italic"))
        for warning in entry.warnings:
            console.print(Text(f"  ! {warning}", style="yellow"))

        table = Table(show_header=False, box=None, padding=(0, 1, 0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="red dim", overflow="fold")
        table.add_column(style="dim", no_wrap=True)
        table.add_column(style="green", overflow="fold")
        for field_name, new in entry.changes.items():
            table.add_row(field_name, _fmt(entry.previous.get(field_name)), "→", _fmt(new))
        if entry.changes:
            console.print(table)
        if entry.artwork:
            console.print(Text(
                f"  artwork: {entry.artwork.width}x{entry.artwork.height} "
                f"[{entry.artwork.digest}]", style="magenta"))
        if entry.rename_to:
            console.print(Text(f"  rename: → {entry.rename_to}", style="cyan"))

    for item in plan.unmatched:
        label = "up-to-date" if item.get("status") == "skipped" else "no-match"
        style = "green" if label == "up-to-date" else "yellow"
        console.print(Text(f"{label} {Path(item['path']).name}", style=style))
        if verbose and item.get("reason"):
            console.print(Text(f"  {item['reason']}", style="dim"))


_APPLY_STYLE = {
    "applied": "bold green", "drift": "bold yellow",
    "missing": "yellow", "failed": "bold red", "skipped": "dim",
}


def render_apply(outcomes: list) -> int:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
        style = _APPLY_STYLE.get(o.status, "white")
        line = Text()
        line.append(f"{o.status:>8} ", style=style)
        line.append(o.path.name)
        if o.detail:
            line.append(f"  — {o.detail}", style="dim")
        console.print(line)

    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(justify="right", style="bold")
    table.add_column()
    for status in ("applied", "drift", "missing", "failed", "skipped"):
        if counts.get(status):
            table.add_row(str(counts[status]), Text(status, style=_APPLY_STYLE[status]))
    console.print(Panel(table, title="[bold]Apply[/bold]", border_style="dim"))

    if counts.get("drift"):
        console.print(
            "[yellow]Some files changed since the plan was made.[/yellow] "
            "[dim]Re-run `autotagger plan` to pick up their current state, "
            "or `autotagger apply --force` to overwrite them anyway.[/dim]"
        )
    return 2 if counts.get("failed") else 0
