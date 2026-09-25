"""Run configuration — one object threaded through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .matching import Thresholds


@dataclass
class Config:
    # --- what to process ---
    paths: list[Path] = field(default_factory=list)
    recursive: bool = True
    extensions: set[str] | None = None
    library_root: Path | None = None
    skip_tagged: bool = False           # skip files that already look complete

    # --- writing ---
    dry_run: bool = True
    overwrite_existing: bool = False    # replace tags that already have a value
    fields: set[str] | None = None      # restrict which tags may be written
    rename: str | None = None           # e.g. "{track:02d} - {title}"
    clean_titles: bool = True           # strip "[Remastered]"-style packaging noise
    artist_style: str = "list"          # list | primary | keep
    backup_dir: Path | None = None      # sidecar JSON of prior tags, for --undo

    # --- providers ---
    country: str = "US"
    itunes_rate: float = 1.0
    use_deezer: bool = False            # search Deezer alongside Apple
    use_musicbrainz: bool = False
    use_acoustid: bool = False
    acoustid_key: str | None = None
    search_limit: int = 25
    no_cache: bool = False

    # --- artwork ---
    artwork: bool = True
    artwork_only: bool = False          # embed cover art, write no other tags
    artwork_if_missing: bool = False
    artwork_max_px: int = 1400
    save_cover_file: bool = False
    cover_filename: str = "cover.jpg"

    # --- matching ---
    thresholds: Thresholds = field(default_factory=Thresholds)

    # --- LLM ---
    llm: bool = False
    llm_always: bool = False            # consult the LLM even on confident matches
    llm_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:latest"
    llm_api: str = "auto"
    llm_api_key: str | None = None
    llm_max_candidates: int = 8
    llm_min_confidence: float = 0.7     # below this, the LLM's pick is not applied
    llm_max_tokens: int = 2048
    llm_think: bool = False             # hybrid reasoning models: think before answering

    # --- web search (needs an LLM and an Ollama API key) ---
    web_search: bool = False
    ollama_api_key: str | None = None
    web_results: int = 5
    web_trust: bool = False             # allow web-derived tags to auto-apply

    # --- interaction / output ---
    interactive: bool = False
    json_report: Path | None = None
    verbose: bool = False
    workers: int = 4

    def may_write(self, field_name: str) -> bool:
        return self.fields is None or field_name in self.fields
