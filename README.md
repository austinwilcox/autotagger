# autoTagger

Tags music files from the Apple/iTunes catalogue — metadata *and* album art — with
a deterministic matcher that does the heavy lifting and an optional local LLM for
the cases where deterministic scoring genuinely can't decide.

Built on the iTunes Search API, the same source as
[album-art-alchemist](../album-art-alchemist).

```
autotagger tag ~/Music                 # dry run — prints a diff, writes nothing
autotagger tag ~/Music --write         # apply
autotagger tag ~/Music --llm --write   # local LLM breaks the ties
autotagger tag ~/Music --interactive   # you break the ties

autotagger plan ~/Music --llm          # freeze the decisions into a file
autotagger apply                       # execute it, exactly
```

---

## Why it gets the right answer

Most taggers match on title and artist strings. That is why they cheerfully write
the studio album's metadata onto a live recording, or credit a karaoke release.

**Artist and duration carry the match; title barely counts.** Hundreds of
unrelated songs are called "One Last Time", and most of them run three minutes —
so a title hit plus a duration hit happens by coincidence constantly. Artist is
weighted highest (30%), and an artist that doesn't appear among the candidates
collapses the score by 80%, whatever the title says.

Fuzzy string matching is the trap here: `WRatio("kazan", "ariana grande")` is
**54**, so a naive scorer hands a totally unrelated act half credit. Names below
a similarity floor score zero, not "partly right".

**Duration is the tiebreaker that can't lie.** A title can be spelled five ways;
3:37 is 3:37. Weighted at 22%, and a miss beyond 12 seconds multiplies the whole
score down by 60–75% — positive evidence of a *different recording*.

**Not every file is in the catalogue.** Independent releases, copyright-free
label music (Dirty Workz, NCS, Epidemic Sound), bootlegs, DAW exports and
unreleased tracks are simply absent, and for those the only right answer is *no
match*. autoTagger reads the file's extended metadata — encoder, publisher,
comment, URL, BPM — as evidence for exactly this: an encoder string of
`LAME in FL Studio 2026` means a producer's own export, not a commercial master.

On top of that:

| Problem | How it's handled |
|---|---|
| `Kazan - One Last Time` vs LP's `One Last Time` | **An artist that isn't in the candidate list is disqualifying** (×0.2). Title is the weakest identifier there is — a shared title plus a coincidental runtime is not a match |
| `Song (Remastered 2011)` vs `Song` | Packaging words are stripped before comparison — same recording |
| `Song` vs `Song (Live)` | Version markers (live, remix, acoustic, demo, radio edit, …) are identity-bearing. A mismatch is a hard penalty, weighted by whether the file's own title is trustworthy |
| `Wonderwall` by "Karaoke Stars" | Soundalike/tribute/"in the style of"/8-bit/lullaby releases are detected and buried (×0.15) |
| `Drake` vs `Drake feat. Future` | Guest credits are parsed out of *both* the title and artist fields and compared as a set. Position doesn't matter |
| `Björk` vs `Bjork` | Unicode folding, `&`↔`and`, smart-quote normalization |
| `OK Computer` vs `OKNOTOK 1997 2017` | Reissues and deluxe editions are demoted so the original album wins a tie |
| `03 - Title.mp3` in `Artist/Album (1997)/` | Track number, disc number, album, year and artist are all inferred from the path |
| `04-rhcp-cant_stop-(official_video)-[HQ].mp3` | Scene tags, source junk and YouTube IDs are stripped before searching |

Every signal that's missing is **redistributed**, not counted as zero — a file
with only a title and a duration is scored on exactly those two things.

### Confidence bands

```
score ≥ 0.88 ............. applied automatically
0.62 ≤ score < 0.88 ...... ambiguous → LLM, or you, or left alone
score < 0.62 ............. no match, file untouched
```

Also accepted: anything at or above 0.62 that beats its runner-up by 0.15. Tune
with `--accept` and `--consider`.

---

## The LLM layer

Optional, off by default, and **never the primary source**. The deterministic
matcher resolves the large majority of files on its own; the model sees only
what it couldn't settle.

It does three jobs:

1. **Selection.** Given the file's evidence and the shortlist of candidates,
   pick the right one — or say none of them is right.
2. **Query repair.** When a filename is too mangled to search
   (`04-rhcp-cant_stop-(official_video)-[HQ].mp3`), propose search strings that
   will actually hit. Every proposal is verified against real catalogue data
   afterward, including the duration check, so a wrong guess here is free.
3. **Identification of files that don't match anything.** It reads the full
   metadata — encoder, publisher, comment, year, BPM — and tells you what the
   recording probably is. This is never written into the file:

   ```
   no-match Kazan - One Last Time.mp3
     ! identified as: Likely an independent or copyright-free hardstyle release
       by Kazan, produced in FL Studio in 2026. Labels such as Dirty Workz
       publish copyright-free tracks that are not in commercial catalogues.
   ```

**The LLM breaks ties; it never overrules a rejection.** Its pick must still
clear the same deterministic bar as everything else, and a `catalogue_plausible:
false` verdict is honoured even if the model contradicts itself by also naming a
candidate. A model that argues confidently for a wrong artist gets refused with
its own reasoning printed, rather than writing that artist into your library.

### The instructions

The prompts live in [`autotagger/prompts.py`](autotagger/prompts.py) and are
written for small local models. The design constraints:

- **The model selects; it does not recall.** Every factual value must be copied
  from a candidate it was shown. A 4B model's memory of a discography is
  confidently wrong often enough to corrupt a library, and the damage isn't
  visible for months. The only thing it's allowed to author is *formatting* —
  capitalization, `feat.` placement.
- **"None of these" is a first-class answer.** A refusal costs one untagged file.
  A confident wrong answer costs a corrupted library.
- **An explicit ordered procedure**, not a vibe. Duration first, then version
  markers, then soundalike rejection, then artist (including guests), then
  release preference, then track numbers.
- **Calibrated confidence with stated bands**, enforced by
  `--llm-min-confidence` (default 0.7) so inflating the number doesn't defeat
  the safety mechanism it drives.
- **Worked examples** for both prompts — including a *refusal*, because a prompt
  that only ever demonstrates selecting biases the model toward selecting. The
  negative example is the real Kazan case: perfect title, 0.2-second duration
  agreement, and the correct answer is still "none".
- **Strict JSON out.** On Ollama the schema is passed to `format`, which
  constrains decoding — small models return valid JSON every time. Elsewhere it
  falls back to `response_format: json_object` plus tolerant parsing that
  survives markdown fences, `<think>` blocks and surrounding prose.

Rules the prompt spells out in full: feat. formatting, title case with
stylization exceptions (`DAMN.`, `thank u, next`), no transliteration, classical
(performer vs composer), Various Artists and soundtracks, covers, explicit/clean,
and which parenthetical suffixes are identity vs packaging.

### Pointing it at a model

```bash
# Ollama (default — native API, schema-constrained decoding)
autotagger tag ~/Music --llm --llm-model qwen3:latest

# LM Studio / llama.cpp / vLLM / anything OpenAI-compatible
autotagger tag ~/Music --llm --llm-url http://localhost:1234/v1 --llm-model my-model

# Consult the model on every file, not just ambiguous ones
autotagger tag ~/Music --llm-always
```

Hybrid reasoning models (qwen3, gpt-oss, deepseek-r1) have thinking **disabled**
by default: the prompt already specifies the procedure step by step, and the
reasoning preamble otherwise eats the token budget before the JSON is finished.
`--llm-think` re-enables it.

Environment variables: `AUTOTAGGER_LLM_URL`, `AUTOTAGGER_LLM_MODEL`,
`AUTOTAGGER_LLM_API_KEY`.

---

## Multi-artist credits

A library fills up with junk artists — "Fox Stevenson", "Fox Stevenson & Cruk",
"Fox Stevenson & Cruk & Priority One" as three unrelated entries — because the
credit gets written as one joined string. By default autoTagger writes **each
act as a separate tag value**, so you get one entry per real artist:

```
artist        Fox Stevenson ∥ Yue      (two values, not one string)
album_artist  Fox Stevenson            (always the lead act — this is what
                                        players group albums by)
```

The hard part is not splitting; it is knowing when *not* to. "Simon &
Garfunkel", "Nick Cave & The Bad Seeds", "Earth, Wind & Fire", "Tyler, The
Creator" and "AC/DC" are single acts whose names contain the very separators a
naive split breaks on — and turning "Simon & Garfunkel" into "Simon" is far
worse than the problem being solved, because it silently renames an artist.

So splitting only happens on authoritative data:

| Source | How it resolves |
|---|---|
| **Apple** | The track's `artistId` is looked up. A collaboration credited "Fox Stevenson & Yue" has the artist entity **Fox Stevenson**, so the remainder is a guest. A band credited "Simon & Garfunkel" has the entity **Simon & Garfunkel**, so nothing is split |
| **Deezer** | `contributors` is already a structured list — one entry means one act, however many ampersands its name contains |
| **`feat.` / `ft.` / `featuring`** | Unambiguous, so split even with no provider data |
| **A bare `&` with no provider data** | Left alone. Conservative on purpose |

```bash
--artist-style list      # default: one tag value per act
--artist-style primary   # lead act only; the rest move into the title as "(feat. …)"
--artist-style keep      # write the joined string exactly as the provider gave it
```

Per format: MP4 and FLAC/Ogg store multiple values natively. MP3 is written as
**ID3v2.4** when there is more than one artist, because v2.3 separates values
with `/` and would split a name like AC/DC in half; single-artist files stay
v2.3. The Picard-convention `ARTISTS` tag is written alongside for players that
prefer it (Jellyfin, Navidrome, MusicBee).

## Plan and apply

A dry run is only a *prediction*. Between reading it and running `--write`, the
answer can change — a local LLM is not deterministic, the catalogue is not
frozen, and the HTTP cache eventually expires. What you reviewed is not
necessarily what gets written.

`plan` closes that gap. It does all the resolution — catalogue lookups, LLM
calls, artwork downloads — and freezes the outcome:

```bash
autotagger plan ~/Music --llm          # writes autotagger.plan.json
autotagger show                        # re-read it whenever
autotagger apply                       # execute it
```

Name the plan whatever you like — the pinned-artwork directory follows it:

```bash
autotagger plan ~/Music -o friday-cleanup.json
autotagger show friday-cleanup.json
autotagger apply friday-cleanup.json
```

`apply` with no argument looks for `autotagger.plan.json` **in the current
directory** — it does not search parent directories. Paths inside a plan are
absolute, so a plan can be applied from anywhere as long as you name it.

`plan` is idempotent: running it again after an apply reports *already up to
date* rather than re-proposing the same changes.

**`apply` makes no network calls, no provider calls and no LLM calls at all.**
It reads the plan and writes what the plan says, including the exact image bytes
that were fetched at plan time — those live in `autotagger.plan.json.artwork/`,
named and verified by digest. Applying with the network completely unreachable
works, artwork included.

### The plan is meant to be edited

It is plain JSON, and hand-editing is a supported workflow rather than a hack.
When most of a match is right and one field is wrong, fix that field instead of
throwing the whole match away. The rules travel inside the file, in `_readme`:

| To do this | Edit this |
|---|---|
| Correct a value | Change it under the entry's `changes` |
| Leave one tag alone | Delete that field from `changes` |
| Skip a whole file | Set the entry's `"apply": false` |
| Keep the existing cover | Set the entry's `"artwork": null` |
| Cancel a rename | Set `"rename_to": null` |

```jsonc
{
  "path": "lib/Daft Punk - One More Time.mp3",
  "apply": true,                    // flip to false to skip this file
  "fingerprint": "23a1fb088d4c5d6d",
  "changes": {
    "title": "One More Time",
    "genre": "Dance",               // edit freely — apply writes what's here
    "isrc": "GBDUW0000053"          // delete the line to leave ISRC untouched
  },
  "previous": { "title": null },    // context only, never written
  "artwork": { "digest": "c3ca31ab…", "width": 1400, "height": 1400 },
  "match": "Daft Punk — One More Time [Discovery] (2000) 5:20",
  "confidence": 0.965,
  "method": "deterministic"
}
```

`apply` validates before touching anything, so a typo gets you a clear message
naming the file and field rather than a half-finished run:

```
autotagger.plan.json has problems — nothing was written:
  • a.mp3: unknown tag 'titel'. Valid tags: album, album_artist, artist, …
  • a.mp3: track_number must be a whole number, got 'abc'
```

### Drift

Each entry records a fingerprint of the file as it was at plan time — size plus
its tag values, not mtime, so syncing or copying doesn't cause false alarms. If a
file changed in between, that entry is refused:

```
   drift Radiohead - Exit Music.flac — file changed since the plan was made —
         re-plan, or use --force to apply anyway
```

This also means applying the same plan twice is caught: the first apply changes
the files, so the second sees drift instead of silently rewriting them.

Apply a subset with `autotagger apply --only ~/Music/Radiohead`.

## Where the metadata comes from

| Source | Default | Key | What it's for |
|---|---|---|---|
| **Apple / iTunes Search** | always on | none | Primary. Best general coverage, genre, release dates, artwork |
| **Deezer** | `--deezer`, plus automatic fallback when Apple returns nothing | none | Much better dance/electronic coverage. Returns duration, **ISRC**, label, UPC and BPM — several of which Apple doesn't expose |
| **MusicBrainz** | `--musicbrainz` | none | Enrichment only: MBIDs, ISRC, label, barcode. Strict matching — an ambiguous hit is discarded, because a wrong MBID silently corrupts Picard/Plex/Jellyfin downstream |
| **AcoustID** | `--acoustid` | free | Acoustic fingerprint. Identifies from the audio itself rather than what someone typed |
| **Ollama web search** | `--web-search` | free | Last resort for tracks no store carries |

**Cross-provider agreement is evidence.** The same recording found by both Apple
and Deezer is merged into one candidate — blanks filled from whichever source
has them, so an Apple match picks up Deezer's ISRC — and scored slightly higher
for the corroboration. Originals and reissues are deliberately *not* merged, so
the reissue penalty still decides between "By the Way" and "By the Way (Deluxe
Edition)" rather than whichever provider answered first.

## Web search

For a track no catalogue contains — an independent release, a copyright-free
label track, a bootleg — the stores have nothing to find. `--web-search` searches
the web and has the LLM work out what the file is.

```bash
ollama signin
# create a key at https://ollama.com/settings/keys
export OLLAMA_API_KEY=...

autotagger ~/Downloads --llm --web-search
```

Two constraints shape how this is built:

**The model does not browse freely.** One search, one structured extraction.
Its preferred output is not metadata at all — it's a *better catalogue query*.
A web result often reveals the correct artist spelling or the compilation a
track appears on, and that query is re-run against Deezer and Apple, where the
result is verified by duration like anything else.

**Web text is untrusted input.** Page content sits in the same context as "write
these tags", so the extraction prompt states explicitly that results are data
rather than instructions, and that anything in them addressing the model is to
be noted and disregarded. Nothing here executes, follows or fetches what a page
asks for.

When the web is the *only* source, the candidate it produces is marked
`source="web"`, carries no duration to verify against, and is capped at 0.80 —
below the 0.88 acceptance bar — so it surfaces for confirmation rather than
being written silently. `--web-trust` lifts the cap if you want it automatic.

## Album art

Apple's search results return a 100×100 thumbnail URL, but the same path serves
much larger renditions. autoTagger rewrites the URL and walks down a ladder —
3000 → 1400 → 1200 → 600 — keeping the first one that is a real image of usable
size, then downscales to `--artwork-size` (default 1400px, needs Pillow). Deezer
covers work the same way on their own grammar, laddering 1800 → 1400 → 1000.

**Artwork only**, leaving every text tag untouched:

```bash
autotagger ~/Music --artwork-only --write
```

It still has to identify the track — that's where the cover comes from — so a
file that doesn't match gets no art.

Existing covers are only replaced by something **larger**; identical art is
detected by hash and skipped. `--artwork-if-missing` never replaces anything.
`--save-cover` also drops a `cover.jpg` beside the file for players that prefer
folder art.

---

## Install

Install it as a command you can run from anywhere:

```bash
cd ~/Software/autoTagger
uv tool install --editable --with pillow .

autotagger doctor                # check everything is wired up
```

That puts `autotagger` in `~/.local/bin`. `--editable` means edits to the source
take effect immediately — no reinstall. To remove it: `uv tool uninstall autotagger`.

For development (running the test suite against a local venv):

```bash
uv venv --python 3.12
uv pip install -e ".[dev]" pillow
.venv/bin/python -m pytest -q
```

Optional acoustic fingerprinting — the strongest signal available, because it
listens to the audio instead of reading what someone typed about it:

```bash
brew install chromaprint                        # provides fpcalc
export ACOUSTID_API_KEY=...                     # free: https://acoustid.org/new-application
autotagger tag ~/Music --acoustid
```

---

## Usage

### Safety

Dry run is the **default**. Nothing is written until you pass `--write`.

Existing tag values are **preserved** by default — only blanks are filled. Pass
`--overwrite` to replace them.

Every write snapshots the prior tags to `~/.autotagger/backups`:

```bash
autotagger undo                      # restore everything from the last run
autotagger undo ~/Music/Radiohead    # restore just this subtree
```

Undo also removes artwork from files that had none before.

### Common runs

```bash
# See what would happen, in detail
autotagger tag ~/Music/Unsorted -v

# Full retag including MusicBrainz IDs, ISRCs and label
autotagger tag ~/Music --write --overwrite --musicbrainz

# LLM first; anything it also can't settle gets handed to you
autotagger tag ~/Music -i --llm

# Only touch genre and artwork, leave everything else
autotagger tag ~/Music --write --overwrite --fields genre

# Tag, then rename from the new metadata
autotagger tag ~/Music --write --rename '{track:02d} - {title}'

# Machine-readable output for scripting
autotagger tag ~/Music --json report.json
```

Rename template fields: `{track}`, `{disc}`, `{title}`, `{artist}`, `{album}`,
`{album_artist}`, `{year}`, `{genre}`. Format specs work: `{track:02d}`.

### Rate limits

Apple soft-limits the public Search API to roughly 20 requests/minute per IP and
answers a throttled client with HTTP 403. autoTagger rate-limits (`--rate`,
default 1/s), backs off on 403/429, and caches every response for 30 days in
SQLite — so a re-run over the same library is nearly instant. `autotagger
clear-cache` empties it.

MusicBrainz's 1 req/s limit is honoured separately, with the identifying
User-Agent they require.

---

## What gets written

| Field | MP3 (ID3v2.3) | MP4 / M4A | FLAC / Ogg / Opus |
|---|---|---|---|
| title / artist / album | `TIT2` `TPE1` `TALB` | `©nam` `©ART` `©alb` | `title` `artist` `album` |
| album artist | `TPE2` | `aART` | `albumartist` |
| track, disc (+ totals) | `TRCK` `TPOS` | `trkn` `disk` | `tracknumber`/`tracktotal`, `discnumber`/`disctotal` |
| date / year | `TDRC` | `©day` | `date` |
| genre / composer | `TCON` `TCOM` | `©gen` `©wrt` | `genre` `composer` |
| compilation | `TCMP` | `cpil` | `compilation` |
| explicit | — | `rtng` | — |
| ISRC / copyright | `TSRC` `TCOP` | freeform / `cprt` | `isrc` `copyright` |
| MusicBrainz IDs | `TXXX` (Picard names) | `----:com.apple.iTunes:*` | `musicbrainz_*` |
| cover art | `APIC` (front cover) | `covr` | FLAC picture block / base64 `metadata_block_picture` |

Formats: `.mp3 .m4a .mp4 .m4b .flac .ogg .oga .opus .wav .aiff .aif .aifc`

Tags autoTagger doesn't manage are left untouched, so hand-curated ratings, play
counts and comments survive.

Titles have release packaging stripped before writing — `Exit Music (For a Film)
[Remastered]` becomes `Exit Music (For a Film)`. Genuine version markers
(`(Live)`, `(Acoustic)`, `(feat. …)`) and stylizations (`DAMN.`) are preserved.
`--keep-packaging` disables this. **Album names are never cleaned**: "By the Way
(Deluxe Edition)" has different track numbers than "By the Way", so renaming the
album while keeping the deluxe numbering would produce metadata that contradicts
itself.

---

## Architecture

```
cli.py          argument parsing, file scanning, subcommands
plan.py         plan format, drift fingerprinting, apply — no network at all
pipeline.py     per-file orchestration and the escalation ladder
  ├─ audio.py       mutagen read/write, one uniform interface per container
  ├─ parse.py       filename and folder-structure inference
  ├─ providers/     itunes.py (primary), deezer.py, musicbrainz.py (enrichment)
  ├─ websearch.py   Ollama web search + LLM extraction, for uncatalogued files
  ├─ fingerprint.py Chromaprint + AcoustID (optional)
  ├─ matching.py    the deterministic scorer — weights, penalties, thresholds
  ├─ normalize.py   the string surgery all of the above depends on
  ├─ artwork.py     URL ladder, header-only dimension parsing, embed policy
  ├─ llm.py         Ollama + OpenAI-compatible client, tolerant JSON extraction
  └─ prompts.py     the instructions
report.py       terminal diff, interactive picker, JSON report
httpcache.py    SQLite response cache + per-host token bucket
```

**The escalation ladder.** Each rung runs only when the one above came back
ambiguous, which keeps a large library fast and saves the slow or rate-limited
resources for the files that need them:

```
1. read tags + audio properties            local, free
2. parse filename and folder structure     local, free
3. acoustic fingerprint                    optional, local CPU + 1 API call
4. catalogue search (Apple, then Deezer)   rate-limited
5. deterministic scoring                   local, free   ← resolves most files
6. LLM query repair, then re-search        only when 4 found nothing
7. LLM selection                           only when 5 was ambiguous
8. web search + LLM identification         only when nothing matched at all
9. interactive prompt                      only when still ambiguous
```

---

## Tests

```bash
.venv/bin/python -m pytest -q
```

Unit tests cover normalization, path parsing, the scorer's penalty behaviour
(including regressions for live-vs-studio and original-vs-reissue), artwork
header parsing, and LLM output extraction. `tests/test_audio_roundtrip.py`
synthesizes a file per container with ffmpeg and round-trips every tag through
it — the per-format writer bugs are the ones that unit tests of the matcher
would never catch.
