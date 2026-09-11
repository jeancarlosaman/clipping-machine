# Clipping Machine

VOD-first AI clipping tool for streamers. Full design in
[`docs/architecture.md`](docs/architecture.md) -- read that first if
you're new here; this README is just "how do I run it."

## What's implemented

Started as a walking-skeleton pass; every pipeline stage below is now real,
not stubbed -- an approved clip can go all the way to a genuine TikTok
draft upload (see "TikTok setup" below for what that needs on TikTok's
side).

| Stage | Status |
|---|---|
| API (create job, list/get, review, upload, creator accounts) | Implemented |
| Ingest worker (ffprobe + audio extraction) | Implemented |
| Transcription worker | Implemented -- open-source local Whisper by default, hosted OpenAI API as a config-switchable option (see below) |
| Segmentation worker | Implemented -- PySceneDetect + transcript silence structure (see below) |
| Scoring worker | Implemented -- deterministic feature scoring + ranking, no LLM (see below) |
| Rendering worker | Implemented -- ffmpeg cut + center-crop to 9:16 + caption burn-in (see below) |
| Upload worker | Implemented -- real TikTok Content Posting API (inbox/draft) integration, OAuth included (see below) |

A job you submit today will progress all the way to `ready_for_review`-equivalent
state: `queued -> ingesting -> ingested -> transcribing -> transcribed ->
segmenting -> segmented -> scoring -> scored`, with one real, watchable
`.mp4` clip per selected candidate (`rendered_clips.status = 'rendered'`).
Nothing in the pipeline is a stub anymore -- an approved clip can go all
the way to a real TikTok draft upload, given a registered TikTok developer
app (see "TikTok setup" below; the app-review/audit step is TikTok-side
lead time, not something this code can shortcut). (There's no
`stream_job.status` value for "all clips rendered" yet -- see "Next
steps.")

## Prerequisites

- Python 3.11+
- PostgreSQL 16 and Redis 7 (via Docker, or installed locally)
- ffmpeg / ffprobe on PATH (`ffmpeg -version` to check)
- (only if `STT_PROVIDER=openai`) `espeak-ng` is **not** required for the app
  itself -- it's only used by one test to synthesize speech audio; skip
  installing it if you're not running `tests/test_local_whisper_provider.py`

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env if your Postgres/Redis aren't on the defaults

docker compose up -d          # starts Postgres + Redis
# (or point DATABASE_URL / REDIS_URL in .env at your own instances)

alembic upgrade head           # applies the schema
```

Everything below assumes you're in `backend/` with the venv active and
`PYTHONPATH=.` set (or run commands as `python -m ...`) so `app.*` imports
resolve.

## Running it

### Windows: one command (`start.ps1`)

```powershell
cd backend
.\start.ps1
```

or double-click `backend\start.bat` (same thing, no execution-policy setup
needed -- the .bat bypasses it for that one process).

`start.ps1` does the whole sequence and stops with a specific error if any
step fails:

| Step | What it does |
| --- | --- |
| Docker | Starts Docker Desktop if the engine isn't answering, waits up to 180s |
| Containers | `docker compose up -d`, then waits until Postgres answers `pg_isready` and Redis answers `PING` -- not just until they're "created" |
| venv | Creates `venv\` and installs `requirements.txt` on first run |
| `.env` | Copies from `.env.example` if missing |
| Migrations | `alembic upgrade head` |
| Ollama | Starts `ollama serve` if installed and port 11434 is free; warns if the model in `CAPTION_OLLAMA_MODEL` was never pulled |
| API | Own window, waits for `/health` to return 200 |
| Worker | Own window |
| Token | Runs `create_dev_user.py`, prints the bearer token and copies it to your clipboard |
| Browser | Opens the dev console |

The API and worker each get their own titled window so their logs stay
readable and separable -- that part is deliberate, not laziness: when a job
fails you want the worker's traceback on its own, and `--reload` output not
interleaved with it.

Useful switches:

```powershell
.\start.ps1 -NoBrowser        # don't open the browser
.\start.ps1 -SkipOllama       # don't start/check Ollama
.\start.ps1 -SkipMigrations   # skip alembic (faster restarts)
.\start.ps1 -Port 8080        # run the API somewhere else
```

Shutting down:

```powershell
.\stop.ps1                 # stops API + worker + Ollama, leaves containers up
.\stop.ps1 -Containers     # also stops Postgres/Redis
.\stop.ps1 -KeepOllama     # leave Ollama running (model stays warm in RAM)
```

`stop.ps1` kills the whole process tree (`taskkill /T`), which matters
because `uvicorn --reload` runs a reloader parent plus a child that actually
holds the port -- killing only the window would orphan the child and leave
port 8000 occupied. It reads the PIDs `start.ps1` recorded in `.run\pids.json`
and falls back to a command-line scan when that file is missing or stale.

Containers are left running by default: they're cheap, survive reboots
healthily, and starting them is the slowest part of `start.ps1`.

### Manual (any OS)

```bash
# Terminal 1: API
uvicorn app.main:app --reload

# Terminal 2: worker (all stages; see worker_entrypoint.py to split by stage)
PYTHONPATH=. python worker_entrypoint.py

# Terminal 3: a dev user + bearer token (no signup/login flow exists yet)
PYTHONPATH=. python scripts/create_dev_user.py you@example.com
```

On Windows, `PYTHONPATH=.` is `$env:PYTHONPATH="."` in PowerShell, and the
venv activates with `.\venv\Scripts\Activate.ps1` (the bare
`venv\Scripts\activate` is the cmd.exe form and fails in PowerShell with
"The module 'venv' could not be loaded"). `PYTHONPATH` matters because
`scripts\` has no `__init__.py`: running a script by path puts `scripts\` on
`sys.path`, not the repo root, so `import app` fails without it.

`worker_entrypoint.py` auto-detects Windows and switches RQ's `Worker` (which
forks a subprocess per job -- `os.fork()`, which doesn't exist on Windows)
for `SimpleWorker` (runs jobs in-process instead). Nothing to configure;
just know that on Windows a job that hangs forever can't be killed by RQ's
timeout the way it can on Linux/macOS -- fine for local dev.

Then:

```bash
TOKEN="<paste the token from create_dev_user.py>"

curl -X POST http://localhost:8000/api/v1/stream-jobs \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/a/video.mp4"

curl http://localhost:8000/api/v1/stream-jobs/<id> \
  -H "Authorization: Bearer $TOKEN"
```

API docs (Swagger UI) are at `http://localhost:8000/docs` once the server
is running.

## Speech-to-text (transcription worker)

`STT_PROVIDER` in `.env` picks the provider -- `app/workers/transcription.py`
only ever talks to the `SttProvider` interface in `app/core/stt/`, so
switching is a config change, not a code change:

- `local` (default): [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
  open-source Whisper weights, runs on this machine's CPU. No API key, no
  per-minute cost, audio never leaves the machine. The **first** job you
  transcribe will download the model weights from Hugging Face Hub (a few
  hundred MB for the default `base` model) -- that download needs real
  internet access; if it fails, the job ends up `failed_transcription` with
  the download error as `last_error`, same as any other permanent failure.
  After that first download the model is cached locally and every
  subsequent job is fast. `STT_LOCAL_MODEL_SIZE` (`tiny`/`base`/`small`/
  `medium`/`large-v3`) trades accuracy for CPU time; `base` is the default
  because it's a reasonable CPU-only balance -- bump it up if you have time
  to spare and want better accuracy, or wire in `device="cuda"` in
  `app/core/stt/local_whisper.py` if this ever runs on a GPU box.
- `openai`: the hosted Whisper API instead -- set `OPENAI_API_KEY`. No local
  compute cost, but a per-minute API cost and a network dependency. Handles
  the API's ~25MB request-size limit by chunking long audio near silence
  boundaries first (`app/core/stt/chunking.py`).

## Segmentation (candidate clip windows)

`app/workers/segmentation.py` turns a transcript into `candidate_segments`
rows -- windows of the video that might become clips, before any
scoring/ranking happens. Two signals feed it, per the architecture doc's
ranking guidance:

- **Scene cuts**: PySceneDetect's content-aware detector on the raw video
  (`app/core/segmentation_logic.py::detect_scene_cuts`). This is typically
  the slowest part of the whole pipeline for a long VOD -- PySceneDetect
  already auto-downscales each analyzed frame (~256-384px effective width,
  so the per-frame pixel-diff math is cheap), but by default it still
  **decodes every single source frame** to get there. `SCENE_DETECT_FRAME_SKIP`
  (default `2`, i.e. analyze every 3rd frame) cuts that decode cost roughly
  proportionally -- PySceneDetect's decode loop calls a cheap `grab()`
  (advance past the frame) instead of a full decode for every frame it
  skips, so this is a real cost reduction, not just discarded work. The
  trade-off is a detected cut can land up to `frame_skip` frames after its
  true boundary, which doesn't matter here since these timestamps are only
  ever *snapped to* by the window-boundary logic below and contribute one
  modest-weight scoring signal, never anything needing frame accuracy.
  Lower toward `0` only if that slop becomes visibly wrong for your content.
- **Speech structure**: transcript segments are merged into continuous
  "speech blocks" wherever the gap between them is short
  (`SEGMENT_SILENCE_GAP_SECONDS`, default 1.2s); a longer gap is treated as
  a natural break between blocks.

A speech block within `[SEGMENT_MIN_CLIP_SECONDS, SEGMENT_MAX_CLIP_SECONDS]`
(default 15-90s, overridable per job -- see "Per-job options" below) becomes
one candidate window as-is. A block longer than the max produces **several
overlapping** max_len-ish candidate windows via a sliding pass across it
(`_sliding_subwindows` in `segmentation_logic.py`), each snapping its
boundaries to a nearby scene cut when one exists rather than hard-cutting at
an arbitrary point -- verified against a real video with real color-change
scene cuts, not just unit-tested logic (see `tests/test_segmentation_worker.py`).

This sliding approach replaced an earlier version that split a long block
into one rigid, fixed-length partition starting from the block's own start --
which meant whatever moment happened to fall at a multiple of the max clip
length became a cut boundary, whether or not that was actually the best
place to cut. Sliding a window across the block and over-generating
candidates lets `scoring.py`'s ranking (next section) pick the best-scoring
sub-part of a long ramble, instead of being stuck with wherever a blind
partition happened to land. A safety cap (`_MAX_SUBWINDOWS_PER_BLOCK`, 60)
keeps one pathologically long unbroken block (e.g. a 30+ minute monologue
with no pauses) from generating an unbounded number of rows.

A block shorter than the min is dropped rather than padded or merged across
its neighboring silence -- a deliberate v1 simplification; if real VODs turn
out heavily fragmented (lots of short blocks getting dropped), that's
documented as the first thing to revisit in `segmentation_logic.py`'s
docstring.

If your test videos are short (under the 15s default minimum), lower
`SEGMENT_MIN_CLIP_SECONDS`/`SEGMENT_MAX_CLIP_SECONDS` in `.env` (or per job,
see below) for local testing -- otherwise every job will legitimately end in
`failed_segmentation` with "no candidate windows found," which is correct
behavior for a video with no speech block long enough to be a real clip,
not a bug.

## LLM segment suggestion

`ENABLE_LLM_SEGMENT_SUGGESTIONS` (default `false`) adds a second, opt-in
candidate proposer alongside the deterministic sliding-window segmentation
above. When on, `app/workers/segmentation.py` calls
`app/core/llm_segmentation.py` once per job with the full transcript and
asks it to suggest a few "this looks like a good clip" moments, each with a
concrete `topic` (what is actually said) and a one-sentence reason.

**The prompt asks for substance, not just energy (2026-09-05).** The
original version listed only affect-based cues ("big reactions, jokes,
dramatic reveals, hot takes, arguments, surprising or high-energy
moments") and never once asked *what was said* -- which reliably surfaced
the loudest moments rather than the most interesting ones. It now puts
content-bearing moments on equal footing: a specific claim/fact/number, a
strong opinion **with** its reasoning, a story with a real payoff, an
explanation of something non-obvious, a genuine disagreement or change of
mind. Reactions and jokes still qualify, but only "when the content behind
it lands, not merely because the reaction is loud." Three further
requirements the old prompt never stated at all: the moment must be
self-contained (understandable without the rest of the VOD), it must start
at the **setup** rather than the punchline (opening mid-thought is the
classic failure mode of automatic clipping), and filler/small
talk/stream-logistics/chat-reading is explicitly excluded however energetic
it sounds. The required `topic` field is the enforcement mechanism: a model
that can't state the point concretely usually picked a moment that only
*sounded* interesting. `topic` and `reason` are stored combined in the
existing `llm_reason` column, so this needed no schema change; a response
omitting `topic` still parses (it's a prompt-level lever, not a hard
contract).

This is deliberately **additive, not a replacement**: the LLM never picks
which clips get made. Its suggestions are inserted into the exact same
`pending_score` `candidate_segments` pool as the heuristic pass's windows
(`origin="llm"` vs. `origin="heuristic"`, with `llm_reason` set on the
former), and the existing deterministic scoring + non-max-suppression
selection in `app/workers/scoring.py` ranks and picks winners from the
combined set exactly as it already does today -- unaware of and unaffected
by which pool a candidate came from. This is the same "opt-in, additive,
degrades cleanly" pattern as audio-event scoring: it only ever *adds* more
candidates for the same unmodified decision process to consider.

**Timestamp-hallucination mitigation.** An LLM asked to emit raw
`start`/`end` seconds for a long transcript will confidently invent numbers
that don't correspond to anything real, especially past the first few
minutes. Instead, the prompt numbers every transcript segment with an index
and asks the LLM to reference `start_index`/`end_index` only
(`app/core/llm_segmentation_logic.py::build_segment_suggestion_prompt`);
the real start/end seconds are then read straight from
`transcript_segments[index]` on our side, never trusted from a number the
model typed. A response naming an out-of-range or backwards index just has
that one suggestion dropped -- never the whole batch.

**Provider config is shared, not duplicated.** This reuses
`CAPTION_LLM_PROVIDER`/`CAPTION_LLM_MODEL`/`CAPTION_OLLAMA_MODEL`/
`CAPTION_OLLAMA_BASE_URL` (see "Clip hashtags & captions" below) rather than
a second parallel provider config -- both features are "call the same
OpenAI-shaped chat completions endpoint," just for a different task.

**Failure posture.** `generate_llm_segment_suggestions` never raises --
disabled, no API key, an unreachable provider, a malformed response, or a
transcript longer than `LLM_SEGMENT_MAX_TRANSCRIPT_SEGMENTS` (default 800;
a safety valve against blowing past context limits on a very long VOD,
rather than silently truncating the transcript and biasing suggestions
toward one end of it) all degrade to zero suggestions. The worker also
wraps the call in its own `try`/`except` as defense in depth, so a future
change to that module breaking the "never raises" contract still can't sink
a segmentation job. One real behavior change from this: if the
deterministic pass finds **zero** windows (previously an automatic
`failed_segmentation`) but the LLM pass finds at least one, the job now
succeeds on the LLM suggestion(s) alone -- only reachable when the flag is
on.

`LLM_SEGMENT_MAX_SUGGESTIONS` (default 8) caps how many suggested candidates
one job can add, so a verbose response can't flood the scoring pool. The dev
console shows a "🤖 LLM-suggested clip" badge (hover for the reason) on any
clip whose candidate window came from this path.

### Learning from your own review comments (few-shot, no training run)

`LLM_SEGMENT_FEEDBACK_EXAMPLES` (default 6, 0 disables) feeds this
creator's own recent review decisions back into the segment-suggestion
prompt as worked examples of what they personally keep and cut. The
segmentation worker gathers them (`_gather_feedback_examples`) and
`build_feedback_examples_block` renders them as a few-shot block ahead of
the general guidance, with an explicit instruction that the creator's taste
overrides that guidance wherever the two disagree.

Only reviews carrying an actual **written comment** count -- an
approve/reject with no reasoning says nothing about *why*, and the
deterministic scorer already covers "was it picked." Rejections are
included alongside approvals on purpose: knowing what you throw away is at
least as informative as knowing what you keep, and negatives are the one
thing a "here are good clips" example set can never teach. The transcript
excerpt shown with each example comes from the clip's own candidate window
against its job's transcript, **not** from the stored caption -- a caption
may have been LLM-rewritten or hand-edited, so it isn't a faithful record
of what the reviewer was actually judging.

**Why this instead of fine-tuning a local model.** Fine-tuning for clip
*selection* is a much worse fit than the caption fine-tuning path described
in "Training on your own approved clips" below: Ollama itself doesn't
fine-tune (you'd need unsloth/axolotl → LoRA → GGUF → `ollama create`, and
a GPU), selection is a ranking task that wants hundreds of examples plus
real negatives rather than the ~50-100 that helps a generation task, and
none of that infrastructure pays off until the data exists. Few-shot
prompting works at ten examples, needs no training run at all, costs
nothing extra per job beyond a slightly longer prompt, and improves the
moment you write another comment. The default of 6 is a middle ground:
enough for a taste signal, few enough that the examples don't crowd the
actual transcript out of the model's context.

Write comments in the dev console's per-clip comment box (saved alongside
whichever Approve/Reject button you press). They're stored in
`review_decisions.notes`, which already existed -- the box is simply the UI
that was missing.

### Stitched multi-part clips

A moment worth clipping isn't always one contiguous stretch: a story can be
told in two passes, or a payoff can call back to something said ten minutes
earlier. `candidate_segments.parts` (nullable JSONB, migration
`e5f6a7b8cadb`) optionally holds a list of `[start, end]` pairs that get
cut out and joined end-to-end at render time. Null -- every heuristic
candidate, and everything before this existed -- means a plain contiguous
window and the entire multi-part path is skipped.

Only the LLM proposer can produce these, and the prompt frames stitching as
the exception ("Most moments should NOT use this"), capped at
`MAX_PARTS = 3` -- more than that is a montage, a different feature that's
much easier to get incoherently wrong. Parts are validated the same
index-only way as everything else, and a malformed `parts` costs the
suggestion its stitching, never the suggestion itself (it falls back to the
plain single-range interpretation).

Three things this touches that are easy to get subtly wrong, all handled:

- **Scoring** (`score_multipart_window`): features are computed over the
  **union** of the parts, never the span they cover. A span-based
  `speech_density` would divide real words by a duration including material
  the viewer never sees; `pause_count`/`scene_changes` would count beats
  happening entirely inside the removed gaps. Counts are summed *within*
  parts and deliberately not across a join -- the cut between parts is an
  edit, not a dramatic pause the creator timed, and crediting it would
  reward stitching for its own sake. `hook_strength` comes from the first
  part alone, since that's the opening a viewer actually sees.
- **Rendering**: `build_concat_prefix` trims each part and `concat`s them
  into `[vsrc]`, then `retarget_source_label` points the layout graph at
  that stream instead of `[0:v]` -- so all three layouts (single crop,
  split reaction, fit frame) work with stitched clips without knowing
  multi-part exists. `setpts=PTS-STARTPTS` per part is what stops `concat`
  producing gaps matching the removed material. No outer `-ss`/`-t`: those
  would shift the timestamps the trims are expressed in and silently cut
  the wrong footage. The audio branch is built only when the source
  actually has an audio stream (`_probe_has_audio`) -- a filtergraph
  referencing a nonexistent `[0:a]` fails the whole command, where the
  single-window path gets away with ffmpeg's optional `-map 0:a?`.
- **Captions** (`build_multipart_srt`): each part's transcript slice is
  remapped onto the stitched timeline, so a caption from the second part
  plays at the summed duration of the parts before it rather than at its
  original source timestamp.

`start_seconds`/`end_seconds` stay populated for a stitched candidate
(first part's start, last part's end) so every existing query, ordering,
overlap check and non-max-suppression comparison keeps working untouched --
they describe the *span* the clip is drawn from, while `parts` describes
what actually plays. Anything that cares about real playing time (scoring,
render duration, the stored `duration_seconds`, the burned-in title's
length, thumbnail sampling) goes through `normalize_parts`/`parts_duration`
instead. The dev console shows a "✂ Stitched from N parts" badge (hover for
the source ranges) -- worth reviewing critically, since the hard cuts
between parts are exactly where a stitched clip goes wrong.

**Not yet run against a real VOD.** Implemented and tested (including a
real ffmpeg stitched render), but no real content has been through it --
same status every recent feature ships with.

**Not implemented, documented here on purpose (post-MVP idea):** a
video-native model (e.g. Gemini's Files API) that watches the actual video
instead of reading only the transcript, so it could also catch
silent/visual-only highlights a transcript can't see (a wordless reaction, a
pure-gameplay moment). That's a real, heavier upgrade path -- a new SDK, a
new API key, materially higher cost/latency on a full VOD, and its own
timestamp-snapping problem to solve (video-LLM timestamps drift too, so it
would need the same "snap to a real boundary, never trust the raw number"
treatment this transcript-only version already applies to segment indices)
-- worth revisiting once this simpler version has proven the pattern is
useful at all, not before.

## Scoring (ranking candidate clips)

`app/workers/scoring.py` scores every `pending_score` candidate segment and
selects up to `stream_job.max_clips` (default 10, overridable per job -- see
"Per-job options" below), optionally filtered by a minimum score. Per the
project's AI/ML principles, this is **deterministic feature scoring, not an
LLM or a virality model** -- the feature functions live in
`app/core/scoring_logic.py` (pure, unit-tested separately, no DB/IO) so
ranking is inspectable and tunable without touching worker/DB code:

- **Speech density** -- words/sec inside the window (energetic talking).
- **Pause count** -- number of >=0.35s gaps between transcript segments
  inside the window (comedic timing / dramatic beats).
- **Question marks** -- rough proxy for question/payoff structure.
- **Emotional language** -- hits against a small hand-picked intensity
  lexicon (`EMOTIONAL_LEXICON`) -- deliberately *not* a profanity filter;
  profanity/banned-content detection is a safety concern, not a ranking
  signal, and belongs in a separate check before it ever gets built.
- **Scene changes** -- PySceneDetect cuts landing inside the window. Scoring
  reuses the scene cuts segmentation already detected and persisted on the
  job (`stream_jobs.scene_cuts_seconds`) rather than re-downloading the raw
  video and re-running PySceneDetect from scratch -- this used to be a real,
  measurable inefficiency (the single biggest lever in "why does a long VOD
  take so long," see "Job timeouts" below). Only falls back to re-detecting
  (old behavior) for a row from before this cache existed, or if
  segmentation somehow didn't get far enough to persist it; if that
  fallback detection also fails, scoring degrades gracefully -- logs a
  warning and scores with `scene_changes=0` for every candidate -- rather
  than failing the whole job over one visual signal.
- **Motion proxy** -- listed in the project's ranking guidance but **not
  implemented** (would need real frame/optical-flow analysis); its function
  always returns 0.0 and its config weight defaults to 0.0 so it's inert,
  not silently wrong. Flagged as a `later research` item, not MVP scope.
- **Laughter / crowd reaction** -- opt-in (`ENABLE_AUDIO_EVENT_SCORING`, off
  by default), computed by a real open-source audio classifier rather than a
  heuristic -- see "Audio-event scoring" below for what it is and why it's
  gated off by default.
- **Hook strength** -- on by default (`SCORE_WEIGHT_HOOK_STRENGTH`, no new
  dependency, no `ENABLE_*` flag) -- a proxy for how strong the window's
  opening beat is. See below for why this exists and exactly how it's
  computed.

Each feature is normalized to 0..1 against a hand-picked cap (see
`_NORMALIZATION_CAPS` in `scoring_logic.py`) and combined into one composite
score using the `SCORE_WEIGHT_*` settings in `.env`/`config.py`. That raw
composite is then rescaled to a **0..10 "score"** (`score_out_of_10` --
composite divided by the sum of the configured weights, so the number stays
comparable across jobs/weight configs instead of shifting whenever
`SCORE_WEIGHT_*` changes) -- this is what's stored on
`candidate_segments.score`, returned by the API on each clip, and shown as a
badge on every clip card in the dev console. It's still the same
deterministic heuristic underneath, just rescaled for display -- **not a
virality prediction**, and the UI says so in the badge's tooltip. Every
candidate's raw features, normalized values, and per-feature weighted
contribution are written to `candidate_segments.score_breakdown` (the raw
pre-rescale composite is in there too, as `score_breakdown["composite"]`,
for anyone debugging the weighting itself) -- a human reviewer (or a future
"why was this picked" LLM explainer, which is a post-MVP addition, not
required for scoring itself) can see exactly why a clip ranked where it did
instead of trusting one opaque number. There's no ground truth yet to fit
these weights against real outcomes -- they're a reasonable starting point,
documented as worth revisiting once real jobs produce enough
approved/rejected clips to sanity-check them.

**A real regression found and fixed (2026-08-23): every score was
deflated, even with audio-event scoring off.** `SCORE_WEIGHT_LAUGHTER`/
`SCORE_WEIGHT_CROWD_REACTION` default to 1.0 each (see "Audio-event
scoring" below), but `ENABLE_AUDIO_EVENT_SCORING` defaults to `false` --
and `app.workers.scoring._weights_from_settings()` was passing those two
weights straight through into `ScoreWeights` regardless of the flag. Since
`score_out_of_10` divides by the *sum of all configured weights*
(see below), this meant every score's denominator included 2.0 worth of
weight (out of the default 6.5 total) that literally no clip could ever
earn while the feature was off -- deflating every single score, hard
enough that even a flawless clip on every real feature capped out around
6.9/10 instead of 10/10, and an otherwise-solid clip could easily land
in the "trash" 2-4 range. Fixed by zeroing `laughter`/`crowd_reaction`
weights in `_weights_from_settings()` whenever `ENABLE_AUDIO_EVENT_SCORING`
is off, restoring the exact scoring behavior from before that feature
existed -- same principle `motion_proxy`'s weight already follows (0.0 by
default specifically because that feature isn't implemented; this was the
same idea, just for "implemented but not turned on"). Regression tests:
`test_weights_from_settings_zeroes_audio_event_weights_when_disabled`/
`..._uses_configured_weights_when_enabled` in `tests/test_scoring_worker.py`.
**If you already ran real jobs before this fix, their stored scores are
lower than they should be, and there's no automatic way to recompute
them** -- `app.workers.scoring.run()` only scores candidates still in
`pending_score` status, and every candidate in an already-finished job has
already moved to `selected`/`rejected`, so simply re-running the scoring
worker on an old `stream_job_id` won't do anything (it'll just fail with
"no pending_score candidate segments to score"). The good news: the bug
applied the *same* inflated denominator to every candidate within a given
job, so it never changed which clips ranked above which others *within
that job* -- any approve/reject decisions you already made based on
relative ranking are still sound. It only distorted the absolute number
(making "6/10" look like "trash" when it was actually a strong relative
pick) and, only if you'd explicitly set a per-job `min_score_threshold`
override, could have wrongly excluded a clip that should have cleared it.
Treat every job scored from now on as using the corrected numbers; old
jobs' displayed scores just read artificially low.

Because segmentation can now hand scoring several overlapping candidates
covering the same moment (see "Segmentation" above), selection isn't a
naive top-N-by-score slice -- it's **greedy non-max suppression**
(`select_top_non_overlapping` in `scoring_logic.py`): take the
highest-scoring candidate, skip any remaining candidate that overlaps it by
more than 50% (IoU), repeat until `max_clips` is filled or candidates run
out. This is what turns "several near-duplicate windows of the same
highlight" back into a diverse final selection. A per-job
`min_score_threshold` (0..10, default 0 = no filtering) is applied
alongside this -- a candidate below the threshold is never selected even if
fewer than `max_clips` end up chosen, which is the intended behavior of "a
minimum score to look for," not a bug.

For each selected candidate, scoring also creates a `rendered_clips` row
(`status='pending'`) with a placeholder caption (`default_caption()` --
just the window's own transcript text, truncated to a whole word) and
enqueues one rendering job per clip. LLM-assisted captions/hashtags are
explicitly a post-MVP enhancement -- the deterministic default has to work
without them.

### Hook strength (2026-08-29)

Added after a direct question about what actually drives TikTok/YouTube
Shorts distribution. Nobody outside those companies knows the real ranking
code, but TikTok's own published explanation
([newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you](https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you))
names **watch completion** as "a strong indicator of interest" (weighted
more for longer videos), alongside user interactions and video metadata
(captions/sounds/hashtags -- already covered by this pipeline's caption
generation). It also explicitly states follower count and past video
performance are **not** direct ranking inputs. Every credible third-party
source on YouTube Shorts points at the same completion/retention lever, with
nothing more specific published.

Given that, the gap in this pipeline's existing scoring was that nothing
scored *for* completion -- speech density, pause count, question marks,
emotional language, and scene changes are all real signals, but none of
them capture "does this clip actually keep you watching past the first
couple of seconds," which is the one lever both platforms keep pointing at.
`hook_strength()` in `scoring_logic.py` is a deterministic proxy for that,
**not** a virality predictor and not trained on outcomes data -- consistent
with the project's "do not assume virality prediction is solved" principle,
this is one more heuristic feature in the same weighted-sum model, not a
new kind of system.

**What it measures**, over the window's first `HOOK_WINDOW_SECONDS` (3.0s,
a guess -- long enough to catch a short opening line, short enough to
actually be about the *opening* beat rather than the first third of a
90s clip):
- **Immediacy** (70% of the score): 1.0 if transcript speech starts right at
  the window's own start (no dead air), decaying linearly to 0.0 as silence
  eats the whole hook window. A window with zero speech in that opening
  stretch scores 0.0 outright, regardless of what happens later in the
  clip -- dead air at the very start is the more unambiguous completion-rate
  killer of the two components here.
- **Attention bonus** (30% flat): added if the opening text contains a
  question mark or hits `EMOTIONAL_LEXICON` (the same lexicon
  `emotional_language_hits` already uses) -- both are classic "keep
  watching to find out" setups. This is a text-pattern match, not a
  judgment of whether the hook is actually *good* -- it can't read tone,
  delivery, or context.

Already normalized 0..1 (same as `laughter`/`crowd_reaction`), so its
`_NORMALIZATION_CAPS` entry is `1.0` (no rescaling). Unlike
`laughter`/`crowd_reaction`, there's no `ENABLE_*` flag -- it needs no new
dependency and no extra impure input beyond the transcript segments every
caller already has, so `SCORE_WEIGHT_HOOK_STRENGTH` (default `1.0`, same
starting-guess scale as `speech_density`/`question_marks`) is passed
through unconditionally in `_weights_from_settings()`. `ScoreWeights`'
own field default stays `0.0` (matching `laughter`/`crowd_reaction`'s
pattern) purely so existing "everything off" test construction
(`ScoreWeights(0, 0, 0, 0, 0, 0)`) doesn't silently pick up a nonzero
weight -- the always-on behavior lives at the config layer, not the
dataclass.

**Explicitly not attempted**: reverse-engineering the actual algorithm,
predicting a view count, or optimizing for anything beyond the one
published signal (completion) this pipeline can plausibly proxy from a
transcript alone. Tune/replace this once real upload + view-count data
exists to check it against (see "Known loose ends" -- there's no ground
truth yet, same caveat as every other `SCORE_WEIGHT_*`).

Tests: `tests/test_scoring_logic.py` (the `hook_strength` pure-function unit
tests, plus `score_window` wiring), `tests/test_scoring_worker.py`
(`test_weights_from_settings_always_passes_through_hook_strength`).

## Audio-event scoring (laughter / crowd reaction)

Opt-in via `ENABLE_AUDIO_EVENT_SCORING` (default `false`). When on,
`app/core/audio_events.py` runs every selected candidate window's audio
through [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) (via
the `panns-inference` package), an open-source classifier pretrained on
Google's AudioSet taxonomy (527 sound classes) -- a fixed model's output,
same "deterministic feature" posture as every other scoring signal, not a
generative/LLM call. It reports two composite 0..1 signals per window:

- **`laughter`** -- max probability across `Laughter`, `Baby laughter`,
  `Giggle`, `Snicker`, `Belly laugh`, `Chuckle, chortle`.
- **`crowd_reaction`** -- max probability across `Cheering`, `Applause`,
  `Crowd`, `Shout`, `Bellow`, `Whoop`, `Yell`, `Battle cry`, `Children
  shouting`.

Both feed into the same weighted composite as every other feature
(`SCORE_WEIGHT_LAUGHTER` / `SCORE_WEIGHT_CROWD_REACTION` in `.env`, default
1.0 each -- see `config.py`'s comment on why that starting value, not
emotional language's higher 1.5). Model loading (a real multi-second cost)
happens once per worker process via a lazy singleton, not once per window
or per clip.

**Why this is off by default**: it's a genuinely new class of dependency
for this project -- `torch` plus a ~300MB pretrained checkpoint, where
previously the heaviest dependency was faster-whisper's CTranslate2 runtime
(no torch/tensorflow at all). The checkpoint and AudioSet label file
download once on first use and cache in `~/panns_data/`. On a CPU-only
Windows dev machine, install the CPU-only torch wheel first --
`pip install torch --index-url https://download.pytorch.org/whl/cpu` --
before `pip install -r requirements.txt`, to avoid pulling a much larger
CUDA-enabled build you can't use anyway.

**Two real Windows bugs found and fixed in `panns_inference` itself** (not
this project's code, verified by installing and inspecting the actual
package): both its `config.py` (AudioSet label CSV) and its `inference.py`
(model checkpoint) fetch their data files via `os.system('wget ...')`.
`wget` isn't a built-in Windows binary, so on Windows that call fails
silently and the next line (opening the file it just tried to fetch)
crashes with `FileNotFoundError`. `app.core.audio_events._ensure_panns_data()`
pre-fetches both files itself via `requests` (already a project dependency,
identical on every platform) before `panns_inference` is ever imported, so
its own "file already exists" checks see them already present and skip the
broken `wget` call entirely. This is exactly the same class of bug, and the
same fix pattern, as the two Windows-only RQ `SIGALRM` issues documented in
"Job timeouts" below -- worth knowing about since it's the kind of thing
that only shows up once you actually run this on Windows, not in this
Linux dev sandbox.

Like every other optional signal in this pipeline (scene detection, LLM
captions), a failure anywhere in this path -- model load, missing audio
file, an unrecognized label in a differently-shaped label file -- degrades
to `laughter=0.0, crowd_reaction=0.0` for the affected window(s) rather
than failing the scoring job.

## Rendering (producing the actual clip)

`app/workers/rendering.py` turns one selected candidate into a real
watchable `.mp4` via ffmpeg. The pure math/string-building parts (crop
dimensions, SRT generation, the Windows path-escaping fix below) live in
`app/core/rendering_logic.py`, unit-tested separately from the worker:

- **Cut**: `-ss <start> -t <duration>` on the raw video -- accurate even
  though `-ss` comes before `-i`, because the output is re-encoded anyway
  (ffmpeg does frame-accurate seeking in that case, it's just the fast path
  instead of the slow one).
- **Reframe**: one of two layouts, both scaled to a fixed 1080x1920 output,
  chosen per clip by `app.core.face_detect` + `classify_reaction_layout`:
  - **Single crop** (the default, and the fallback whenever face detection
    finds nothing): a 9:16 crop of the full frame (`compute_vertical_crop`
    picks whichever axis needs cropping so the result is always exactly
    9:16, floored to even pixel dimensions since libx264/yuv420p reject odd
    ones). The crop is **face-aware but still static**: `face_detect`
    samples a few frames from the clip's own window and runs OpenCV's
    bundled Haar cascade face detector on each; if a face is found,
    `compute_crop_offset` centers the crop on it instead of the frame's
    geometric center (median across sampled frames, clamped to stay in
    bounds). This is the right layout for "IRL"/single-camera content,
    where the crop just needs to keep the subject in frame.
  - **Split reaction layout**: when the detected face looks like a small,
    corner-positioned webcam box rather than someone filling the frame
    (`classify_reaction_layout` -- face area ≤18% of the frame AND its
    center sits within 40% of the frame size from some corner on both
    axes; deliberately simple and deterministic, same "signal first"
    philosophy as `scoring_logic.py`, not a learned classifier) -- i.e. a
    streamer reacting to separate gameplay/video content -- the render
    instead stacks two crops with ffmpeg's `vstack` filter
    (`build_split_reaction_filtergraph`): the top half is a tight zoom on
    the facecam (`compute_face_zoom_crop`), the bottom half is a plain crop
    of the full frame. This doesn't attempt to detect and exclude the
    webcam box's exact rectangle from the bottom half -- that would need
    real rectangle detection this MVP doesn't have -- so the bottom half
    may still show a small sliver of it; a known, flagged limitation, not a
    bug. Toggle independently with `ENABLE_REACTION_SPLIT_LAYOUT` (default
    `true`; requires `ENABLE_FACE_AWARE_CROP=true` too, since it needs a
    face profile to classify).

  **Per-job override for a real false-positive**: `classify_reaction_layout`'s
  fixed size/corner thresholds are a reasonable guess, not tuned against any
  specific creator's actual webcam size/position -- in practice this means it
  can misfire in both directions on real content (splitting a VOD that never
  actually has a facecam, or missing one that does). Rather than trying to
  blindly re-tune the thresholds without real labeled data (which would just
  move the false-positive/negative rate around, not fix it), `StreamJob
  .camera_layout_mode` (`camera_layout_mode` on `POST /stream-jobs`, or the
  dev console's "Camera layout" dropdown) lets a creator who knows the
  answer for a given VOD just say so: `single_crop` never splits for this
  job's clips no matter what the classifier would have said (so a
  false-positive detection can't steal screen space from the actual
  content), `split_reaction` always splits when ANY face is found on a
  clip, skipping the classifier entirely (for a VOD the creator knows has a
  facecam that the auto thresholds keep missing). Both still respect
  `ENABLE_REACTION_SPLIT_LAYOUT` as a hard kill switch, and `split_reaction`
  still degrades to `single_crop` on a clip where no face was detected at
  all -- there's no "cam" half to zoom in on without one. Leaving this
  unset (`auto`, the default) keeps today's automatic behavior unchanged.

  - **Fit whole frame** (`camera_layout_mode=fit_frame`, 2026-09-05): no
    crop at all. The entire source frame is scaled to the output width and
    centered vertically over a blurred, zoomed copy of itself
    (`build_fit_frame_filtergraph`). Both other layouts *crop*, and
    cropping 16:9 to 9:16 discards roughly two thirds of the width -- fine
    when the subject is a person who can be centered, actively harmful for
    gameplay/screen-share content where the important thing (a scoreboard,
    a UI element, the other half of the map) is exactly what falls outside
    the crop window. This loses nothing; the trade-off is a smaller content
    band. Verified empirically rather than assumed: a 1280x720 source
    renders as a 608px-tall sharp band centered in the 1920px canvas
    (34.2%-65.8%), which is exactly `1080 * 720/1280`. It ignores
    `ENABLE_REACTION_SPLIT_LAYOUT` and face detection entirely -- there's
    nothing to classify when nothing is cropped.

  **Crop side override** (`crop_bias`, migration `d4e5f6a7b8ca`):
  `left`/`center`/`right`, for the two layouts that *do* crop. An explicit
  bias beats face detection (same precedence principle as
  `camera_layout_mode` overriding `classify_reaction_layout`: when a
  creator has said where the important part of their frame is, an automatic
  guess doesn't get to argue). `center` is stored rather than treated as a
  default, unlike `camera_layout_mode`'s `auto`, because it's a real
  instruction -- "center it and do NOT follow a face" -- not an absence of
  one. The escape hatch for "the important thing is consistently on one
  side of my layout" when `fit_frame`'s smaller picture isn't wanted.

  All layouts are still **one static choice per clip**, not
  motion-tracked/smart reframing that follows movement within the clip --
  that's real added complexity (the crop/split would need to visibly
  update) this MVP doesn't need yet, and stays explicitly out of scope per
  the architecture doc's trade-off table. Any failure in face detection (no
  face found, cv2 error, a bad frame) degrades to the original centered
  single crop -- this can never make a render worse than before the
  feature existed. Turn `ENABLE_FACE_AWARE_CROP` off entirely for content
  that never has a visible face (pure gameplay capture) to skip the extra
  per-clip frame extracts + detection cost.
- **ffmpeg internals**: both layouts are built as `-filter_complex` graphs
  (not the simpler `-vf`), since the split layout needs a branching graph
  (two crops off the same source frame, merged with `vstack`) that `-vf`'s
  linear-chain-only syntax can't express -- the single-crop layout's graph
  is just as valid under `-filter_complex`, so `_render()` is one code path
  for both rather than two. `-filter_complex` outputs aren't auto-selected
  like a plain `-vf`'s are, so the video stream is explicitly `-map`'d
  (`-map "[vout]" -map "0:a?"`, with audio marked optional so a source with
  no audio track doesn't fail outright).
- **Captions**: an SRT file built from the transcript slice covering the
  window (`build_srt` -- one entry per transcript segment, timestamps
  shifted so the clip's own start is 0), burned in via ffmpeg's `subtitles`
  filter (libass). The style (`_CAPTION_STYLE` in `rendering.py`) was tuned
  by rendering a real frame and eyeballing it, not guessed from the ASS
  spec -- libass's actual font sizing on a given output resolution doesn't
  map cleanly to the raw `FontSize` number. Compact, sitting low in the
  bottom third (`FontSize=10`, `MarginV=28`) reads better at arm's length
  than the original oversized/mid-frame default did. (`MarginV` went
  `60` → `40` → `28` across two rounds of user feedback wanting it lower;
  each move re-verified the same way, by rendering a real 1080x1920 frame
  and measuring the actual white-pixel rows, not just eyeballed. At `28`
  the caption band sits roughly 10% up from the bottom edge -- closer to
  TikTok's own bottom UI band [estimated at "typically 15-20%" in this
  codebase, never verified against the live app from this dev environment]
  than any previous value. If a real render shows the caption fighting the
  platform UI, move `MarginV` back up rather than lower.)
- **Clickbait title banner**: a bold, bigger hook line burned into the top
  of the frame for the clip's whole duration -- a second `subtitles`
  burn-in stage, chained after the captions above
  (`build_title_srt` + `_TITLE_STYLE`). The text comes from the same
  LLM/heuristic annotation call described in "Clip hashtags & captions"
  below, called **before** the filtergraph is built (not after rendering,
  like hashtags/caption/explanation) since it has to be known in time to
  burn in. Toggle independently with `ENABLE_CLIP_TITLE_OVERLAY` (default
  `true`) -- off still generates/stores the title text, it just doesn't get
  baked into the pixels. `_TITLE_STYLE`'s `FontSize` went `16` → `13` on
  user feedback that the banner took up too much of the top of the frame --
  re-verified the same "render a real frame, measure the white-pixel rows"
  way, and it's a bigger win than "10% smaller text" implies: a typical
  ~30-40 char title that wrapped to 2 lines at `16` (~15% of frame height)
  fits on ONE line at `13` (~7%) -- roughly half the footprint, not just a
  smaller font. Even a full `MAX_TITLE_CHARS=70` title, which still wraps
  to 2 lines at `13`, comes out smaller than the old default's typical
  (non-worst-case) footprint.
  - **A real, non-obvious gotcha worth calling out**: `force_style`'s
    `Alignment` field, for this exact ffmpeg/libass SRT-burn-in path, turned
    out to follow the *legacy SSA v4* alignment numbering (6 = top-center),
    not the ASS v4+ numpad-style numbering (`_CAPTION_STYLE`'s own bottom
    alignment happens to be `2` under both schemes, which is why this never
    came up before). `Alignment=8` (the "correct" ASS numpad top-center)
    rendered the banner across the vertical *middle* of the frame instead --
    caught by rendering a real frame and measuring actual pixel positions,
    not by reading the spec. See `_TITLE_STYLE`'s comment in `rendering.py`
    for the full numbering table.
- Both burn-in stages share the same libass build, so if one fails (most
  likely a different ffmpeg build without libass compiled in -- a real risk
  since this project runs on the user's own Windows machine and ffmpeg
  builds vary), the other almost certainly would too: rendering **retries
  once without either stage** rather than losing the whole clip, and flags
  the clip (`captions_failed` and/or `title_failed`, whichever was actually
  dropped) so it's visible to a reviewer instead of silently missing text.
- **Windows path gotcha, handled proactively**: ffmpeg's `subtitles` filter
  syntax treats `:` as a key=value separator, which collides with Windows
  drive-letter paths (`C:\Users\...`) -- a well-known ffmpeg issue, not
  specific to this project. `ffmpeg_subtitles_filter_path()` normalizes to
  forward slashes and escapes the colon before it's ever a live bug report.
- **ffmpeg internals**: both layouts, and however many burn-in stages get
  chained onto them, are built as `-filter_complex` graphs (not the simpler
  `-vf`), since the split layout needs a branching graph (two crops off the
  same source frame, merged with `vstack`) that `-vf`'s linear-chain-only
  syntax can't express -- the single-crop layout's graph is just as valid
  under `-filter_complex`, so `_render()` is one code path for all of it
  rather than several. `-filter_complex` outputs aren't auto-selected like
  a plain `-vf`'s are, so the video stream is explicitly `-map`'d
  (`-map "[vout]" -map "0:a?"`, with audio marked optional so a source with
  no audio track doesn't fail outright).

- **Thumbnail selection (2026-08-30)**: originally just `-ss 0.1 -frames:v 1`
  on the finished render -- whatever frame happened to sit a tenth of a
  second in, which could just as easily be a fade transition, a mid-blink
  freeze frame, or a blank loading screen as anything worth showing a
  reviewer. `app.core.thumbnail_selection.pick_best_frame` now samples 6
  frames spread across the rendered clip (`_THUMBNAIL_SAMPLE_FRACTIONS` in
  `rendering.py`, deliberately avoiding the very start/end where a
  transition is most likely to land) and ranks them by: sharpness (a
  Laplacian-variance edge-detection score -- penalizes motion blur/fast
  pans), whether a face is visible (reusing `app.core.face_detect`'s
  already-loaded Haar cascade, no new dependency), and excludes
  near-black/near-white "dead" frames (almost always a transition) whenever
  a non-dead candidate exists. Sharpness is ranked *within one clip's own
  candidate set* (min-max normalized), not against a hand-picked global
  cap like `scoring_logic.py`'s `_NORMALIZATION_CAPS` -- there's nothing to
  keep comparable across clips here, only "which of these 6 frames from
  this clip looks best," so there was no number to guess. Face presence and
  sharpness are a guessed weight split (0.5/0.5), same "no click-through
  data to calibrate against yet" caveat as every `SCORE_WEIGHT_*`. Falls
  back to the original fixed-frame extract if every sampled candidate is
  unreadable or scoring itself fails -- never a *worse* thumbnail than
  before this feature existed, just sometimes not a better one either.
  **Not yet done**: hasn't been checked against a real VOD's thumbnails yet
  to see whether the picks actually look better in practice, and the
  0.5/0.5 weight split is a guess like everything else in scoring --
  revisit once there's a reason to (e.g. a reviewer regularly overriding
  the auto-picked thumbnail, if/when that becomes possible; there's
  currently no UI to manually pick a different frame, only to accept
  whatever this function chose).

Output is one `rendered_clips` row per clip with `object_key`/
`thumbnail_key` set and `status='rendered'` -- the dev console can now
actually play it (see below). The clip's title/hashtags/caption/explanation
annotation is generated as part of this same worker run (see next section)
and already sitting on the row by the time `status='rendered'` -- there is
no longer a separate after-render step for it. Two things are explicitly
**not** built here, same as before: motion-*tracked* (frame-by-frame)
reframing (later research, needs a real tracking signal, distinct from the
static face-aware crop above) and the "all clips terminal ->
job.status=ready_for_review" aggregate rollup (small, but deliberately
deferred so it doesn't get bolted onto this pass half-thought-out -- see
"Next steps").

### Marking the facecam by hand (beats detection)

Face detection had to answer two questions to frame a clip: *is there a
facecam* and *where is it*. It was wrong often enough to be the main source
of bad framing -- it split clips on VODs with no facecam at all, and missed
real ones.

The dev console's job detail now has **Mark the facecam**: load a frame from
the VOD, drag a box around the facecam, save. That box
(`stream_jobs.facecam_rect`, migration `f6a7b8cadbec`) overrides detection
completely -- no Haar cascade, no `classify_reaction_layout` guess, and the
split layout is forced on, because you have just said there is a facecam and
exactly where it is.

- Stored **normalized** (`{"x","y","w","h"}` as 0..1 fractions of the source
  frame), so it stays correct at any source resolution and the browser never
  has to know the real frame size.
- The marked box is a **minimum**: the crop window grows to the output
  panel's aspect ratio around it, never cutting into what you drew.
- Applies to the **next** render. Mark it right after upload -- transcription
  and segmentation take minutes, so the first render already uses it.
- An explicit `camera_layout_mode=single_crop` still wins (a later, more
  specific instruction), and a malformed rect silently falls back to
  detection rather than failing the render.
- `PUT /api/v1/stream-jobs/{id}/facecam-rect` with `{"rect": null}` clears it.

`GET /api/v1/stream-jobs/{id}/frame?at_seconds=N` serves the frame. It seeks
with `-ss` before `-i` (keyframe-accurate, effectively instant hours into a
VOD) since this is a visual reference for drawing a box, not a frame-accurate
extract.

### Auto-detection was too eager to split (fixed 2026-09-11)

`classify_reaction_layout` decides whether a detected face is a small corner
webcam box. Its two thresholds were far too loose:

| | before | after |
| --- | --- | --- |
| max face area | 18% of frame (a **611x611px** box on 1920x1080) | 6% (~353x353px) |
| "corner" margin | outer 40% per axis -- **96%** of all face positions qualified | outer 25% -- 75% |

Measured against realistic framings, the old thresholds misclassified 4 of 8
scenarios: every rule-of-thirds IRL composition (a person at x=33%, y=35% is
normal framing, not a webcam overlay) came out as "reaction layout" and got
split. The new values get 8 of 8 right and still detect real webcam boxes.

### TikTok safe zones: captions and titles were rendering under the UI

TikTok draws its own chrome over every video in the feed. Researched figures
for a 1080x1920 frame (community-measured against the live app, not published
by TikTok, so treat them as accurate to a few percent):

| region | px | what covers it |
| --- | --- | --- |
| top | ~200 (10.4%) | search icon, LIVE badge, Following/For You tabs |
| bottom | ~334 (17.4%) | @username, caption text, audio marquee (ads reserve ~450) |
| right | ~140 (13.0%) | avatar, like, comment, bookmark, share rail |
| left | ~86 (8.0%) | bezel / rounded-corner buffer |

Measuring the real libass output against those numbers found both burn-ins
sitting inside TikTok's own UI:

- **Caption** `MarginV` 28 put the band 188px above the frame bottom -- the
  entire band underneath TikTok's caption text. Two earlier "move it lower"
  adjustments (60 -> 40 -> 28) had walked it in there. Now **55** (368px
  clearance, 34px of margin).
- **Title** `MarginV` 10 started the banner 80px from the top, so its top
  120px sat behind the platform's navigation -- losing exactly the hook the
  title exists to deliver. Now **35** (starts at 247px).

The dev console's clip preview has a **Show TikTok safe zones** toggle that
shades these regions over the video, because the preview shows the raw clip
and that is precisely why the collision went unnoticed for three iterations.

These constants live in `app/core/rendering_logic.py` as `TIKTOK_SAFE_*`;
`web/app.js` mirrors them for the overlay.

## Clip hashtags & captions

`app.core.caption_generation.generate_caption_annotation` is called
**synchronously from within `rendering.run`**, before the ffmpeg
filtergraph is built -- moved there specifically so the clickbait title
(previous section) can be burned into the clip itself; hashtags/caption/
explanation just came along for the ride since they're generated by the
same call. This used to be a separate after-render async worker/queue
(`app/workers/caption_generation.py`) precisely so a slow/failed LLM call
could never block or fail the render -- once the title had to exist
*before* rendering, that ordering stopped being possible. That worker
still exists as a **manual regenerate path** (e.g. redo a clip's
title/hashtags/caption without re-rendering the video) but is no longer
auto-invoked by anything; not yet exposed through an API endpoint.

- **What it produces**: for each clip, a short punchy **title** (the
  burned-in on-screen hook, see "Rendering" above), 3-6 **hashtags**, a
  short punchy **caption** for the post itself (overwrites `rendered_clips
  .caption_text`, which starts out as `default_caption`'s plain transcript
  excerpt from scoring), and a one/two-sentence **explanation** for the
  reviewer of what happens in the clip and why it's recommended -- all
  grounded in that clip's own transcript text plus the ranking signals that
  got it selected (`candidate_segments.score_breakdown`), not a learned
  "creator voice" (no such profile exists in this schema).
- **LLM path**: one chat completion per clip via `app.core.caption_generation`,
  prompted with the clip's transcript + a ranking hint, asked to reply with
  strict JSON (`app.core.caption_logic.parse_caption_response`
  validates/clamps it). `CAPTION_LLM_PROVIDER` picks the backend:
  - `openai` (default): hosted, `CAPTION_LLM_MODEL` (default `gpt-4o-mini`),
    needs `OPENAI_API_KEY`, costs per call.
  - `ollama`: free, self-hosted open-weight model via Ollama's
    OpenAI-compatible endpoint (`CAPTION_OLLAMA_MODEL`, default
    `llama3.1:8b`; `CAPTION_OLLAMA_BASE_URL`, default
    `http://localhost:11434/v1`) -- same client, same prompt, same JSON
    parsing as the `openai` path, just pointed at a different endpoint with
    no real key required. Run `ollama pull llama3.1:8b && ollama serve`
    first. Same "local, no API key, no per-call cost" tradeoff as
    `STT_PROVIDER=local` above, applied to captions instead of
    transcription -- expect noticeably lower output quality than
    `gpt-4o-mini` from a 7-8B model, and budget ~8GB RAM/VRAM for it.
- **Non-LLM fallback, always available**: per the project's AI/ML
  principle ("if proposing ML scoring, always include a non-LLM
  fallback"), any of the following produces a **heuristic** annotation
  instead, built entirely from data already on hand (no network call,
  and *not* grounded in the transcript -- it's templates keyed off which
  scoring signal won, see below): `ENABLE_LLM_CAPTIONS=false`, no API key
  for the `openai` provider, or the call/response failing for any reason
  (including Ollama not running for the `ollama` provider). The heuristic
  title/explanation and a small set of hashtags are derived from
  `score_breakdown`'s top contributing signal(s)
  (`app.core.caption_logic.heuristic_title` / `heuristic_explanation` /
  `heuristic_hashtags`); the caption stays whatever `default_caption`
  already produced. A clip is **never** left without a title/hashtags/
  caption just because the LLM path is unavailable -- which matters more
  now than it used to, since the title is load-bearing for the render
  itself, not just optional metadata. If titles look generic/templated
  rather than specific to what's said in the clip, this fallback is what
  fired -- check `candidate_segments.llm_annotation.reason` (`"disabled"`,
  `"no_api_key"`, or `"llm_error: ..."`) for why.
- **Storage, no migration needed**: this reused two columns the schema
  already had sitting unused -- `candidate_segments.llm_annotation`
  (JSONB: the full `{source, title, hashtags, caption, explanation, model,
  generated_at}` dict) and `rendered_clips.caption_text` (already existed
  for `default_caption`). `RenderedClip.caption_title` / `.caption_hashtags`
  / `.caption_explanation` / `.caption_source` are `@property` proxies onto
  `candidate_segment.llm_annotation`, same pattern as the existing
  `.score`/`.score_breakdown` proxies -- see `app/db/models.py`. All four
  (plus `caption_text`) are on `RenderedClipOut`; since generation now
  happens synchronously during rendering, they're populated by the time
  `GET /stream-jobs/{id}/clips` first shows a clip as `status='rendered'` --
  no more polling gap. The dev console shows the title prominently at the
  top of each clip card, the hashtag pills, a caption-source badge (`✨ AI`
  / `heuristic` / `✏️ edited`), and the explanation.
- **Manual editing before upload**: `PATCH /api/v1/clips/{id}/caption`
  (`app/api/routers/clips.py`) lets a reviewer overwrite the generated
  `title`, `hashtags`, and/or `caption` -- any of the three, or all, in one
  call (at least one required). Hashtags go through the same
  `normalize_hashtags` used to parse the LLM's own response (strips
  whitespace/leading `#`, drops blanks/duplicates, caps the count), so a
  manual edit can't produce a malformed or oversized hashtag list. All
  three are stored the same place the generated version was
  (`candidate_segments.llm_annotation` / `rendered_clips.caption_text` --
  no new columns), with `source` flipped to `"manual_edit"` so the dev
  console's `✨ AI`/`heuristic` badge reflects the change (shown as
  `✏️ edited`) instead of still claiming a source that's no longer
  accurate. The dev console's title and hashtag pills each have their own
  `✏️ edit` button that opens an inline text field and saves via this
  endpoint. **Important caveat for `title`**: editing it here only updates
  the stored record -- it does **not** re-render the video, so it never
  changes an already-burned-in on-screen banner. The dev console says so
  in the save confirmation; a title fix that needs to show up in the
  actual pixels requires re-rendering the clip (not yet a one-click action
  -- see "Next steps").
- **Not built here (by design)**: this only *labels/describes* clips
  scoring already picked -- per the AI/ML principles, an LLM never chooses
  or reorders which clips get made. The manual-edit endpoint above covers
  title/hashtags/caption text; there's no equivalent UI yet to edit the
  *explanation* text (lower priority -- it's reviewer-facing context, not
  something that ships with the clip).

## Per-job options

Every one of these can be overridden per upload (form fields on
`POST /api/v1/stream-jobs`, or the "Options" panel in the dev console) --
omit any of them to fall back to the matching global `.env` default, same
as before these existed:

| Field | Default | Bound | What it controls |
|---|---|---|---|
| `stt_model_size` | `STT_LOCAL_MODEL_SIZE` (`base`) | one of `tiny`/`base`/`small`/`medium`/`large-v3` | Transcription worker's faster-whisper model for *this job* -- see "Speech-to-text" above. `tiny` cuts transcription time substantially on a slow/CPU-only machine at some accuracy cost; useful for a long VOD where `base`+ would otherwise dominate total processing time. |
| `max_clips` | `DEFAULT_MAX_CLIPS_PER_JOB` (10) | 1..`HARD_MAX_CLIPS_PER_JOB` (20) | How many clips scoring selects at most. |
| `min_score_threshold` | `DEFAULT_MIN_SCORE_THRESHOLD` (0 = no filtering) | 0..10 | A candidate below this score (see "Scoring" below) is never selected, even if fewer than `max_clips` end up chosen. |
| `min_clip_seconds` / `max_clip_seconds` | `SEGMENT_MIN_CLIP_SECONDS` / `SEGMENT_MAX_CLIP_SECONDS` | `MIN_CLIP_SECONDS_FLOOR` (5) .. `MAX_CLIP_SECONDS_CEILING` (180) | Segmentation's window length bounds for this job. |

Validation happens in `app.api.routers.stream_jobs._validate_job_overrides`
before the job is even created -- an out-of-bounds value 400s immediately
rather than silently clamping or producing a pathological job later in the
pipeline. Every field is stored on the `stream_jobs` row itself (see
migration `e093020f1c7c`), so a job's actual settings are always visible in
its own record, not just whatever `.env` happened to say at upload time.

## Job timeouts (why a longer VOD used to fail with "exceeded maximum timeout")

RQ's own default job timeout is 180 seconds -- fine for the short synthetic
clips this pipeline was verified against during development, nowhere near
enough for a real upload once transcription, scene detection, or rendering
has real work to do. `app.workers.common.estimate_job_timeout_seconds`
scales each stage's timeout with the video's own duration (or, for
rendering, the individual clip's duration, since candidate windows are
bounded by `SEGMENT_MAX_CLIP_SECONDS` regardless of how long the source VOD
is) instead of using one fixed number everywhere:

| Stage enqueued | Multiplier | Reasoning |
|---|---|---|
| ingest (from the initial upload) | flat 1800s | duration isn't known yet -- that's what ingest's own ffprobe call determines |
| transcription (from ingest) | 8x duration + 600s buffer | the slowest realistic path -- local CPU Whisper, plus a first-run model-download allowance |
| segmentation (from transcription) | 3x duration + 120s buffer | one PySceneDetect pass over the whole video |
| scoring (from segmentation) | 3x duration + 120s buffer | re-runs the same scene detection segmentation did, plus fast per-candidate math |
| rendering (from scoring, per clip) | 6x *clip* duration + 120s buffer | libx264 encode + caption burn-in on a short (<=90s) clip |

Every multiplier is deliberately generous (worst-case slow/loaded hardware,
not this sandbox's fast synthetic test clips) and floored at a sane minimum
so a short clip doesn't get an unreasonably tight timeout either. If you
still hit a timeout on a very long VOD, these are the numbers to look at
first (`app/workers/common.py`) before assuming something else is wrong.

**Where the actual processing time goes, and what's been done about it:**
transcription (local CPU Whisper) is the dominant cost and scales with audio
duration -- the main lever there is `stt_model_size` (see "Per-job options"
above): a much smaller/faster model for a long VOD trades some accuracy for
real wall-clock time. Scene detection used to run **twice** per job (once
in segmentation, then again from scratch in scoring) -- scoring now reuses
segmentation's persisted `scene_cuts_seconds` instead (see "Scoring" above),
cutting that cost roughly in half and skipping a second raw-video download
entirely for S3-backed storage. Rendering was already cheap relative to the
others: it only ever touches each selected clip's own short window, never
the whole source video. If a long VOD is still slow after picking a smaller
transcription model, transcription itself is almost certainly why --
everything downstream of it is now fast by comparison.

## TikTok setup (Content Posting API)

Real OAuth + real uploads, not stubs -- but TikTok's side needs setup
before either endpoint does anything:

1. Register an app at [developers.tiktok.com](https://developers.tiktok.com),
   add the **Content Posting API** product, enable **Direct Post**, and
   request the `video.upload` scope (the inbox/draft scope this project
   uses -- **not** `video.publish`, which is the direct-to-feed scope for
   the "direct" target_mode this project deliberately doesn't implement,
   per architecture doc §6).
2. Set `TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET` / `TIKTOK_REDIRECT_URI`
   (see `.env.example`) -- the redirect URI must exactly match one
   registered on the app. TikTok requires `https` with no localhost
   exception ("URIs must be absolute and begin with https" -- no
   dev-environment carve-out in their Login Kit docs), so local dev needs a
   tunnel (e.g. `ngrok http 8000`) in front of this server; use the
   tunnel's `https://...` URL + `/api/v1/creator-accounts/tiktok/oauth/callback`
   as both `TIKTOK_REDIRECT_URI` and the URI registered in the developer
   portal. Free tunnel URLs change on restart, so expect to update both
   places again after restarting the tunnel.
3. Until the app passes TikTok's own audit, it's restricted to **private
   (`SELF_ONLY`) posts from up to 5 users in a 24h window** -- enough to
   fully exercise this pipeline, but nothing goes genuinely public until
   audit passes. Start that process early; it's TikTok-side lead time
   this code can't shortcut (architecture doc §10).

**Connecting an account** (dev console, "4. Creator accounts" -- "Connect
TikTok account" button, or directly):
`GET /api/v1/creator-accounts/tiktok/oauth/start` (authenticated) returns
`{authorize_url}`; redirecting a browser there and approving lands back on
`GET /api/v1/creator-accounts/tiktok/oauth/callback` (TikTok's own redirect,
unauthenticated -- it's a plain browser navigation, not an API call, so it
returns a small standalone HTML page, not the API's usual JSON envelope).
That callback exchanges the code for tokens (`app/core/tiktok_client.py`)
and upserts a `creator_accounts` row -- updating tokens in place on
reconnect rather than erroring, unlike the manual path's 409-on-duplicate.
The `state` param is a short-lived, purpose-scoped JWT
(`app.core.auth.create_oauth_state_token`) carrying the connecting user's
id, since the callback itself has no bearer header to authenticate with;
it's rejected if reused as a real access token (`typ` claim mismatch) or if
it's expired/tampered.

`POST /api/v1/creator-accounts` still exists alongside OAuth -- for
platforms without OAuth wired up here yet (youtube, instagram), or to
register a token from TikTok's own sandbox tooling without the
consent-screen round trip. Manually- and OAuth-registered accounts are
identical from that point on; `app/workers/upload.py` doesn't know or care
which path created a given row.

`access_token`/`refresh_token` are encrypted (`app/core/crypto.py`, Fernet)
before ever reaching the DB and are never returned by the API afterward --
`GET /api/v1/creator-accounts` only ever exposes platform/external id/daily
cap/id. In dev, the encryption key is derived from `JWT_SECRET` so this
works with zero extra setup; set a real, separate `CREATOR_ACCOUNT_ENCRYPTION_KEY`
(see `.env.example`) for anything beyond local dev, since rotating
`JWT_SECRET` would otherwise also silently break decrypting every
already-stored token. `daily_upload_cap` is validated against
`HARD_MAX_DAILY_UPLOAD_CAP` (default 10) the same way `max_clips` is
validated against its own hard cap -- see "Per-job options" above.

**Uploading**: `app/workers/upload.py` does the real thing now -- init an
inbox (draft) upload, PUT the video bytes (chunked per
`TIKTOK_UPLOAD_CHUNK_SIZE_BYTES`, though a typical short vertical clip is
well under that and goes as a single chunk), then poll TikTok's status
endpoint (bounded: `TIKTOK_STATUS_POLL_MAX_ATTEMPTS` x
`TIKTOK_STATUS_POLL_INTERVAL_SECONDS`, ~1 minute by default) until it
reports a terminal state. `upload_tasks.publish_id` is what makes a retry
safe: it's persisted the moment TikTok's init call succeeds, and a retried
attempt checks it first -- already set means "bytes are already uploaded,
just resume polling," not "start over" (which would otherwise leave a
second, orphaned draft on TikTok's side). Retryable errors (network
failures, timeouts, TikTok 5xx, "still processing" after this attempt's
polling budget) `raise` so RQ's own retry/backoff handles it; everything
else (blocked by cap/confidence/target_mode, a missing account/clip, a
TikTok 4xx/policy rejection) is marked `failed`/`blocked` in place and
`run()` returns normally, so RQ doesn't retry something a retry can't fix.
An access token nearing expiry is refreshed automatically first (if a
refresh_token is on file) -- see that function's docstring for what happens
when there isn't one.

Available from the dev console too (the "Add a creator account" panel and
"Connect TikTok account" button under "4. Creator accounts").

## Reviewer feedback & ratings

`POST /api/v1/clips/{id}/review` (the same endpoint that records
approve/reject/skip) now also accepts an optional `rating` (1..5,
`app/schemas.py`'s `ReviewRequest.rating`), stored on `review_decisions` and
validated (422 outside 1..5) the same way any other request field is. This
is a deliberately separate signal from the approve/reject decision itself:
`decision` gates whether a clip is allowed to post at all; `rating` is a
richer, optional "how good is this, really" signal a reviewer can attach
regardless of decision (you might approve a clip you'd only call a 2/5
because it's fine but forgettable). The dev console shows a row of ☆/★
buttons under each clip's Approve/Reject row -- pick 1-5 before clicking
Approve or Reject to attach a rating to that decision, or a "clear" link to
remove a selection. `RenderedClipOut.latest_rating` surfaces the most
recent rating across a clip's review history (not erased by a later
re-review that doesn't set one) so the stars restore correctly on reload.

**Why this exists, and what it's *not*** (see project instructions'
AI/ML principles -- heuristic first, learnable later): there's no
automatic learning loop consuming these ratings yet, and there shouldn't be
one built prematurely -- a handful of ratings from one creator is nowhere
near enough signal to retrain anything, and doing so would contradict the
project's explicit stance against pretending virality/quality prediction is
solved. The intended near-term use is manual: once enough ratings
accumulate across real jobs, a simple offline query correlating
`review_decisions.rating` against each clip's `score_breakdown` (already
stored per candidate, see "Scoring" above) lets you sanity-check whether
`SCORE_WEIGHT_*` actually lines up with what you thought was good -- a
human-in-the-loop tuning aid, not an autonomous model. A `clip_performance`
table pulling real TikTok view/like counts post-publish is the natural
next step beyond ratings alone (flagged as **post-MVP** -- it needs real
production upload volume to be worth anything) and would slot into the
same "manual weight review" use, not a different one.

## Training on your own approved clips

You can turn your own real reviewer ratings (see "Reviewer feedback &
ratings" above) into an actual OpenAI fine-tuned model, once you have
enough of them -- a genuinely simple path, but worth understanding what
it actually is before spending money on it.

**What this is, concretely.** `scripts/export_finetune_dataset.py` reads
every `rendered_clips` row that's both `approved` and rated at or above
`--min-rating` (default 4/5), and re-derives the *exact* prompt
`app.core.caption_generation` sends in production for that clip (same
transcript window, same ranking-note hint, same fallback caption), paired
with the title/hashtags/caption/explanation you actually approved. That
becomes one line of an OpenAI-format JSONL file:

```
PYTHONPATH=. python scripts/export_finetune_dataset.py --min-rating 4 --out finetune_dataset.jsonl
```

From there, kick off a real fine-tune job with OpenAI's own CLI/API (not
something this project wraps -- it's already about as simple as it gets):

```python
import openai
client = openai.OpenAI()
client.files.create(file=open("finetune_dataset.jsonl", "rb"), purpose="fine-tune")
client.fine_tuning.jobs.create(training_file="<file id from above>", model="gpt-4.1-mini-2025-04-14")
```

Once the job completes, OpenAI gives you a new model ID -- set
`CAPTION_LLM_MODEL` to it (and `CAPTION_LLM_PROVIDER=openai`) and every
future clip's title/hashtags/caption gets generated by your fine-tuned
model instead of the stock one.

**What this is NOT, and why the export script filters the way it does.**
This is real supervised fine-tuning (OpenAI hosts and runs the training,
you're not managing any ML infrastructure), not the free-form "train an
AI on everything" idea -- it's narrowly scoped to *this one task*
(clip title/hashtag/caption generation), trained only on clips you
personally reviewed and approved. Only `approved` clips at or above your
rating threshold go in -- a clip you rated 5/5 but rejected for an
unrelated reason (e.g. sponsor/legal concerns) is deliberately excluded,
since the point is teaching the model your posting judgment, not just
"things that scored high." This also sidesteps the copyright/ToS concerns
of training on other creators' clips discussed earlier -- every example is
something you made and chose to post.

**Before running an actual training job, know the real numbers** (verified
against OpenAI's current docs, not assumed from memory -- these change,
recheck before spending money):

- OpenAI's supervised fine-tuning currently supports the `gpt-4.1`/
  `gpt-4.1-mini`/`gpt-4.1-nano` family, **not** `gpt-4o-mini` (this
  project's current `CAPTION_LLM_MODEL` default) -- you'd point
  `CAPTION_LLM_MODEL` at `gpt-4.1-mini-2025-04-14` or similar once fine-tuned,
  not keep the current default.
- **10 examples is the hard technical minimum** the API will accept; OpenAI's
  own guidance says real quality improvement generally shows up around
  **50-100+ examples**. The export script warns you at the bottom of its
  output if you're under 10 and reminds you to keep rating clips.
  Realistically: don't run a real training job until you've shipped enough
  real jobs to clear that bar -- a fine-tune on 8 examples is closer to
  memorizing them than learning a style.
- Fine-tuning has a real, recurring cost: a one-time per-token charge to
  train, **and** a higher per-token rate for every inference call against
  the resulting model afterward, for as long as you use it (check
  https://openai.com/api/pricing for current numbers -- they change, and
  varied noticeably across the sources checked while writing this). This
  is an ongoing cost tradeoff versus the current `gpt-4o-mini`/Ollama setup,
  not a one-time expense.
- This is a **batch process, not continuous learning** -- rating more clips
  doesn't retrain the model automatically. Re-run the export script and
  kick off a new fine-tune job manually whenever you want to fold in more
  recent ratings (e.g. monthly, or whenever you've accumulated another
  50-100 good examples).

**Recommended default**: keep rating clips as you review them (free,
already built) and don't run an actual fine-tune job until the export
script's example count comfortably clears 50+. There's no cost or
commitment to waiting -- the data just keeps accumulating in
`review_decisions` either way.

## Deleting things

Both `DELETE /api/v1/stream-jobs/{id}` and `DELETE /api/v1/clips/{id}` are
real -- not stubs. Deleting a stream_job removes its transcript,
candidate_segments, and every rendered_clip under it (cascades at the DB
level via `ondelete="CASCADE"`), plus the actual storage objects (raw
video, extracted audio, each clip's rendered video/thumbnail) -- a DB-only
delete would just leak files. Deleting a single clip does the same for
just that clip's video/thumbnail. Both are available from the dev console
(a 🗑 button per job row, and per clip card) as well as directly via the
API. Neither endpoint currently guards against deleting a job that's
mid-pipeline -- an in-flight worker job for a since-deleted stream_job_id
already handles "row not found" as a clean no-op everywhere in this
codebase, so it's safe either way.

## Dev console (frontend)

A plain HTML/JS dev console is served by the API itself at
`http://localhost:8000/` -- no separate frontend server, no build step, no
npm install. It's `web/index.html` + `web/app.js` + `web/style.css`, mounted
via `StaticFiles` in `app/main.py`. Open it, paste the bearer token from
`scripts/create_dev_user.py` into the token field (saved in that browser's
localStorage so you don't re-paste it every reload), and you can:

- upload a video, optionally tuning it via the "Options" panel (transcription
  model, max clips, minimum score, clip length bounds -- see "Per-job
  options" above), and watch its `stream_job` status update live (polls
  every 3s; toggle off with the checkbox),
- see the raw job JSON and `last_error` for whichever stage a job failed
  on, if any,
- once a job reaches `scored`, see each selected clip's card: its
  **score** (a `pill-ok`/`pill-warn`/`pill-unknown` badge, 0..10 -- hover
  for a per-feature breakdown tooltip; see "Scoring" above for what it
  means and, just as importantly, what it doesn't), a thumbnail (loads
  automatically), a "▶ Watch clip" button that loads and plays the actual
  rendered `.mp4` inline, its caption, generated hashtags (a "generating
  hashtags…" placeholder until `caption_generation` finishes, then pills
  plus a `✨ AI` / `heuristic` / `✏️ edited` source badge and a short
  reviewer-facing explanation, with an `✏️ edit` button to overwrite the
  hashtags inline -- see "Clip hashtags & captions" above), and any flags (e.g.
  `captions_failed`) -- then review (approve/reject) and
  upload-to-TikTok-draft it. Video/thumbnail bytes come from
  `GET /api/v1/clips/{id}/video` and `/thumbnail`, fetched as an
  authenticated blob (a plain `<video src>` can't carry the bearer token) --
  fine for this dev console, not how a production player would serve media
  (that wants signed URLs / a CDN in front of storage, not the API process
  proxying file bytes),
- delete a job you don't need anymore (🗑 next to it in the jobs table) or
  a single clip you've reviewed and don't want cluttering the queue (🗑
  Delete on its card) -- both confirm before deleting, both are permanent
  (see "Deleting things" above),
- register a creator account by pasting a token you already have (the "Add a
  creator account" panel -- see "Creator accounts" below), then use it as an
  upload target on any approved clip.

This is a dev tool, not the product's creator-facing UI -- it exists to
make the API and pipeline state visible while you're building, not to be
pretty. Same origin as the API, so there's no CORS config to worry about.

## Running tests

Tests talk to a real Postgres/Redis (no mocking of the DB or queue), using
a separate `clipping_machine_test` database and Redis DB index 15 so they
don't touch your dev data:

```bash
createdb clipping_machine_test   # once, if it doesn't exist yet
PYTHONPATH=. pytest tests/ -v
```

One test (`tests/test_local_whisper_provider.py`) does a real end-to-end
run of the local Whisper path against real synthesized speech -- it needs
`espeak-ng` on PATH (`apt install espeak-ng` / `brew install espeak-ng`) to
generate that speech and real internet access to download the model
weights; it skips itself cleanly if either isn't available, so it's fine to
leave uninstalled. It's the one test worth running manually once on a real
machine as final confirmation that open-source transcription genuinely
works end to end, since the rest of the STT test suite fakes the provider
to stay fast and network-independent.

`tests/conftest.py` creates/drops tables per session via SQLAlchemy
metadata (not Alembic) for speed -- that's a deliberate choice for test
speed, not a claim that migrations don't need their own verification
(they were hand-verified: `alembic upgrade head` / `downgrade base` /
`upgrade head` round-trips cleanly, see `alembic/versions/`).

## Repo layout

```
backend/
  app/
    api/routers/     # FastAPI endpoints -- one router per resource
    core/            # config, logging, storage, queue, auth -- cross-cutting
    core/stt/         # speech-to-text: provider interface + local Whisper / OpenAI implementations
    core/segmentation_logic.py  # scene-cut + speech-block window building (pure, unit-tested separately)
    core/scoring_logic.py       # feature extraction + composite scoring + non-max suppression (pure, unit-tested separately)
    core/rendering_logic.py     # crop math + SRT building (pure, unit-tested separately)
    core/face_detect.py         # best-effort face detection for the crop's focal point (see "Rendering")
    core/caption_logic.py       # hashtag/caption prompt building + response parsing (pure, unit-tested separately)
    core/caption_generation.py  # the OpenAI-calling half of caption_logic (see "Clip hashtags & captions")
    core/crypto.py              # creator-account token encryption at rest
    db/               # SQLAlchemy models + session
    workers/          # one module per pipeline stage
  alembic/            # migrations
  scripts/            # dev-only helper scripts (not part of the app)
  tests/
  web/                # dev console -- static HTML/JS/CSS, served by the API
  worker_entrypoint.py
docs/
  architecture.md      # full system design doc
```

## Next steps (in priority order)

1. **TikTok app review/audit**: the code side (OAuth + real inbox upload)
   is done -- what's left is TikTok-side lead time. Start the developer app
   review process now if it hasn't been already (see "TikTok setup"
   above); until it passes, posts are restricted to private/`SELF_ONLY`
   and a handful of test users.
2. **Job-status rollup**: add the small aggregate check that moves
   `stream_job.status` to `ready_for_review` (or `failed_rendering`) once
   every selected clip's render job reaches a terminal state --
   deliberately not built alongside rendering itself (see `rendering.py`'s
   docstring); today you check per-clip status via
   `GET /api/v1/stream-jobs/{id}/clips` instead.
3. **Motion-tracked reframing** (later research, not MVP): rendering's crop
   (and the split reaction layout's two crops) are face-aware but still a
   single static choice per clip, not frame-by-frame subject-tracking --
   fine for a webcam that stays roughly put, less so for gameplay where the
   "moment" moves around the frame or a streamer who moves a lot within one
   clip. Needs a real tracking signal this MVP doesn't have yet.
4. **Reaction-layout webcam-box exclusion** (later research, not MVP): the
   split layout's bottom half is a plain full-frame crop, not aware of
   exactly where the webcam box sits, so it may still show a small sliver
   of it -- real rectangle detection/exclusion would fix this but isn't
   needed for a usable split layout today.
5. **Status-polling as its own job** (later research, not MVP): the upload
   worker currently polls TikTok's status endpoint inline (`time.sleep`
   between calls), tying up one worker slot for up to about a minute per
   attempt -- fine at one-creator/one-replica scale, worth splitting into a
   separate poll-only job if upload queue depth ever becomes a real
   bottleneck.

Nothing in the pipeline is a stub anymore -- every stage in the table above
does the real thing.
