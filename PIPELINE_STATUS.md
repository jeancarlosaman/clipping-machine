# Clipping Machine — Pipeline Documentation & Project Status

Last updated: 2026-08-22

This is the single "where are we, how does it work, what's next" reference for the project. `README.md` (in the repo root) is the developer-facing setup/reference doc with more implementation detail per feature; this document is the higher-level picture.

## 1. Where things stand right now

The core pipeline — upload a video, get ranked short-form clips with captions and burned-in titles, review them, send one to a TikTok draft — is **fully built and passing 253 automated tests**. Nothing in the pipeline is a stub or placeholder anymore. What's *not* yet true:

- **Never run end-to-end against your own real footage.** Every stage has been verified with real code paths and a real database in the development sandbox, but not watched start-to-finish on an actual stream VOD on your machine until very recently (you've now run it once, hit two real bugs along the way — see section 6 — both fixed).
- **TikTok upload has never talked to a real TikTok account.** The OAuth flow and upload worker are real, complete implementations of TikTok's documented API, verified with mocked HTTP calls — not a live round trip. You still need to register a TikTok developer app.
- **The AI-written titles' quality is unverified/unproven** — you've reported hashtags looking better but titles still weak, and we added tooling this session specifically to help diagnose whether that's a wiring problem or a small-model quality ceiling (see section 6).

## 2. How the pipeline works

A stream job moves through seven stages, each a separate background worker pulling from its own queue (Redis + RQ), so one stage's backlog never blocks another:

```
 upload
   │
   ▼
┌─────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────┐   ┌───────────┐   ┌────────┐
│ INGEST  │──▶│ TRANSCRIPTION│──▶│ SEGMENTATION │──▶│ SCORING │──▶│ RENDERING │──▶│ REVIEW │──▶ (optional) UPLOAD
└─────────┘   └──────────────┘   └──────────────┘   └─────────┘   └───────────┘   └────────┘
  extract       speech-to-text     candidate clip      rank &       burn captions    human        TikTok
  audio,        with timestamps    windows (scene       select      + title,          approve/     draft
  probe                            cuts + silence       top N        two layouts       reject
  duration                         gaps)
```

**Ingest** — pulls the raw video, extracts its audio track, probes duration. Enqueues transcription on success. Codec/container errors fail immediately (no point retrying a corrupt file); transient I/O errors retry 3x.

**Transcription** — turns the audio into a timestamped transcript. Provider is a config switch: `STT_PROVIDER=local` (default) uses open-source Whisper running on your own CPU, free, no API key; `STT_PROVIDER=openai` uses OpenAI's hosted Whisper API, costs per minute, needs `OPENAI_API_KEY`. Both go through the same interface, so this is a config change, never a code change.

**Segmentation** — combines PySceneDetect's visual scene-cut detection with the transcript's speech/silence structure to propose candidate clip windows (start/end timestamps). This is deterministic, not AI — pure window-building logic based on where people stop talking and where the camera/scene changes.

**Scoring** — ranks every candidate window with a deterministic heuristic (no LLM involved in choosing clips, per the project's core AI/ML principle): speech density, pause structure, question/payoff patterns, emotional language, scene changes, motion. Picks the top N non-overlapping windows (so two overlapping candidates of the same moment don't crowd out a genuinely different moment elsewhere), optionally filtered by a minimum score. Creates one "rendered clip" row and rendering job per selected window.

**Rendering** — the actual video work, via ffmpeg: crops to vertical 9:16, picks one of two layouts per clip (a single face-aware crop, or a split layout with a facecam zoom on top and full content below, chosen automatically based on whether a small cornered webcam box is detected), burns in the transcript as bottom captions and a clickbait-style hook as a top title banner. The title/hashtags/caption text come from `app.core.caption_generation`, called synchronously here (has to happen before rendering because the title gets burned into pixels) — either a real LLM call (OpenAI or, as of this week, a free self-hosted Ollama model) grounded in that clip's own transcript, or a generic heuristic template if the LLM path is off/unavailable/failed. Each clip renders independently, so one clip's failure doesn't block its siblings.

**Review** — the dev console (`http://localhost:8000/`) shows every rendered clip with its score, title, hashtags, and a preview player. You approve, reject, edit the title/hashtags/caption inline, or download the raw mp4. Nothing uploads without an explicit "approved" review decision first — this is the "human review required by default" rule from the project's core constraints, enforced at the API level, not just the UI.

**Upload (optional)** — for an approved clip, sends it to TikTok as a private draft in your TikTok inbox (not a public post — you still tap "post" yourself in the TikTok app). Real OAuth (connect your TikTok account once), real chunked upload, real status polling, with daily-cap and confidence-threshold gates that block a low-confidence or over-quota upload before it ever reaches TikTok.

### Data model

Nine tables carry a stream job from upload to (optional) TikTok draft:

| Table | Holds |
|---|---|
| `users` | dev-minted accounts (no real signup/login yet) |
| `creator_accounts` | connected TikTok/YouTube/Instagram accounts, encrypted tokens |
| `stream_jobs` | one row per uploaded VOD, tracks pipeline status |
| `transcripts` | timestamped speech-to-text segments |
| `candidate_segments` | proposed clip windows, scores, and the AI/heuristic title+hashtags+caption annotation |
| `rendered_clips` | the actual rendered mp4 + thumbnail + status |
| `review_decisions` | approve/reject/skip history per clip |
| `upload_tasks` | one row per upload attempt, with retry-safe `publish_id` tracking |
| `upload_audit_logs` | every upload attempt logged, per the project's hard safety-limit requirements |

A `stream_job.status` moves through `queued → ingesting → ingested → transcribing → transcribed → segmenting → segmented → scoring → scored → rendering → ready_for_review | failed_*` at each stage boundary — **except** the final `rendering → ready_for_review` step, which is a known gap (see section 5, item 1): right now nothing rolls the job status forward once every clip finishes rendering, so a job can sit at `rendering` in the job list even though every clip underneath it is actually done and viewable.

### API surface

Standard REST, JSON, bearer-token auth (dev tokens minted via `scripts/create_dev_user.py`, no real signup flow yet):

- `POST/GET /api/v1/stream-jobs`, `GET .../{{id}}`, `DELETE .../{{id}}`, `GET .../{{id}}/clips`
- `POST /api/v1/clips/{{id}}/review`, `PATCH .../caption`, `POST .../upload`, `DELETE .../{{id}}`, `GET .../video`, `GET .../thumbnail`
- `GET/POST /api/v1/creator-accounts`, `GET .../tiktok/oauth/start`, `GET .../tiktok/oauth/callback`
- `GET /api/v1/upload-tasks/{{id}}`

### Key configuration switches

Everything below lives in `backend/.env` (see `.env.example` for the full annotated list):

- `STT_PROVIDER` — `local` (free, open-source Whisper) or `openai` (hosted, costs per minute)
- `CAPTION_LLM_PROVIDER` — `openai` (hosted, costs per call) or `ollama` (free, self-hosted, needs Ollama running locally)
- `ENABLE_LLM_CAPTIONS` — off falls back to generic heuristic titles/hashtags/captions entirely
- `ENABLE_FACE_AWARE_CROP` / `ENABLE_REACTION_SPLIT_LAYOUT` — rendering layout behavior
- `ENABLE_CLIP_TITLE_OVERLAY` — whether the title actually gets burned into the video (vs. just stored as metadata)
- `DEFAULT_MAX_CLIPS_PER_JOB`, `DEFAULT_MIN_SCORE_THRESHOLD`, `SEGMENT_MIN/MAX_CLIP_SECONDS` — how many clips you get and how long they are
- `DEFAULT_DAILY_UPLOAD_CAP`, `UPLOAD_CONFIDENCE_THRESHOLD` — the hard safety limits the project's own rules require
- `TIKTOK_CLIENT_KEY` / `SECRET` / `REDIRECT_URI` — needed before "Connect TikTok account" works at all

## 3. What's actually verified vs. assumed

| Piece | Status |
|---|---|
| Full pipeline logic (ingest → rendering) | Verified via 253 automated tests against a real database, and once live on your machine |
| Windows compatibility | Two real Unix-only RQ bugs found and fixed this week from your actual terminal output — see section 6. Assume a third might still exist; this sandbox is Linux and can't catch Windows-only bugs by running tests alone |
| TikTok OAuth + upload | Real implementation of TikTok's documented API, verified only via mocked HTTP calls — never hit TikTok's real servers |
| Ollama caption/title provider | Confirmed running (server up, model pulled) on your machine; title *quality* on your real content not yet confirmed good — tooling to check this was added this session (hover a clip's badge in the dev console) |
| Face-aware crop / split reaction layout | Built and unit-tested; not yet confirmed to look right on your actual footage |
| Job-status rollup | **Not implemented** — see section 5 |

## 4. Recommendations (priority order)

1. **Finish one full real run and actually watch the output.** You're close — you've gotten a job through ingest/transcription/segmentation on your machine already. The highest-value next step isn't a new feature, it's watching 5-10 real clips come out the other end and judging whether the crop, captions, and titles actually look good on your content. Everything downstream (TikTok upload, storyline compilation, anything else) is premature to prioritize until you know the core clip quality is right.
2. **Don't chase the Ollama title quality too hard yet.** A 7-8B local model will never match `gpt-4o-mini`'s writing. If titles stay weak after confirming (via the badge) that Ollama really is running, the fix is either a bigger local model or switching back to OpenAI for this one call — it's cheap per-call even if you keep everything else local, since it's one short text generation per clip, not per-minute video processing.
3. **Register the TikTok developer app now, in parallel** — not because you need it this week, but because TikTok's own app-review lead time is the slowest thing in this whole loop and doesn't depend on anything else being finished first.
4. **Hold off on bigger features** (the "storyline across the whole VOD" idea we scoped earlier, motion-tracked reframing, etc.) until the core single-clip loop has actually produced a clip you'd post. Per the project's own founding principle, the smallest version that actually works beats a more ambitious version that hasn't been tested.

## 5. Next steps (concrete, prioritized)

1. **Job-status rollup** (small, real gap): nothing currently flips `stream_jobs.status` to `ready_for_review` once every clip under it finishes rendering — it can sit at `rendering` in the job list forever even though the clips are done. Worth fixing before this becomes confusing at real usage volume.
2. **TikTok developer app registration + audit** — register at developers.tiktok.com, add the Content Posting API product, set the three `TIKTOK_*` env vars, stand up an `https` tunnel (ngrok) for local OAuth testing since TikTok requires `https` with no localhost exception, and start their audit process (unaudited apps are private-only, capped at 5 test users).
3. **A real end-to-end pass on your own footage** — upload an actual stream VOD (not just a short test clip), let it run all the way through, and actually watch the resulting clips: does the crop look right, are captions positioned well, do titles feel honest to what's said, does the split-reaction layout trigger correctly if you have a webcam.
4. **Ollama title quality check** — now that you can see the model badge, confirm titles are genuinely LLM-written (not silently falling back), then judge on merit whether `llama3.1:8b`'s output is good enough or whether a bigger local model / switching back to OpenAI for this one call is worth it.
5. Later research, not urgent: motion-tracked reframing (crops are currently static per clip, not frame-by-frame), excluding the webcam box's own rectangle from the split-layout's bottom half, tuning the reaction-layout/scene-detection thresholds against your real content once you have some to look at, and the "storyline across the whole VOD" compilation feature we scoped as a real but substantial post-MVP addition (schema change + new selection logic + rendering changes).

## 6. What we fixed this week (context, not action items)

For the record, since some of this happened over several back-and-forth turns: implemented the full real TikTok OAuth + upload flow (previously stubbed), added a free self-hosted Ollama option for AI-written titles/hashtags (previously OpenAI-only), added a one-click clip download button for manual posting, added a way to see in the dev console exactly which model produced a given title (to debug quality complaints), corrected a wrong claim in the docs about TikTok allowing localhost redirect URIs (it doesn't — needs a tunnel), and found + fixed two separate Windows-only crashes in the background worker (RQ, the queue library, hardcodes Unix-only timeout mechanisms in two different, unrelated places — the first was caught before you ran the app; the second only surfaced when you actually ran the worker on your real machine and pasted the traceback).
