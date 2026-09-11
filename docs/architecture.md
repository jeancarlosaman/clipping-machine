# Clipping Machine — MVP System Architecture

Status: Draft v1
Date: 2026-08-20
Owner: Jean Carlo

## 1. Goal

Design the full MVP architecture for Clipping Machine end to end — ingest through TikTok draft upload — so a creator can upload one VOD and get 5–10 ranked, captioned, vertical clips to review and send to TikTok as a draft. This doc is the foundation other design docs (scoring model detail, TikTok integration detail, frontend spec) will build on.

## 2. Scope

**MVP now**
- VOD-first ingest (upload; VOD-import URL stubbed but not required to work day one).
- Transcription with timestamps.
- Scene detection + candidate window generation.
- Deterministic scoring/ranking, with an optional LLM pass limited to reranking/annotating the top N candidates.
- Vertical (9:16) rendering with burned-in captions.
- Review queue (approve/reject, one creator at a time).
- TikTok draft/inbox upload via the official Content Posting API, with daily caps, confidence thresholds, and audit logging.

**Explicitly post-MVP / later research** (per project non-goals — flagged here so nothing above quietly grows scope)
- Real-time/live clipping.
- Multi-platform orchestration beyond TikTok draft (YouTube/Instagram are schema-ready but not built).
- Direct publish (non-draft) upload.
- Smart reframing (face/speaker tracking) — MVP uses a fixed center-crop.
- Learned/ML virality scoring — MVP is heuristic-only.
- Team collaboration, analytics dashboards, marketplace features.

## 3. Requirements

**Functional**
- Creator uploads a video file (or later, a VOD URL) and gets a trackable job.
- System produces a transcript with word/segment-level timestamps.
- System proposes candidate clip windows (scene boundaries + speech structure).
- System scores and ranks candidates, keeps top 5–10 per job.
- System renders selected candidates as vertical clips with burned-in captions.
- Creator reviews clips (approve/reject), optionally with LLM-generated caption/hashtag suggestions.
- Creator can send an approved clip to TikTok as a draft; system respects caps and confidence gates.
- Every upload attempt is logged (success, failure, or blocked-and-why).

**Non-functional**
- One creator's job should complete in low-single-digit multiples of VOD length, not real-time — MVP target: a 1-hour VOD fully processed (ingest→ready-for-review) in well under 1 hour end to end; exact SLO is a tuning target, not a hard MVP gate.
- Reliability over cleverness: every worker stage is idempotent and retry-safe; a crashed worker must not corrupt state or double-charge an external API.
- Human review is mandatory by default — no code path uploads without a review_decision of `approved`.
- Single creator workflow first; multi-tenant isolation only needs to be correct (row-level `user_id` scoping), not optimized.
- Cost control: STT and LLM calls are the two expensive per-job costs — both must be boundable (STT: one call per job; LLM: capped to top-N candidates, feature-flaggable off).

**Constraints**
- Python-first, FastAPI, PostgreSQL, FFmpeg, PySceneDetect, OpenAI STT — per project defaults.
- Monorepo, avoid microservices — pipeline stages are workers in one codebase, not separate services, at MVP scale.
- TikTok: official Content Posting API only, draft/inbox target only.

## 4. High-Level Architecture

Single FastAPI service for the API layer, a shared PostgreSQL database for all state, object storage for media artifacts, and a set of queue-consuming worker processes — one worker *type* per pipeline stage, each independently scalable (as separate processes/replicas) but living in the same codebase/repo.

```
                    ┌────────────────────┐
   Creator  ───────▶│   FastAPI (API)    │
                    │  auth, validation, │
                    │  job/read endpoints│
                    └──────────┬─────────┘
                               │ writes job rows / enqueues
                               ▼
                    ┌────────────────────┐        ┌───────────────────┐
                    │    PostgreSQL      │◀──────▶│  Object Storage    │
                    │ (jobs, transcripts,│  keys  │ (S3-compatible:    │
                    │  segments, clips,  │        │  raw video, clips, │
                    │  review, uploads)  │        │  thumbnails)       │
                    └──────────┬─────────┘        └─────────┬─────────┘
                               │                              │
                               ▼                              │
                    ┌────────────────────┐                    │
                    │   Job Queue         │                    │
                    │ (Redis + RQ, see    │                    │
                    │  §8 trade-off)      │                    │
                    └──────────┬─────────┘                    │
                               │                                │
        ┌───────────┬─────────┼───────────┬───────────┐        │
        ▼           ▼         ▼           ▼           ▼        │
  ┌─────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
  │ Ingest  │▶│Transcribe │▶│Segment  │▶│ Score   │▶│ Render  ││
  │ Worker  │ │ Worker    │ │ Worker  │ │ Worker  │ │ Worker  ││
  │(ffmpeg  │ │(OpenAI    │ │(PySceneD│ │(heurist.│ │(ffmpeg  ││
  │ probe/  │ │ STT)      │ │+silence/│ │+optional│ │ cut/crop│◀┘
  │ extract)│ │           │ │speech)  │ │ LLM top-N│ │/caption)│
  └─────────┘ └───────────┘ └─────────┘ └─────────┘ └────┬────┘
                                                            │
                                                            ▼
                                                  ┌───────────────────┐
                                                  │  Review Queue      │
                                                  │  (API surface,     │
                                                  │   human-in-loop)   │
                                                  └─────────┬─────────┘
                                                             │ approve
                                                             ▼
                                                  ┌───────────────────┐
                                                  │  Upload Worker      │
                                                  │  (TikTok Content     │
                                                  │   Posting API,       │
                                                  │   caps + audit log) │
                                                  └───────────────────┘
```

Data flow narrative:

1. Creator `POST`s a video → API stores it in object storage, creates a `stream_jobs` row (`queued`), enqueues an ingest job.
2. **Ingest worker**: validates the file, probes it with ffprobe, extracts audio, stores a normalized proxy if needed → `ingested` → enqueues transcription.
3. **Transcription worker**: sends audio to OpenAI STT, stores timestamped segments in `transcripts` → `transcribed` → enqueues segmentation.
4. **Segmentation worker**: runs PySceneDetect for visual scene boundaries, combines with transcript-derived speech-density/silence signals to propose candidate windows → inserts `candidate_segments` rows → `segmented` → enqueues scoring.
5. **Scoring worker**: computes deterministic features per candidate, ranks them, takes the top N (bounded, e.g. 15–20) and optionally asks an LLM to rerank/annotate (caption, hashtags, "why this clip") those N only, marks the final 5–10 as `selected` → enqueues rendering for each selected candidate.
6. **Rendering worker**: ffmpeg cuts the segment, reframes to 9:16, burns in captions from the transcript slice, writes the output to object storage, creates a `rendered_clips` row (`rendered`). When all selected clips for a job are rendered (or failed), `stream_jobs.status → ready_for_review`.
7. Creator reviews clips via the API; each decision writes a `review_decisions` row.
8. On approve + explicit "send to TikTok" action, an `upload_tasks` row is created (`queued`); the **upload worker** checks the daily cap and confidence threshold, calls the TikTok Content Posting API in draft mode, writes an `upload_audit_log` entry for every attempt, and updates the task to `uploaded`/`failed`/`blocked`.

Each worker stage: reads its input rows from Postgres, does its work, writes its output rows, and enqueues the next stage — never skips ahead, never mutates another stage's rows. This keeps stages independently retryable and testable.

## 5. Data Model

Ownership chain: `users → stream_jobs → transcripts / candidate_segments → rendered_clips → review_decisions / upload_tasks`. `creator_accounts` hangs off `users` and is referenced by `upload_tasks`.

```sql
-- ============ users & accounts ============

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE creator_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    platform TEXT NOT NULL CHECK (platform IN ('tiktok','youtube','instagram')),
    external_account_id TEXT NOT NULL,
    access_token_encrypted TEXT NOT NULL,
    refresh_token_encrypted TEXT,
    token_expires_at TIMESTAMPTZ,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    daily_upload_cap INT NOT NULL DEFAULT 3,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, platform, external_account_id)
);

-- ============ pipeline: stream_jobs ============

CREATE TYPE stream_job_status AS ENUM (
    'queued',
    'ingesting','ingested','failed_ingest',
    'transcribing','transcribed','failed_transcription',
    'segmenting','segmented','failed_segmentation',
    'scoring','scored','failed_scoring',
    'rendering','ready_for_review','failed_rendering',
    'archived'
);

CREATE TABLE stream_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('upload','vod_import')),
    source_url TEXT,                    -- set when source_type = 'vod_import'
    raw_object_key TEXT NOT NULL,        -- object storage key of source file
    duration_seconds NUMERIC,
    status stream_job_status NOT NULL DEFAULT 'queued',
    retry_count INT NOT NULL DEFAULT 0,
    last_error TEXT,
    max_clips INT NOT NULL DEFAULT 10,   -- hard cap, product constraint
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_stream_jobs_user_status ON stream_jobs(user_id, status);

-- ============ transcripts ============

CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_job_id UUID NOT NULL REFERENCES stream_jobs(id) ON DELETE CASCADE,
    provider TEXT NOT NULL DEFAULT 'openai',
    language TEXT,
    full_text TEXT,
    segments JSONB NOT NULL,   -- [{start, end, text, words:[{w,start,end}]}]
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stream_job_id)
);

-- ============ candidate_segments ============

CREATE TYPE candidate_status AS ENUM ('pending_score','scored','selected','rejected');

CREATE TABLE candidate_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_job_id UUID NOT NULL REFERENCES stream_jobs(id) ON DELETE CASCADE,
    start_seconds NUMERIC NOT NULL,
    end_seconds NUMERIC NOT NULL,
    status candidate_status NOT NULL DEFAULT 'pending_score',
    features JSONB,            -- deterministic feature vector, see §9 ranking
    score NUMERIC,
    score_breakdown JSONB,
    llm_annotation JSONB,      -- optional: reasoning, suggested caption, hashtags
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (end_seconds > start_seconds)
);
CREATE INDEX idx_candidate_segments_job ON candidate_segments(stream_job_id, status);

-- ============ rendered_clips ============

CREATE TYPE render_status AS ENUM ('pending','rendering','rendered','failed');

CREATE TABLE rendered_clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_segment_id UUID NOT NULL REFERENCES candidate_segments(id) ON DELETE CASCADE,
    stream_job_id UUID NOT NULL REFERENCES stream_jobs(id) ON DELETE CASCADE,
    status render_status NOT NULL DEFAULT 'pending',
    object_key TEXT,
    thumbnail_key TEXT,
    caption_text TEXT,
    format TEXT NOT NULL DEFAULT 'vertical_9x16',
    duration_seconds NUMERIC,
    requires_review BOOLEAN NOT NULL DEFAULT true,   -- always true in MVP; field exists for future confidence-based skip
    flag_reasons TEXT[] NOT NULL DEFAULT '{}',        -- 'low_confidence','sponsor_sensitive','profanity','copyright_risk'
    retry_count INT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_rendered_clips_job_status ON rendered_clips(stream_job_id, status);

-- ============ review_decisions ============

CREATE TYPE review_decision_value AS ENUM ('approved','rejected','skipped');

CREATE TABLE review_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rendered_clip_id UUID NOT NULL REFERENCES rendered_clips(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    decision review_decision_value NOT NULL,
    notes TEXT,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Multiple rows allowed per clip (a creator can change their mind); the API
-- always reads the latest row by decided_at as the current decision.
CREATE INDEX idx_review_decisions_clip_time ON review_decisions(rendered_clip_id, decided_at DESC);

-- ============ upload_tasks ============

CREATE TYPE upload_status AS ENUM ('queued','uploading','uploaded','failed','blocked');

CREATE TABLE upload_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rendered_clip_id UUID NOT NULL REFERENCES rendered_clips(id) ON DELETE CASCADE,
    creator_account_id UUID NOT NULL REFERENCES creator_accounts(id),
    platform TEXT NOT NULL DEFAULT 'tiktok',
    status upload_status NOT NULL DEFAULT 'queued',
    target_mode TEXT NOT NULL DEFAULT 'draft' CHECK (target_mode IN ('draft','direct')),
    confidence_score NUMERIC,
    block_reason TEXT,          -- 'low_confidence' | 'daily_cap' | 'flagged_content' | 'no_review_approval'
    external_post_id TEXT,
    retry_count INT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_upload_tasks_account_created ON upload_tasks(creator_account_id, created_at);

-- ============ upload_audit_log ============

CREATE TABLE upload_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    upload_task_id UUID NOT NULL REFERENCES upload_tasks(id) ON DELETE CASCADE,
    event TEXT NOT NULL,        -- 'attempt' | 'success' | 'failure' | 'blocked'
    detail JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_upload_audit_log_task ON upload_audit_log(upload_task_id, created_at);
```

Notes:
- `target_mode` defaults to `draft`; a `CHECK` plus an application-level gate (not just a DB default) should refuse `direct` until direct publish ships, since defaults alone don't stop a bad caller.
- `max_clips` and `daily_upload_cap` encode the two hard product limits directly in the schema so they're enforced at write time, not just in application logic.
- `requires_review` and `flag_reasons` exist now (empty/true by default) so the flagging worker logic in a later phase doesn't need a migration — but MVP always requires review regardless of these fields.

## 6. API Design

Assumption: JWT bearer auth on every endpoint below; `user_id` is derived from the token, never trusted from the request body. All list endpoints are paginated (`?limit=&cursor=`). Standard error shape:

```json
{ "error": { "code": "string", "message": "string", "details": {} } }
```

| Method & Path | Purpose |
|---|---|
| `POST /api/v1/stream-jobs` | Create a job from an uploaded file (multipart) or `{source_type:"vod_import", source_url}` |
| `GET /api/v1/stream-jobs` | List the caller's jobs, filterable by `status` |
| `GET /api/v1/stream-jobs/{id}` | Job detail + current status/progress |
| `GET /api/v1/stream-jobs/{id}/clips` | List `rendered_clips` for a job (the review queue payload) |
| `POST /api/v1/clips/{clip_id}/review` | `{decision: "approved"\|"rejected", notes?}` — writes a `review_decisions` row |
| `POST /api/v1/clips/{clip_id}/upload` | `{creator_account_id, platform:"tiktok", target_mode:"draft"}` — creates an `upload_tasks` row |
| `GET /api/v1/upload-tasks/{id}` | Upload status + audit trail |
| `GET /api/v1/creator-accounts` | List the caller's linked platform accounts |
| `POST /api/v1/creator-accounts/tiktok/oauth/callback` | TikTok OAuth token exchange |

Example — create job:

```http
POST /api/v1/stream-jobs
Content-Type: multipart/form-data

file=<video/mp4>
```
```json
201 Created
{
  "id": "9c1e...",
  "status": "queued",
  "source_type": "upload",
  "max_clips": 10,
  "created_at": "2026-08-20T10:00:00Z"
}
```

Example — approve + upload:

```json
POST /api/v1/clips/9c1e.../review
{ "decision": "approved" }

201 Created
{ "id": "a12b...", "decision": "approved", "decided_at": "2026-08-20T11:00:00Z" }
```

```json
POST /api/v1/clips/9c1e.../upload
{ "creator_account_id": "acc_1", "platform": "tiktok", "target_mode": "draft" }

202 Accepted
{ "id": "up_1", "status": "queued", "target_mode": "draft" }
```

Validation rules (representative, not exhaustive):
- Upload file: reject unsupported container/codec at ingest, not at the API layer where possible (return `202` and let ingest fail fast with `failed_ingest` + `last_error`) — this keeps the API layer thin and lets one place (ingest worker) own "is this file usable."
- `target_mode` other than `draft` → `400` until direct publish ships.
- Upload request when no `approved` review decision exists for the clip → `409 Conflict` with `code: "review_required"`.
- Upload request when `creator_account_id`'s daily cap is already hit → `429 Too Many Requests` with `code: "daily_cap_reached"` (checked at request time for fast feedback; re-checked in the worker to close the race).

Error responses use standard codes: `400` validation, `401` unauthenticated, `403` not the resource owner, `404` not found, `409` state conflict (e.g., reviewing a clip that isn't rendered yet), `422` semantically invalid payload, `429` rate/cap limit, `500` unhandled.

## 7. Worker Design

All workers share a pattern: pull a job id off the queue, re-read authoritative state from Postgres (never trust queue payload as source of truth beyond the id), do the work, write results transactionally, enqueue the next stage, and always update `status`/`last_error`/`retry_count` even on failure so the job never goes silently stuck.

| Worker | Trigger | Input | Output artifacts | State transition | Retries | Failure handling |
|---|---|---|---|---|---|---|
| Ingest | `stream_job.created` enqueued by API | `stream_job_id` | Normalized audio track + probe metadata in object storage | `queued → ingesting → ingested` (or `failed_ingest`) | 3x exponential backoff on transient (storage/network) errors; codec/container errors are not retried | On exhaustion: `failed_ingest`, `last_error` set, no further enqueue; surfaced in job detail API |
| Transcription | `stream_job.ingested` | `stream_job_id`, audio object key | `transcripts` row | `transcribing → transcribed` (or `failed_transcription`) | 3x with backoff on API errors (rate limit, timeout); no retry on 4xx from OpenAI (bad audio) | On exhaustion: `failed_transcription`; job is terminal until manual retry endpoint (post-MVP) or job recreation |
| Segmentation | `stream_job.transcribed` | `stream_job_id`, transcript, video object key | `candidate_segments` rows | `segmenting → segmented` (or `failed_segmentation`) | 2x — deterministic CPU work, retries only cover transient I/O | On exhaustion: `failed_segmentation` |
| Scoring | `stream_job.segmented` | `stream_job_id`, all `candidate_segments` for the job | Updated `candidate_segments.{features,score,status}`; optional `llm_annotation` on top-N only | `scoring → scored` (or `failed_scoring`) | 2x for the deterministic pass; LLM sub-step retries 2x independently and is skippable (feature flag) without failing the whole worker | Deterministic scoring failing is a hard failure (`failed_scoring`); LLM annotation failing is logged and degrades gracefully (clip still gets selected, just without LLM caption/reasoning) |
| Rendering | Selected `candidate_segments` after scoring | `candidate_segment_id` | `rendered_clips` row, video + thumbnail in object storage | `pending → rendering → rendered` (or `failed`) per clip; job moves to `ready_for_review` once all selected clips reach a terminal render state | 3x per clip, isolated — one clip's ffmpeg failure doesn't block siblings | Failed clip stays `failed` with `last_error`; job still reaches `ready_for_review` if at least one clip rendered, otherwise `failed_rendering` |
| Upload | Creator action (`POST /clips/{id}/upload`) | `upload_task_id` | TikTok draft post id, `upload_audit_log` rows | `queued → uploading → uploaded` (or `failed`/`blocked`) | 3x with backoff on 5xx/timeout from TikTok API; no retry on 4xx (auth/policy rejection) | Every attempt writes an `upload_audit_log` row regardless of outcome; cap/confidence checks happen *before* the external call and write `blocked` immediately with no API call made |

Logs/metrics (every worker, structured JSON logs): `stream_job_id`, worker name, stage duration, outcome, retry count. Minimum metrics to track from day one: per-stage success/failure rate, per-stage duration (p50/p95), queue depth per stage, upload block rate by reason. This is enough to answer "where is the pipeline slow or breaking" without building a full observability stack.

## 8. Trade-off Analysis

| Decision | Default | Fallback | Why |
|---|---|---|---|
| Job queue | Redis + RQ | Postgres-as-queue (`SELECT ... FOR UPDATE SKIP LOCKED`) | RQ is simple, Python-native, and gives clean retry/backoff semantics without much ops burden — one small extra service. Postgres-as-queue avoids adding Redis at all (one less moving part, appealing given "avoid microservices"), and is a legitimate MVP-scale pattern, but hand-rolls retry/visibility-timeout logic RQ gives for free. Default to Redis+RQ; reach for Postgres-as-queue only if you want to run with zero extra infra for the first weeks. |
| Object storage | S3-compatible (AWS S3 or Cloudflare R2) | Local disk (dev/local only) | Workers and API need to share large binary artifacts across processes/machines; local disk doesn't survive worker autoscaling and isn't a real option past a single dev box. |
| Vertical reframing | Fixed center-crop via ffmpeg | Motion/face-tracked dynamic crop | Center-crop is a few lines of ffmpeg and ships this week. Smart reframing is a real project (detection model, per-frame crop path, smoothing) — explicitly post-MVP, don't build it now. |
| Scoring | Deterministic weighted features, LLM reranks/annotates only the top N | Pure deterministic, no LLM at all | Bounds LLM cost and keeps a non-LLM fallback always available (per project AI/ML principles) while still getting LLM-quality captions/reasoning on the handful of clips that matter. If LLM cost or latency becomes a problem, cutting it entirely is a one-flag change, not a rearchitecture. |
| Captions | Burn in via ffmpeg from transcript segments (styled ASS subtitles) | Simpler SRT `subtitles=` filter | ASS gives per-word/phrase styling (the "punchy captions" look creators expect) for modest extra ffmpeg complexity. SRT is the fallback if styling turns out not to matter for v1. |
| Pipeline stages as workers vs services | All stages in one monorepo codebase, separate worker *processes* per stage, one deployable | Split into separate microservices per stage | Matches the project's explicit "avoid microservices unless truly necessary." Separate processes already give independent scaling and failure isolation without the operational cost of separate services/repos/deploys. |

## 9. Ranking Guidance (deterministic feature set)

Per `candidate_segments.features`, computed with no ML model required:

- **Speech density**: words per second within the window.
- **Silence-to-speech transition count**: sudden silence→speech (or vice versa) often marks a punchline or reaction.
- **Question/payoff structure**: transcript contains a question mark followed by a response within N seconds.
- **Emotional/profanity language**: keyword/lexicon match against transcript text (also feeds `flag_reasons` for manual review, not just scoring).
- **Scene-change count**: PySceneDetect cuts within the window (too many = choppy, zero = static — both penalized relative to a sweet spot).
- **Visual motion proxy**: frame-difference magnitude sampled at low fps (cheap, no model).

Composite score = weighted sum, weights tunable via config (not hardcoded) so they can be adjusted without a redeploy of worker logic. Face/webcam presence detection is explicitly deferred (needs a model — later research). No claim of virality prediction: this is presented to the creator as a *ranking*, not a probability.

## 10. Risks

- **STT and rendering cost/latency at scale**: one full-VOD STT call and several ffmpeg renders per job; fine for one creator, needs load testing before onboarding many. Mitigation: keep both stages horizontally scalable worker pools from day one, even if MVP runs one replica each.
- **TikTok API approval and domain verification**: the Content Posting API requires app review and, for pull-from-URL, a verified domain; this can block the "upload to draft" success criterion if not started early. Mitigation: begin TikTok developer app review in parallel with pipeline build, not after.
- **Human review bottleneck**: mandatory review is a product requirement, not just a risk, but if scoring quality is poor, creators reject most clips and lose trust fast. Mitigation: ship the heuristic scorer with a small internal test set before wiring the LLM pass, so "does ranking feel right" is validated cheaply.
- **Copyright/sponsor-sensitive content**: `flag_reasons` exists in the schema but the keyword/lexicon detection behind it is unbuilt; shipping without it means relying entirely on human review to catch this. Mitigation: treat basic keyword flagging as part of MVP scoring, not a later add-on — it's cheap and the schema already supports it.
- **Worker crash mid-stage leaving a job stuck**: mitigated by design (idempotent stages, explicit status/retry_count columns) but needs an actual periodic reaper (e.g., a cron job that requeues jobs stuck `>N minutes` in a non-terminal `*ing` status) — not yet designed, should be a near-term follow-up, not assumed to "just work."

## 11. Next Action

Scaffold the repo: FastAPI skeleton, Alembic migration for the schema in §5, and a stub ingest worker wired to Redis+RQ — get one job through `queued → ingesting → ingested` end to end before writing any scoring or rendering code. That walking skeleton validates the queue choice, the object storage wiring, and the state-machine pattern all the other workers will copy.
