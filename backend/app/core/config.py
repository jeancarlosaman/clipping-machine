"""Centralized app configuration, loaded from environment variables / .env.

Every other module should import `settings` from here rather than reading
os.environ directly -- keeps config surface auditable in one place.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    database_url: str = "postgresql+psycopg2://clipping_machine:clipping_machine@localhost:5432/clipping_machine"

    # --- Queue ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Object storage ---
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_dir: str = "./data/objects"
    s3_bucket: str = ""
    s3_region: str = ""
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # --- App ---
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    jwt_secret: str = "change-me-dev-only"
    max_upload_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GiB

    # --- Creator account token encryption (app/core/crypto.py) ---
    # A dedicated Fernet key for encrypting creator_accounts.access_token_encrypted
    # / refresh_token_encrypted at rest. If unset, app/core/crypto.py derives a
    # key from jwt_secret instead so this works out of the box in dev -- set a
    # real, separate value here for anything beyond local dev (rotating
    # jwt_secret would otherwise also silently break decrypting stored tokens).
    creator_account_encryption_key: str = ""

    # --- Product limits (hard caps -- see architecture doc §5) ---
    default_max_clips_per_job: int = 10
    hard_max_clips_per_job: int = 20  # absolute ceiling regardless of what a creator requests
    default_daily_upload_cap: int = 3
    hard_max_daily_upload_cap: int = 10  # absolute ceiling on a creator account's own daily_upload_cap
    upload_confidence_threshold: float = 0.55

    # --- Per-job override bounds (app.api.routers.stream_jobs validates
    # user-supplied max_clips/min_score_threshold/min_clip_seconds/
    # max_clip_seconds against these before storing them on the job) ---
    min_clip_seconds_floor: float = 5.0
    max_clip_seconds_ceiling: float = 180.0
    default_min_score_threshold: float = 0.0  # 0..10 scale; 0 = no filtering

    # --- Scoring (app/core/scoring_logic.py) ---
    # Weights for the deterministic composite score -- heuristic ranking,
    # not a virality model. Tune these once real usage data exists; there's
    # no ground truth to derive them from yet. motion_proxy defaults to 0
    # because that feature isn't implemented (see scoring_logic.py) -- a
    # nonzero weight on an always-zero feature would just be misleading.
    score_weight_speech_density: float = 1.0
    score_weight_pause_count: float = 0.5
    score_weight_question_marks: float = 1.0
    score_weight_emotional_language: float = 1.5
    score_weight_scene_changes: float = 0.5
    score_weight_motion_proxy: float = 0.0
    # On by default (unlike laughter/crowd_reaction below) -- hook_strength
    # is a pure transcript-timing feature, same cost class as
    # speech_density/question_marks, no new dependency. Weighted the same
    # as speech_density/question_marks as a starting guess: it's a
    # completion-rate proxy (see scoring_logic.py's module docstring for the
    # TikTok/YouTube Shorts research behind it), not proven stronger or
    # weaker than the existing signals yet -- there's no ground truth to
    # calibrate against until real approved/rejected clips accumulate.
    score_weight_hook_strength: float = 1.0
    # Only actually used when enable_audio_event_scoring is on below --
    # weighted the same as speech_density/question_marks as a starting
    # guess (comparable "one signal among several" scale, not "emotional
    # language"'s higher 1.5, since audio-detected reactions and
    # text-detected emotional language overlap somewhat and shouldn't both
    # be weighted as the single strongest signal). Revisit once real jobs
    # produce enough approved/rejected clips to sanity-check against.
    score_weight_laughter: float = 1.0
    score_weight_crowd_reaction: float = 1.0

    # --- Audio-event detection (app/core/audio_events.py) ---
    # Opt-in: pulls in torch + the panns-inference package + a ~300MB
    # pretrained checkpoint (downloaded once, cached in ~/panns_data/) on
    # first scoring job after this is turned on -- a real new dependency
    # weight this project didn't carry before (previously the heaviest
    # dependency was faster-whisper's CTranslate2 runtime, no torch at
    # all). Off by default so this stays fully optional, same posture as
    # STT_PROVIDER=openai vs. local. See app.core.audio_events' module
    # docstring for two real Windows-only bugs found and fixed in the
    # underlying panns_inference package's own data-download code.
    enable_audio_event_scoring: bool = False

    # --- Segmentation (app/core/segmentation_logic.py) ---
    segment_min_clip_seconds: float = 15.0
    segment_max_clip_seconds: float = 90.0
    # Gap between two transcript segments treated as a natural pause
    # boundary between distinct speech blocks, not just a breath.
    segment_silence_gap_seconds: float = 1.2
    # PySceneDetect analyzes only every (this+1)th frame of the source video
    # -- see detect_scene_cuts' docstring for why this is a real decode-cost
    # reduction (not just discarded work), and why the resulting
    # up-to-N-frames timestamp slop doesn't matter for how these cuts get
    # used. 2 = analyze every 3rd frame (~3x less decode work for this
    # stage, the dominant cost of segmentation for a long VOD); 0 = analyze
    # every frame (PySceneDetect's own default, most precise, slowest).
    # 2 -> 4 (2026-09-11). Benchmarked against a 120s 1080p30 clip with 11
    # known hard cuts: frame_skip=2 ran at 21.8x realtime, frame_skip=4 at
    # 25.5x, BOTH finding 11/11 cuts with zero false positives. ~17% off the
    # slowest stage for no measured accuracy cost. An ffmpeg-native
    # `select=gt(scene,...)` detector was benchmarked as the alternative and
    # REJECTED: barely faster (26.8x) and it missed 2 of the 11 cuts.
    # Caveat: hard cuts between synthetic sources are the easy case; soft
    # transitions in real footage are harder, so if scene-derived boundaries
    # start looking sloppy on real VODs, drop this back to 2 before
    # suspecting anything else.
    scene_detect_frame_skip: int = 4

    # --- Rendering / vertical crop (app/core/rendering_logic.py, app/core/face_detect.py) ---
    # The default 9:16 crop is centered on the source frame, which cuts off
    # a streamer's webcam whenever it isn't dead-center (i.e. almost always
    # -- webcams are typically in a corner). When enabled, rendering samples
    # a few frames from each clip's own window and biases the crop toward
    # the largest detected face instead of the frame's geometric center;
    # falls back to the old centered behavior if no face is found or
    # detection itself fails for any reason, so this can never make a clip
    # worse than before the feature existed. Disable if a stream never has
    # a visible face (pure gameplay capture) to skip the extra per-clip
    # ffmpeg frame extracts + cv2 detection cost.
    enable_face_aware_crop: bool = True
    # When face detection finds a small, corner-positioned face (a webcam
    # box over separate main content -- see app.core.rendering_logic
    # .classify_reaction_layout), render a top-facecam/bottom-content split
    # layout instead of a single crop. Has no effect if
    # enable_face_aware_crop is off (nothing to classify). Off falls back
    # to the single face-aware crop even for a clip that would classify as
    # a reaction layout -- useful if the split look doesn't fit a
    # particular streamer's layout.
    enable_reaction_split_layout: bool = True

    # --- LLM-assisted clip title/captions (app/core/caption_generation.py) ---
    # Layered on top of app.core.scoring_logic.default_caption's
    # deterministic placeholder -- never the only caption/title a clip can
    # get. Runs synchronously as part of rendering itself (app.workers
    # .rendering.run, before the ffmpeg filtergraph is built) -- moved
    # there specifically so the "title" field can be burned into the clip
    # (see enable_clip_title_overlay below); it used to run after
    # rendering as its own async step before that requirement existed.
    # With this flag off, or the selected provider unavailable (no API key
    # for "openai"; Ollama not reachable for "ollama"), every clip still
    # gets a title/caption/hashtags -- just the heuristic template versions
    # (app.core.caption_logic.heuristic_title etc.) instead of ones
    # actually grounded in this clip's transcript.
    enable_llm_captions: bool = True
    # "openai": hosted, needs openai_api_key, costs per call. "ollama":
    # free/self-hosted open-weight model via Ollama's OpenAI-compatible API
    # (same request/response shape, just a different base_url and no real
    # key needed) -- same "local, no API key, no per-call cost" pattern as
    # stt_provider="local" above, just for captions instead of transcription.
    # Run `ollama pull llama3.1:8b && ollama serve` (or any instruct model)
    # before switching this on; caption_generation falls back to the
    # heuristic template if Ollama isn't reachable, same as any other LLM
    # failure.
    caption_llm_provider: Literal["openai", "ollama"] = "openai"
    caption_llm_model: str = "gpt-4o-mini"  # model name when caption_llm_provider="openai"
    caption_ollama_model: str = "llama3.1:8b"  # model name when caption_llm_provider="ollama"
    caption_ollama_base_url: str = "http://localhost:11434/v1"
    # Whether the generated "title" (see above) actually gets burned into
    # the top of the rendered video as a second subtitles stage
    # (app.core.rendering_logic.build_title_srt / app.workers.rendering
    # ._TITLE_STYLE) -- independent of enable_llm_captions, which only
    # controls whether that title text is LLM-written vs. the deterministic
    # heuristic fallback. Off skips the extra burn-in stage entirely (the
    # title/hashtags/caption/explanation are still generated and stored,
    # just not baked into the pixels) -- useful if a streamer's own
    # on-screen elements already cover the top of frame, or a reviewer
    # simply doesn't want it.
    enable_clip_title_overlay: bool = True

    # --- LLM segment suggestion (app/core/llm_segmentation.py) ---
    # Opt-in, additive candidate proposer for app/workers/segmentation.py:
    # reads the job's transcript ONCE and asks the same LLM already used
    # for captions above (reuses CAPTION_LLM_PROVIDER/CAPTION_LLM_MODEL/
    # CAPTION_OLLAMA_MODEL/CAPTION_OLLAMA_BASE_URL on purpose -- a second,
    # parallel provider config for the same "call an OpenAI-shaped chat
    # completions endpoint" operation would be pure config sprawl) to
    # suggest a handful of "this looks like a good clip" windows. Those
    # suggestions are inserted into the exact same pending_score candidate
    # pool as the deterministic sliding-window segmentation's output --
    # this NEVER replaces or ranks candidates itself; the existing
    # deterministic scoring + non-max-suppression selection in
    # app/workers/scoring.py picks winners from the combined set exactly
    # like it already does today. Off by default: this is a genuinely new
    # signal, not a proven one yet, and every other optional signal in this
    # codebase (audio events, face-aware crop) shipped opt-in first too.
    # Degrades to zero LLM-suggested candidates (never fails the
    # segmentation job) if disabled, no API key, unreachable, or the
    # response is malformed -- see llm_segmentation.py's module docstring.
    enable_llm_segment_suggestions: bool = False
    # Hard cap on how many LLM-suggested candidates one job can add, so a
    # verbose/misbehaving response can't flood the scoring pool. 8 is a
    # deliberately modest starting point -- these are *additional*
    # candidates on top of whatever the deterministic pass already found.
    llm_segment_max_suggestions: int = 8
    # Safety valve for a pathologically long transcript: sending thousands
    # of transcript segments in one prompt risks blowing past the model's
    # context window and burning a lot of tokens/cost for one job. Above
    # this many segments, skip the LLM suggestion call entirely (degrades
    # to zero suggestions, same as any other failure) rather than silently
    # truncating the transcript, which would bias suggestions toward
    # whichever end of the VOD got kept. Chunking a long transcript into
    # multiple LLM calls is a reasonable improvement but explicitly
    # post-MVP -- see README.
    llm_segment_max_transcript_segments: int = 800
    # How many of this creator's own past reviewed clips (approved OR
    # rejected, but only ones carrying a written reviewer comment) get
    # rendered into the segment-suggestion prompt as few-shot examples of
    # what they personally keep and cut. This is the project's "learn from
    # my feedback" path *without* a training run: few-shot works at ten
    # examples, where fine-tuning a selection/ranking model would want
    # hundreds plus real negatives, a GPU, and a LoRA->GGUF pipeline (see
    # README's "Training on your own approved clips"). 6 is a deliberate
    # middle: enough for a taste signal, few enough that the examples
    # don't crowd out the actual transcript in the model's context. Set to
    # 0 to prompt exactly as this feature did before it existed.
    llm_segment_feedback_examples: int = 6

    # --- Speech-to-text (app/core/stt) ---
    # "local" (default): faster-whisper, open-source, CPU, no API key, no
    # per-minute cost. "openai": hosted Whisper API, needs openai_api_key.
    stt_provider: Literal["local", "openai"] = "local"
    stt_local_model_size: str = "base"  # tiny|base|small|medium|large-v3 -- bigger = slower+more accurate
    stt_local_compute_type: str = "int8"  # int8 is the practical default for CPU inference speed
    stt_openai_model: str = "whisper-1"
    stt_openai_chunk_seconds: float = 600.0  # ~10min chunks to stay under the API's request-size limit

    # --- External providers (optional until the relevant worker is wired up) ---
    openai_api_key: str = ""

    # --- TikTok Content Posting API (app/core/tiktok_client.py, wired up
    # by app/api/routers/creator_accounts.py's OAuth endpoints and
    # app/workers/upload.py) -- see that worker's module docstring for the
    # full flow. tiktok_client_key/secret come from a registered TikTok
    # developer app (Content Posting API product, Direct Post enabled);
    # tiktok_redirect_uri must exactly match one of that app's registered
    # redirect URIs, or TikTok's token exchange rejects the callback.
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    tiktok_redirect_uri: str = ""
    # Per-chunk size for the FILE_UPLOAD flow's PUT requests, bytes. TikTok
    # requires 5MB-64MB per chunk (except when the whole video is the only
    # chunk); 10MB is a safe default for short vertical clips -- most will
    # be a single chunk anyway (see app/core/tiktok_client.py's chunk-plan
    # helper), this only matters once a clip exceeds that.
    tiktok_upload_chunk_size_bytes: int = 10_000_000
    # How long/often app/workers/upload.py polls TikTok's publish-status
    # endpoint after a successful upload, before giving up on *this*
    # attempt and raising a retryable error (RQ then re-enqueues -- the
    # retry resumes polling the same publish_id rather than re-uploading,
    # see that worker's docstring on why that's safe). 20 * 3s = ~1 minute
    # of polling per attempt, generous for an inbox/draft upload which is
    # typically fast to leave PROCESSING_UPLOAD.
    tiktok_status_poll_interval_seconds: float = 3.0
    tiktok_status_poll_max_attempts: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# faster-whisper's valid model sizes -- shared by the API's per-job override
# validation (app.api.routers.stream_jobs) and the dev console's model
# picker, so both stay in sync with what app/core/stt/local_whisper.py can
# actually load without needing to duplicate the list.
STT_LOCAL_MODEL_SIZES: tuple[str, ...] = ("tiny", "base", "small", "medium", "large-v3")
