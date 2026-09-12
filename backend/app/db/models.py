"""SQLAlchemy models -- mirrors the schema in architecture doc §5 exactly.

If you change a table here, update the architecture doc (and vice versa) --
they're meant to stay in lockstep, and an Alembic migration is the only
supported way to apply the change (see alembic/versions/).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

# --- Enums (native Postgres enums, matching architecture doc §5) ---

stream_job_status_enum = Enum(
    "queued",
    "ingesting", "ingested", "failed_ingest",
    "transcribing", "transcribed", "failed_transcription",
    "segmenting", "segmented", "failed_segmentation",
    "scoring", "scored", "failed_scoring",
    "rendering", "ready_for_review", "failed_rendering",
    "archived",
    # Set by POST /stream-jobs/{id}/cancel. Cooperative: workers check it at
    # the top of each stage (see app.workers.common.job_is_cancelled) and
    # stop rather than being killed mid-run, so a job halts at the next
    # stage boundary with nothing left half-written.
    "cancelled",
    name="stream_job_status",
)

candidate_status_enum = Enum(
    "pending_score", "scored", "selected", "rejected",
    name="candidate_status",
)

render_status_enum = Enum(
    "pending", "rendering", "rendered", "failed",
    name="render_status",
)

review_decision_enum = Enum(
    "approved", "rejected", "skipped",
    name="review_decision_value",
)

upload_status_enum = Enum(
    "queued", "uploading", "uploaded", "failed", "blocked",
    name="upload_status",
)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String)

    # --- per-account LLM configuration (app/core/llm_config.py) ---
    # "openai" | "ollama" | None. None means "use settings.caption_llm_provider",
    # which is how every account behaved before this existed.
    llm_provider: Mapped[str | None] = mapped_column(String)
    # Encrypted with app.core.crypto (Fernet), same as creator_accounts'
    # TikTok tokens. This key can spend the holder's money, so it is never
    # stored in plaintext and never returned by the API -- the settings
    # endpoint reports only whether one is set and its last four characters.
    openai_api_key_encrypted: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # passive_deletes=True on every one-to-many below: these FKs all declare
    # ondelete="CASCADE" at the DB level (see each child table's __table_args__
    # equivalent). Without passive_deletes, SQLAlchemy's default ORM behavior
    # is to load the children and UPDATE...SET <fk>=NULL before deleting the
    # parent -- which violates the NOT NULL constraint on every one of these
    # FKs instead of ever reaching the DB's actual CASCADE. This tells the
    # ORM to leave cascading deletes to the database, which is what the
    # schema was already designed to do.
    creator_accounts: Mapped[list["CreatorAccount"]] = relationship(back_populates="user", passive_deletes=True)
    stream_jobs: Mapped[list["StreamJob"]] = relationship(back_populates="user", passive_deletes=True)


class CreatorAccount(Base):
    __tablename__ = "creator_accounts"
    __table_args__ = (
        UniqueConstraint("user_id", "platform", "external_account_id", name="uq_creator_account_identity"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    platform: Mapped[str] = mapped_column(
        String, CheckConstraint("platform IN ('tiktok','youtube','instagram')"), nullable=False
    )
    external_account_id: Mapped[str] = mapped_column(String, nullable=False)
    access_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    daily_upload_cap: Mapped[int] = mapped_column(nullable=False, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="creator_accounts")


class StreamJob(Base):
    __tablename__ = "stream_jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String, CheckConstraint("source_type IN ('upload','vod_import')"), nullable=False
    )
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_object_key: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(stream_job_status_enum, nullable=False, default="queued")
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    max_clips: Mapped[int] = mapped_column(nullable=False, default=10)

    # --- Per-job overrides (all nullable; null means "use the global
    # settings.* default"). Added so a creator can tune a specific upload
    # (e.g. a longer VOD, or one they want fewer/higher-bar clips from)
    # without changing every future job -- see app.api.routers.stream_jobs
    # for validation and app/workers/{transcription,segmentation,scoring}.py
    # for where each is consumed.
    stt_model_size: Mapped[str | None] = mapped_column(String)
    min_clip_seconds: Mapped[float | None] = mapped_column(Numeric)
    max_clip_seconds: Mapped[float | None] = mapped_column(Numeric)
    min_score_threshold: Mapped[float | None] = mapped_column(Numeric)
    # Overrides app.core.rendering_logic.classify_reaction_layout's
    # per-clip auto-detection for every clip in this job. Null (default) =
    # "auto": keep the existing face-size/corner-position heuristic (gated
    # by settings.enable_reaction_split_layout). "single_crop": never
    # split, even if a face is detected as reaction-shaped -- for a VOD
    # with no facecam at all, so a false-positive detection doesn't steal
    # screen space from the actual content. "split_reaction": always use
    # the split layout when ANY face is found on a clip, skipping the
    # size/position classifier -- for a VOD the creator knows has a
    # facecam, when the heuristic's thresholds are guessing wrong for
    # their specific webcam size/position. A clip with no face detected
    # at all still can't be split (nothing to zoom the "cam" half on) --
    # see app/workers/rendering.py for exactly how this gates the decision.
    #
    # A fourth value, "fit_frame", skips cropping entirely: the whole
    # source frame is scaled to the output width over a blurred copy of
    # itself (app.core.rendering_logic.build_fit_frame_filtergraph). For
    # gameplay/screen-share VODs where a 9:16 crop cuts away the part that
    # actually matters, this trades a smaller content band for losing
    # nothing at all.
    camera_layout_mode: Mapped[str | None] = mapped_column(String)
    # Which side of the frame the 9:16 crop should favour, for the two
    # layouts that DO crop: "left"/"center"/"right", null = today's
    # behavior (centered, or following a detected face). An explicit bias
    # wins over face detection -- see
    # app.core.rendering_logic.compute_crop_offset. The escape hatch for
    # "the important thing is consistently on one side of my layout" when
    # fit_frame's smaller content band isn't wanted.
    crop_bias: Mapped[str | None] = mapped_column(String)

    # Where the facecam actually is, marked by hand on a real frame of THIS
    # VOD in the dev console -- normalized to the source frame so it stays
    # correct whatever the source resolution is:
    #     {"x": 0.72, "y": 0.60, "w": 0.26, "h": 0.36}
    # x/y are the box's top-left corner, all four are fractions of the
    # source width/height in 0..1.
    #
    # Set, this beats face detection outright: no Haar cascade run, no
    # classify_reaction_layout guess, and the split layout is forced on
    # (the creator has just told us there IS a facecam and exactly where).
    # That is the whole point -- detection guessing wrong was the original
    # complaint, and the creator looking at their own VOD is the one source
    # of truth that cannot be wrong.
    #
    # Null (the default, and every job predating this) = behave exactly as
    # before: detect a face and let camera_layout_mode/classify decide.
    facecam_rect: Mapped[dict | None] = mapped_column(JSONB)

    # The other half of the same idea: which part of the frame is the actual
    # CONTENT (gameplay, screen share, whatever the viewer is meant to be
    # looking at), same normalized {"x","y","w","h"} shape as facecam_rect.
    #
    # Used as the bottom panel of the split layout, and -- when no split is
    # happening -- as the region a single 9:16 crop is taken from, which is
    # the direct answer to "the screen is not showing the most important
    # parts": a centered crop of a 16:9 gameplay frame throws away two
    # thirds of the width with no idea which third mattered.
    #
    # Null = behave exactly as before (center/face-driven crop of the whole
    # frame).
    gameplay_rect: Mapped[dict | None] = mapped_column(JSONB)

    # Per-job burn-in styling, set from the console's phone preview:
    #   {"split_facecam_fraction": 0.35, "caption_font_size": 8,
    #    "caption_margin_v": 55, "title_font_size": 13, "title_margin_v": 30}
    # Any subset is valid -- missing keys fall back to the settings.* default,
    # and null (every job predating this) means "all defaults", so nothing
    # about existing behaviour changes.
    style_overrides: Mapped[dict | None] = mapped_column(JSONB)

    # Scene-cut timestamps (seconds) detected once by segmentation.py and
    # persisted here so scoring.py can reuse them instead of re-downloading
    # the raw video and re-running PySceneDetect from scratch -- see
    # app/workers/scoring.py's docstring for the perf reasoning. Null until
    # segmentation completes; also null (rather than an error) if scene
    # detection itself failed there, matching that stage's existing
    # degrade-gracefully behavior.
    scene_cuts_seconds: Mapped[list | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_stream_jobs_user_status", "user_id", "status"),)

    user: Mapped["User"] = relationship(back_populates="stream_jobs")
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="stream_job", uselist=False, passive_deletes=True
    )
    candidate_segments: Mapped[list["CandidateSegment"]] = relationship(
        back_populates="stream_job", passive_deletes=True
    )


class Transcript(Base):
    __tablename__ = "transcripts"
    __table_args__ = (UniqueConstraint("stream_job_id", name="uq_transcript_stream_job"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    stream_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stream_jobs.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String, nullable=False, default="openai")
    language: Mapped[str | None] = mapped_column(String)
    full_text: Mapped[str | None] = mapped_column(Text)
    segments: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    stream_job: Mapped["StreamJob"] = relationship(back_populates="transcript")


class CandidateSegment(Base):
    __tablename__ = "candidate_segments"
    __table_args__ = (
        CheckConstraint("end_seconds > start_seconds", name="ck_candidate_segment_end_after_start"),
        Index("idx_candidate_segments_job", "stream_job_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    stream_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stream_jobs.id", ondelete="CASCADE"), nullable=False
    )
    start_seconds: Mapped[float] = mapped_column(Numeric, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Numeric, nullable=False)
    status: Mapped[str] = mapped_column(candidate_status_enum, nullable=False, default="pending_score")
    features: Mapped[dict | None] = mapped_column(JSONB)
    score: Mapped[float | None] = mapped_column(Numeric)
    score_breakdown: Mapped[dict | None] = mapped_column(JSONB)
    llm_annotation: Mapped[dict | None] = mapped_column(JSONB)
    # "heuristic" (app.core.segmentation_logic's deterministic sliding-window
    # pass -- every candidate before this column existed) or "llm"
    # (app.core.llm_segmentation's opt-in additive proposer). Purely
    # informational -- origin never affects scoring/selection, which treats
    # every pending_score candidate identically regardless of where it came
    # from. See app/workers/segmentation.py for where both origins get
    # inserted into this same table.
    origin: Mapped[str] = mapped_column(String, nullable=False, default="heuristic", server_default="heuristic")
    # Only set when origin == "llm" -- the one-sentence reason the LLM gave
    # for suggesting this specific window (app.core.llm_segmentation_logic
    # .parse_segment_suggestions). None for every heuristic candidate.
    llm_reason: Mapped[str | None] = mapped_column(Text)
    # Optional multi-part ("stitched") clip: a JSON list of [start, end]
    # pairs to cut out of the source and join end-to-end, for a moment that
    # is spread across several non-adjacent stretches of the VOD (a story
    # told in two passes, a callback to something said earlier). Null --
    # every candidate before this existed, and every heuristic candidate
    # today -- means a plain contiguous window described by
    # start_seconds/end_seconds alone.
    #
    # start_seconds/end_seconds are still populated for a multi-part
    # candidate (first part's start, last part's end) so every existing
    # query, sort, overlap check and non-max-suppression comparison keeps
    # working untouched -- they describe the SPAN the clip is drawn from,
    # while `parts` describes what actually plays. Anything that cares
    # about real playing time (scoring, rendering, duration) goes through
    # app.core.rendering_logic.normalize_parts/parts_duration instead.
    parts: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    stream_job: Mapped["StreamJob"] = relationship(back_populates="candidate_segments")
    rendered_clips: Mapped[list["RenderedClip"]] = relationship(
        back_populates="candidate_segment", passive_deletes=True
    )


class RenderedClip(Base):
    __tablename__ = "rendered_clips"
    __table_args__ = (Index("idx_rendered_clips_job_status", "stream_job_id", "status"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    candidate_segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidate_segments.id", ondelete="CASCADE"), nullable=False
    )
    stream_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stream_jobs.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(render_status_enum, nullable=False, default="pending")
    object_key: Mapped[str | None] = mapped_column(Text)
    thumbnail_key: Mapped[str | None] = mapped_column(Text)
    caption_text: Mapped[str | None] = mapped_column(Text)
    format: Mapped[str] = mapped_column(String, nullable=False, default="vertical_9x16")
    duration_seconds: Mapped[float | None] = mapped_column(Numeric)
    requires_review: Mapped[bool] = mapped_column(nullable=False, default=True)
    flag_reasons: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    candidate_segment: Mapped["CandidateSegment"] = relationship(back_populates="rendered_clips")
    stream_job: Mapped["StreamJob"] = relationship()
    review_decisions: Mapped[list["ReviewDecision"]] = relationship(
        back_populates="rendered_clip", passive_deletes=True, order_by="ReviewDecision.decided_at"
    )
    upload_tasks: Mapped[list["UploadTask"]] = relationship(back_populates="rendered_clip", passive_deletes=True)

    # The score itself lives on candidate_segment (the thing that got scored;
    # a RenderedClip is just its rendered output), but the API returns clips,
    # not candidates, and reviewers want to see "why was this picked" right
    # on the clip they're reviewing. These proxy properties let
    # RenderedClipOut (Pydantic, from_attributes=True) pick them up like any
    # other attribute without duplicating the values into a second column.
    @property
    def score(self) -> float | None:
        """0..10 -- see app.core.scoring_logic.score_window's score_out_of_10."""
        return self.candidate_segment.score if self.candidate_segment else None

    @property
    def score_breakdown(self) -> dict | None:
        return self.candidate_segment.score_breakdown if self.candidate_segment else None

    @property
    def latest_rating(self) -> int | None:
        """Most recent reviewer rating (1..5) across this clip's review
        decisions, or None if never rated -- relies on review_decisions
        being ordered by decided_at (see the relationship's order_by above)
        so this is just "last one with a rating set", not a second query.
        A clip can be re-reviewed more than once (e.g. a reviewer changes
        their mind), so this always reflects the latest, not the first.
        """
        for decision in reversed(self.review_decisions):
            if decision.rating is not None:
                return decision.rating
        return None

    @property
    def latest_notes(self) -> str | None:
        """Most recent free-text reviewer comment across this clip's review
        decisions, or None if never commented -- same "last one that
        actually set it" semantics as latest_rating above (a later
        re-review that left the comment box empty doesn't erase the earlier
        comment from this view; it just isn't a newer one).

        Surfaced so a reviewer can see what they already said on reload,
        and -- more importantly -- so app.core.llm_segmentation can feed
        real "this creator approved/rejected this, and here's why in their
        own words" examples back into the segment-suggestion prompt (see
        that module's few-shot example block). That feedback loop is the
        whole reason this is worth exposing rather than being write-only.
        """
        for decision in reversed(self.review_decisions):
            if decision.notes:
                return decision.notes
        return None

    # Same proxy pattern as score/score_breakdown above, reading through to
    # candidate_segment.llm_annotation (see app.core.caption_generation) --
    # caption_text itself is a real column on this table (scoring.py writes
    # a deterministic placeholder there immediately; rendering.py
    # overwrites it with the LLM/heuristic version as part of rendering
    # itself, before status flips to 'rendered'), so it doesn't need a
    # proxy.
    @property
    def caption_title(self) -> str | None:
        """The clickbait-style hook -- burned into the top of the rendered
        video as its own subtitles stage when ENABLE_CLIP_TITLE_OVERLAY is
        on (app.core.rendering_logic.build_title_srt /
        app.workers.rendering._TITLE_STYLE); this property is just the
        annotation's stored copy of that text. Editable afterward via
        PATCH .../caption for record-keeping, but editing it here never
        changes the already-rendered video -- see that endpoint's
        docstring."""
        annotation = self.candidate_segment.llm_annotation if self.candidate_segment else None
        return annotation.get("title") if annotation else None

    @property
    def caption_hashtags(self) -> list[str] | None:
        annotation = self.candidate_segment.llm_annotation if self.candidate_segment else None
        return annotation.get("hashtags") if annotation else None

    @property
    def caption_explanation(self) -> str | None:
        annotation = self.candidate_segment.llm_annotation if self.candidate_segment else None
        return annotation.get("explanation") if annotation else None

    @property
    def caption_source(self) -> str | None:
        """'llm' or 'heuristic_fallback' -- see app.core.caption_generation.
        None until app.workers.caption_generation has run for this clip."""
        annotation = self.candidate_segment.llm_annotation if self.candidate_segment else None
        return annotation.get("source") if annotation else None

    @property
    def caption_model(self) -> str | None:
        """Which model actually produced this clip's title/hashtags/caption
        -- e.g. "gpt-4o-mini" or "llama3.1:8b" when caption_source == "llm",
        None for a heuristic-fallback annotation (no model was called).
        Exists specifically so a reviewer can confirm which
        CAPTION_LLM_PROVIDER actually ran for a given clip rather than just
        assuming the .env setting took effect -- see caption_reason below
        for *why* it didn't, when it didn't."""
        annotation = self.candidate_segment.llm_annotation if self.candidate_segment else None
        return annotation.get("model") if annotation else None

    @property
    def caption_reason(self) -> str | None:
        """Why this clip got the heuristic fallback instead of an LLM-written
        annotation -- "disabled" (ENABLE_LLM_CAPTIONS=false), "no_api_key"
        (CAPTION_LLM_PROVIDER=openai with no key set), or "llm_error: ..."
        (the call/response itself failed, e.g. Ollama unreachable or a
        malformed response) -- see app.core.caption_generation. None when
        caption_source == "llm" (nothing to explain)."""
        annotation = self.candidate_segment.llm_annotation if self.candidate_segment else None
        return annotation.get("reason") if annotation else None

    # Same proxy pattern as score/score_breakdown above, but for
    # CandidateSegment.origin/llm_reason (app.core.llm_segmentation) --
    # NOT to be confused with caption_source/caption_reason above, which
    # are about who wrote this clip's title/hashtags/caption. These two
    # are about who *proposed this clip's time window in the first place*
    # ("heuristic" sliding-window segmentation, or "llm" suggestion).
    @property
    def segment_origin(self) -> str:
        """'heuristic' or 'llm' -- see app.core.llm_segmentation and
        app/workers/segmentation.py. Defaults to 'heuristic' (the column's
        own DB default) if the candidate row is somehow missing."""
        return self.candidate_segment.origin if self.candidate_segment else "heuristic"

    @property
    def segment_parts(self) -> list | None:
        """This clip's stitched parts (a list of [start, end] pairs), or
        None for an ordinary contiguous clip -- see CandidateSegment.parts.
        Surfaced so the dev console can tell a reviewer a clip was joined
        from separate stretches of the VOD, which is exactly the kind of
        edit they should be looking at critically before approving it.
        """
        return self.candidate_segment.parts if self.candidate_segment else None

    @property
    def segment_llm_reason(self) -> str | None:
        """The LLM's one-sentence reason for suggesting this window --
        only set when segment_origin == 'llm'. None for a heuristic
        candidate (nothing to explain; it wasn't picked by a model)."""
        return self.candidate_segment.llm_reason if self.candidate_segment else None


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    __table_args__ = (
        Index("idx_review_decisions_clip_time", "rendered_clip_id", "decided_at"),
        CheckConstraint("rating IS NULL OR (rating >= 1 AND rating <= 5)", name="ck_review_decisions_rating_range"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    rendered_clip_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rendered_clips.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    decision: Mapped[str] = mapped_column(review_decision_enum, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    # 1..5 subjective quality rating -- deliberately separate from `decision`
    # (approved/rejected/skipped), which gates whether a clip is *allowed to
    # post at all*. `rating` is a richer, optional quality signal a reviewer
    # can attach regardless of the decision (e.g. approve a clip you'd only
    # rate 2/5 because it's fine but not great) -- kept nullable since the
    # existing approve/reject flow shouldn't suddenly require a rating.
    # Intended use (see README's "Reviewer feedback & ratings"): once
    # enough of these accumulate across real jobs, they're a manual input
    # for sanity-checking SCORE_WEIGHT_* against what a human actually
    # thought was good -- not an automatic retraining signal. No ground
    # truth/outcome data (e.g. real view counts) feeds this yet; that's a
    # separate, post-MVP data source (see README).
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    rendered_clip: Mapped["RenderedClip"] = relationship(back_populates="review_decisions")


class UploadTask(Base):
    __tablename__ = "upload_tasks"
    __table_args__ = (Index("idx_upload_tasks_account_created", "creator_account_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    rendered_clip_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rendered_clips.id", ondelete="CASCADE"), nullable=False
    )
    creator_account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("creator_accounts.id"), nullable=False)
    platform: Mapped[str] = mapped_column(String, nullable=False, default="tiktok")
    status: Mapped[str] = mapped_column(upload_status_enum, nullable=False, default="queued")
    target_mode: Mapped[str] = mapped_column(
        String, CheckConstraint("target_mode IN ('draft','direct')"), nullable=False, default="draft"
    )
    confidence_score: Mapped[float | None] = mapped_column(Numeric)
    block_reason: Mapped[str | None] = mapped_column(Text)
    # TikTok's own id for this upload attempt (returned by
    # /v2/post/publish/inbox/video/init/), set as soon as init succeeds --
    # NOT the same as external_post_id below. Its job is idempotency: if a
    # retry re-runs app.workers.upload.run() (RQ retry, or a crash between
    # init and status success), a publish_id already being set means "the
    # bytes are already uploaded, don't upload again -- just resume polling
    # status," see that worker's module docstring.
    publish_id: Mapped[str | None] = mapped_column(Text)
    external_post_id: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    rendered_clip: Mapped["RenderedClip"] = relationship(back_populates="upload_tasks")
    audit_log: Mapped[list["UploadAuditLog"]] = relationship(back_populates="upload_task", passive_deletes=True)


class UploadAuditLog(Base):
    __tablename__ = "upload_audit_log"
    __table_args__ = (Index("idx_upload_audit_log_task", "upload_task_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    upload_task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("upload_tasks.id", ondelete="CASCADE"), nullable=False
    )
    event: Mapped[str] = mapped_column(
        String, CheckConstraint("event IN ('attempt','success','failure','blocked')"), nullable=False
    )
    detail: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    upload_task: Mapped["UploadTask"] = relationship(back_populates="audit_log")
