"""Pydantic request/response contracts -- mirrors architecture doc §6."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class StreamJobOut(BaseModel):
    id: uuid.UUID
    status: str
    source_type: str
    duration_seconds: float | None
    max_clips: int
    stt_model_size: str | None
    min_clip_seconds: float | None
    max_clip_seconds: float | None
    min_score_threshold: float | None
    # 'auto' (None here means "auto") | 'single_crop' | 'split_reaction' |
    # 'fit_frame' -- see StreamJob.camera_layout_mode and
    # app/workers/rendering.py.
    camera_layout_mode: str | None
    # 'left' | 'center' | 'right' | None -- which side of the frame a
    # cropping layout should favour. See StreamJob.crop_bias.
    crop_bias: str | None
    # {"x","y","w","h"} as fractions of the source frame, or None. The
    # facecam box marked by hand in the dev console -- see
    # StreamJob.facecam_rect.
    facecam_rect: dict | None
    retry_count: int
    last_error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class FacecamRect(BaseModel):
    """A box on the source frame, normalized to 0..1.

    Normalized rather than pixels so the value stays correct regardless of
    the source resolution, and so the dev console can send what it measured
    on a scaled-down preview image without knowing the real frame size.
    """

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class FacecamRectRequest(BaseModel):
    """Body for PUT /stream-jobs/{id}/facecam-rect.

    `rect: null` clears the mark and returns the job to automatic face
    detection.
    """

    rect: FacecamRect | None = None


class VodImportRequest(BaseModel):
    source_type: Literal["vod_import"] = "vod_import"
    source_url: str = Field(min_length=1)


class RenderedClipOut(BaseModel):
    id: uuid.UUID
    stream_job_id: uuid.UUID
    status: str
    object_key: str | None
    thumbnail_key: str | None
    caption_text: str | None
    format: str
    duration_seconds: float | None
    requires_review: bool
    flag_reasons: list[str]
    # 0..10 -- see app.core.scoring_logic.score_window's score_out_of_10.
    # None until scoring has run for this clip's candidate (should be rare in
    # practice: a RenderedClip row is only created once scoring selects its
    # candidate, so this is really just defensive typing).
    score: float | None
    score_breakdown: dict | None
    # LLM-assisted (or heuristic-fallback) enrichment -- see
    # app.core.caption_generation, called synchronously as part of
    # rendering. Populated together with the rest of this row by the time
    # status flips to 'rendered'; all None only for a clip still mid-render
    # or one rendered before this field existed.
    caption_title: str | None
    caption_hashtags: list[str] | None
    caption_explanation: str | None
    caption_source: str | None
    # Which model produced it ("llm" only, e.g. "gpt-4o-mini"/"llama3.1:8b")
    # and, for a heuristic-fallback annotation, why the LLM path didn't run
    # -- lets a reviewer/the dev console confirm which CAPTION_LLM_PROVIDER
    # actually fired for this clip instead of just trusting the .env
    # setting. See RenderedClip.caption_model/.caption_reason.
    caption_model: str | None
    caption_reason: str | None
    # Which system proposed this clip's time window in the first place --
    # 'heuristic' (app.core.segmentation_logic's deterministic sliding-
    # window pass, every candidate before this field existed) or 'llm'
    # (app.core.llm_segmentation's opt-in additive proposer). NOT the same
    # thing as caption_source/caption_reason above (those are about who
    # wrote the title/hashtags/caption for an already-selected clip).
    segment_origin: str
    segment_llm_reason: str | None
    # A list of [start, end] pairs when this clip was stitched together
    # from separate stretches of the VOD, else None -- see
    # CandidateSegment.parts.
    segment_parts: list | None
    # Most recent 1..5 reviewer rating (see ReviewRequest.rating), or None
    # if this clip has never been rated -- lets the dev console show the
    # existing rating on reload instead of only right after submitting it.
    latest_rating: int | None
    # Most recent free-text reviewer comment (see ReviewRequest.notes), or
    # None if never commented. Shown in the dev console so a reviewer sees
    # what they already said, and read back by the segmentation worker to
    # build few-shot "what this creator likes, in their own words" examples
    # for the LLM segment proposer.
    latest_notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected", "skipped"]
    notes: str | None = None
    # Optional 1..5 subjective quality rating, independent of `decision` --
    # see app.db.models.ReviewDecision.rating's docstring for why these are
    # kept separate. Field's ge/le bounds reject anything outside 1..5 with
    # a normal 422 validation error, same as any other request field.
    rating: int | None = Field(None, ge=1, le=5)


class ClipCaptionUpdate(BaseModel):
    """Manual edit of a clip's generated title/hashtags/caption -- see
    app.api.routers.clips.update_clip_caption. Partial update: omit a field
    (leave it None) to leave that part unchanged. At least one of the three
    must be provided, or there's nothing to update. Editing `title` here
    only changes the stored record -- it does NOT re-render the video, so
    it won't change the already-burned-in on-screen banner (see
    RenderedClip.caption_title's docstring); that's a known limitation, not
    a bug.
    """

    title: str | None = None
    hashtags: list[str] | None = None
    caption: str | None = None


class ReviewDecisionOut(BaseModel):
    id: uuid.UUID
    rendered_clip_id: uuid.UUID
    decision: str
    notes: str | None
    rating: int | None
    decided_at: datetime

    model_config = {"from_attributes": True}


class UploadRequest(BaseModel):
    creator_account_id: uuid.UUID
    platform: Literal["tiktok"] = "tiktok"
    target_mode: Literal["draft"] = "draft"  # 'direct' rejected at MVP -- see architecture doc §6


class UploadTaskOut(BaseModel):
    id: uuid.UUID
    rendered_clip_id: uuid.UUID
    platform: str
    status: str
    target_mode: str
    block_reason: str | None
    publish_id: str | None
    external_post_id: str | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CreatorAccountOut(BaseModel):
    id: uuid.UUID
    platform: str
    external_account_id: str
    daily_upload_cap: int
    created_at: datetime

    model_config = {"from_attributes": True}


class TikTokOAuthStartOut(BaseModel):
    """Response of GET .../tiktok/oauth/start -- the caller (dev console)
    redirects the browser to `authorize_url` itself; this endpoint doesn't
    redirect server-side since it's a normal authenticated JSON call, not
    the browser-facing hop (see that route's docstring)."""

    authorize_url: str


class CreatorAccountCreate(BaseModel):
    """Manual account registration -- see app.api.routers.creator_accounts.
    This is NOT the OAuth flow (that's the still-stubbed
    /tiktok/oauth/callback, which needs a registered TikTok developer app):
    it's the interim "I already have a token, register it" path a creator
    (or, today, a developer testing the pipeline) uses until real OAuth
    exists. access_token/refresh_token are encrypted before being stored
    (app.core.crypto) and are never returned by the API afterward.
    """

    platform: Literal["tiktok", "youtube", "instagram"]
    external_account_id: str = Field(min_length=1)
    access_token: str = Field(min_length=1)
    refresh_token: str | None = None
    daily_upload_cap: int | None = None
