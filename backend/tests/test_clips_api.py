"""Tests for PATCH /api/v1/clips/{clip_id}/caption -- manual hashtag/caption
editing, see app.api.routers.clips.update_clip_caption. Same fixture/pattern
as test_stream_jobs_api.py / test_creator_accounts_api.py.
"""
import uuid

from app.core.auth import create_access_token
from app.db.models import CandidateSegment, RenderedClip, StreamJob, User


def _make_clip(db_session, user, *, llm_annotation=None, caption_text=None):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/caption-test.mp4", status="scored")
    db_session.add(job)
    db_session.flush()

    candidate = CandidateSegment(
        stream_job_id=job.id, start_seconds=0, end_seconds=5, status="selected", llm_annotation=llm_annotation
    )
    db_session.add(candidate)
    db_session.flush()

    clip = RenderedClip(
        candidate_segment_id=candidate.id, stream_job_id=job.id, status="rendered", caption_text=caption_text
    )
    db_session.add(clip)
    db_session.commit()
    db_session.refresh(clip)
    return clip


_EXISTING_ANNOTATION = {
    "title": "He Did NOT See That Coming",
    "hashtags": ["#gaming", "#clip", "#viral"],
    "caption": "original caption",
    "explanation": "high speech intensity",
    "source": "llm",
    "model": "gpt-4o-mini",
}


def test_clip_exposes_caption_model_for_llm_sourced_annotation(client, auth_headers, db_session, user):
    # Lets a reviewer confirm which CAPTION_LLM_PROVIDER actually produced a
    # given clip's title (e.g. "llama3.1:8b" vs "gpt-4o-mini") without
    # digging into the DB -- see RenderedClip.caption_model's docstring.
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION))

    resp = client.patch(f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"hashtags": ["#x"]})

    assert resp.status_code == 200
    body = resp.json()
    # untouched by the hashtags-only edit -- still reflects the original
    # LLM-generated annotation's model, not the manual_edit that just landed
    # on caption_source for the hashtags field itself.
    assert body["caption_model"] == "gpt-4o-mini"
    assert body["caption_reason"] is None  # nothing to explain -- the LLM call succeeded


def test_clip_exposes_caption_reason_for_heuristic_fallback(client, auth_headers, db_session, user):
    fallback_annotation = {
        "title": "You Have To See This Clip",
        "hashtags": ["#clip", "#streamer", "#twitchclips"],
        "caption": "existing caption",
        "explanation": "Selected by the ranking algorithm; no single signal stood out strongly.",
        "source": "heuristic_fallback",
        "reason": "llm_error: connection refused",
        "model": None,
    }
    clip = _make_clip(db_session, user, llm_annotation=fallback_annotation, caption_text="existing caption")

    resp = client.patch(f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"hashtags": ["#x"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["caption_model"] is None  # heuristic fallback never calls a model
    assert body["caption_reason"] == "llm_error: connection refused"


def test_update_clip_hashtags_only(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION), caption_text="original caption")

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"hashtags": ["#new", "#tags"]}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["caption_hashtags"] == ["#new", "#tags"]
    assert body["caption_source"] == "manual_edit"
    # caption text/title untouched since only hashtags were provided
    assert body["caption_explanation"] == "high speech intensity"
    assert body["caption_title"] == "He Did NOT See That Coming"


def test_update_clip_title_only(client, auth_headers, db_session, user):
    # Note: this is the burned-in on-screen title's *stored record* -- it
    # never re-renders the video, see ClipCaptionUpdate's docstring. This
    # test only covers the API/DB side of that.
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION), caption_text="original caption")

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"title": "Wait For It..."}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["caption_title"] == "Wait For It..."
    assert body["caption_source"] == "manual_edit"
    # hashtags/caption untouched since only title was provided
    assert body["caption_hashtags"] == ["#gaming", "#clip", "#viral"]

    db_session.expunge_all()
    refreshed = db_session.get(RenderedClip, clip.id)
    assert refreshed.caption_text == "original caption"  # unchanged -- title edit doesn't touch caption_text


def test_update_clip_caption_only(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION), caption_text="original caption")

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"caption": "a much better caption"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["caption_source"] == "manual_edit"
    # hashtags untouched since only caption was provided
    assert body["caption_hashtags"] == ["#gaming", "#clip", "#viral"]

    db_session.expunge_all()
    refreshed = db_session.get(RenderedClip, clip.id)
    assert refreshed.caption_text == "a much better caption"


def test_update_clip_caption_and_hashtags_together(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION), caption_text="original caption")

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption",
        headers=auth_headers,
        json={"hashtags": ["#one", "#two"], "caption": "updated"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["caption_hashtags"] == ["#one", "#two"]
    assert body["caption_source"] == "manual_edit"

    db_session.expunge_all()
    refreshed = db_session.get(RenderedClip, clip.id)
    assert refreshed.caption_text == "updated"


def test_update_clip_caption_works_from_no_prior_annotation(client, auth_headers, db_session, user):
    # A clip whose LLM caption-generation step never ran (or is still
    # pending) -- llm_annotation is None going in, the manual edit should
    # still succeed and become the source of truth.
    clip = _make_clip(db_session, user, llm_annotation=None, caption_text=None)

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"hashtags": ["#first", "#edit"]}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["caption_hashtags"] == ["#first", "#edit"]
    assert body["caption_source"] == "manual_edit"


def test_update_clip_caption_rejects_empty_body(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION))

    resp = client.patch(f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "no_fields_to_update"


def test_update_clip_caption_rejects_all_unusable_hashtags(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION))

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"hashtags": ["   ", ""]}
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_hashtags"


def test_update_clip_caption_rejects_empty_caption(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION))

    resp = client.patch(f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"caption": "   "})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_caption"


def test_update_clip_caption_rejects_empty_title(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION))

    resp = client.patch(f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"title": "   "})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_title"


def test_update_clip_caption_clamps_long_title(client, auth_headers, db_session, user):
    clip = _make_clip(db_session, user, llm_annotation=dict(_EXISTING_ANNOTATION))

    resp = client.patch(
        f"/api/v1/clips/{clip.id}/caption", headers=auth_headers, json={"title": "x" * 200}
    )

    assert resp.status_code == 200
    assert len(resp.json()["caption_title"]) == 70


def test_update_clip_caption_enforces_ownership(client, db_session, user):
    other_user = User(email=f"{uuid.uuid4()}@example.com", password_hash="x")
    db_session.add(other_user)
    db_session.commit()
    db_session.refresh(other_user)

    clip = _make_clip(db_session, other_user, llm_annotation=dict(_EXISTING_ANNOTATION))

    my_headers = {"Authorization": f"Bearer {create_access_token(str(user.id))}"}
    resp = client.patch(f"/api/v1/clips/{clip.id}/caption", headers=my_headers, json={"caption": "sneaky"})

    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_update_nonexistent_clip_caption_404s(client, auth_headers):
    resp = client.patch(
        f"/api/v1/clips/{uuid.uuid4()}/caption", headers=auth_headers, json={"caption": "x"}
    )
    assert resp.status_code == 404
