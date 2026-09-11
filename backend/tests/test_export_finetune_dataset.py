"""Tests for scripts/export_finetune_dataset.py -- run in-process (not as a
subprocess) against the real test DB, same as every other worker/script
test in this suite (see tests/conftest.py's module docstring: this suite
talks to real Postgres, it does not mock the database).
"""
import json
import sys

from app.core.caption_logic import build_caption_prompt, heuristic_explanation
from app.core.scoring_logic import window_text
from app.db.models import CandidateSegment, RenderedClip, ReviewDecision, StreamJob, Transcript
from scripts import export_finetune_dataset

SEGMENTS = [{"start": 0.0, "end": 3.0, "text": "You will not believe what just happened."}]
SCORE_BREAKDOWN = {"composite": 5.0, "raw": {}, "normalized": {}, "weights": {}, "contributions": {}, "score_out_of_10": 8.0}


def _seed_rated_clip(db_session, user, *, rating, decision, with_title=True):
    job = StreamJob(user_id=user.id, source_type="upload", raw_object_key="raw/x.mp4", status="ready_for_review")
    db_session.add(job)
    db_session.flush()

    transcript = Transcript(stream_job_id=job.id, segments=SEGMENTS, full_text=SEGMENTS[0]["text"])
    db_session.add(transcript)

    candidate = CandidateSegment(
        stream_job_id=job.id, start_seconds=0.0, end_seconds=3.0, status="selected",
        score_breakdown=SCORE_BREAKDOWN,
        llm_annotation={
            "source": "llm", "title": "You Will NOT Believe This",
            "hashtags": ["#omg", "#clip"], "caption": "wait for it", "explanation": "big reaction moment",
            "model": "gpt-4o-mini",
        } if with_title else None,
    )
    db_session.add(candidate)
    db_session.flush()

    clip = RenderedClip(
        candidate_segment_id=candidate.id, stream_job_id=job.id, status="rendered", caption_text="wait for it",
    )
    db_session.add(clip)
    db_session.flush()

    db_session.add(ReviewDecision(rendered_clip_id=clip.id, user_id=user.id, decision=decision, rating=rating))
    db_session.commit()
    return clip


def test_exports_approved_highly_rated_clip(db_session, user, tmp_path, monkeypatch, capsys):
    _seed_rated_clip(db_session, user, rating=5, decision="approved")
    out_path = tmp_path / "dataset.jsonl"

    monkeypatch.setattr(sys, "argv", ["export_finetune_dataset.py", "--out", str(out_path)])
    export_finetune_dataset.main()

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    example = json.loads(lines[0])
    assert example["messages"][0]["role"] == "user"
    assert example["messages"][1]["role"] == "assistant"

    expected_prompt = build_caption_prompt(
        window_text(SEGMENTS, 0.0, 3.0), heuristic_explanation(SCORE_BREAKDOWN), "wait for it"
    )
    assert example["messages"][0]["content"] == expected_prompt

    assistant = json.loads(example["messages"][1]["content"])
    assert assistant["title"] == "You Will NOT Believe This"
    assert assistant["hashtags"] == ["omg", "clip"]  # leading '#' stripped back off, per the prompt's own contract
    assert assistant["caption"] == "wait for it"


def test_excludes_clip_below_min_rating(db_session, user, tmp_path, monkeypatch):
    _seed_rated_clip(db_session, user, rating=3, decision="approved")
    out_path = tmp_path / "dataset.jsonl"

    monkeypatch.setattr(sys, "argv", ["export_finetune_dataset.py", "--min-rating", "4", "--out", str(out_path)])
    export_finetune_dataset.main()

    assert out_path.read_text().strip() == ""


def test_excludes_rejected_clip_even_if_highly_rated(db_session, user, tmp_path, monkeypatch):
    # A clip you rated 5/5 but ultimately rejected (e.g. legal/sponsor
    # reasons unrelated to quality) shouldn't teach the model "post this."
    _seed_rated_clip(db_session, user, rating=5, decision="rejected")
    out_path = tmp_path / "dataset.jsonl"

    monkeypatch.setattr(sys, "argv", ["export_finetune_dataset.py", "--out", str(out_path)])
    export_finetune_dataset.main()

    assert out_path.read_text().strip() == ""


def test_excludes_clip_missing_title_or_hashtags(db_session, user, tmp_path, monkeypatch):
    _seed_rated_clip(db_session, user, rating=5, decision="approved", with_title=False)
    out_path = tmp_path / "dataset.jsonl"

    monkeypatch.setattr(sys, "argv", ["export_finetune_dataset.py", "--out", str(out_path)])
    export_finetune_dataset.main()

    assert out_path.read_text().strip() == ""


def test_rejects_out_of_range_min_rating(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_finetune_dataset.py", "--min-rating", "6"])
    try:
        export_finetune_dataset.main()
        assert False, "expected SystemExit"
    except SystemExit:
        pass
