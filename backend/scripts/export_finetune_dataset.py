"""Export your own approved, highly-rated clips as an OpenAI supervised
fine-tuning dataset (JSONL) -- see README's "Training on your own approved
clips" section for the full reasoning and how to actually kick off a
fine-tune job with the file this produces.

This is a manual, one-off/occasional script, NOT part of the running
pipeline -- run it yourself once you've accumulated enough real reviewer
ratings (see README for OpenAI's own minimum/recommended example counts;
running this with too little data just produces a small file, it won't
stop you, but a fine-tune job on too few examples is not worth the money).

Each output line reproduces the EXACT prompt app.core.caption_generation
sends in production (same transcript-window text, same ranking-note hint,
same fallback-caption text) paired with the title/hashtags/caption/
explanation you actually approved for that clip -- so a model fine-tuned
on this file is learning to do the *exact same call* your pipeline already
makes, just hopefully better/more your style, not a different task.

Usage:
    PYTHONPATH=. python scripts/export_finetune_dataset.py --min-rating 4 --out finetune_dataset.jsonl

Only clips that are BOTH "approved" (you actually posted or would post it)
AND rated at or above --min-rating are included -- a clip you approved but
only rated 2/5 ("fine, not great") is deliberately excluded, since the
whole point is training toward your *best* work, not just your safe/postable
average.
"""
from __future__ import annotations

import argparse
import json

from app.core.caption_logic import build_caption_prompt, heuristic_explanation
from app.core.scoring_logic import window_text
from app.db.models import CandidateSegment, RenderedClip, ReviewDecision, Transcript
from app.db.session import SessionLocal


def _latest_rating_and_decision(db, clip_id) -> tuple[int | None, str | None]:
    """Mirrors RenderedClip.latest_rating's "most recent decision that had
    a rating" semantics (see app.db.models), but also returns that
    decision's own `decision` value -- we need both together here (a
    clip's *latest* rating might come from a review whose decision was
    later superseded by a fresh, unrated re-review), not two separate
    "latest of each" lookups that could disagree about which review they
    came from.
    """
    decisions = (
        db.query(ReviewDecision)
        .filter_by(rendered_clip_id=clip_id)
        .order_by(ReviewDecision.decided_at.desc())
        .all()
    )
    for decision in decisions:
        if decision.rating is not None:
            return decision.rating, decision.decision
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-rating", type=int, default=4, help="Minimum reviewer rating (1-5) to include. Default 4.")
    parser.add_argument("--out", default="finetune_dataset.jsonl", help="Output JSONL path. Default finetune_dataset.jsonl.")
    args = parser.parse_args()

    if not (1 <= args.min_rating <= 5):
        raise SystemExit("--min-rating must be between 1 and 5")

    db = SessionLocal()
    examples = []
    skipped_no_transcript = 0
    skipped_no_annotation = 0
    try:
        clips = (
            db.query(RenderedClip)
            .filter(RenderedClip.status == "rendered")
            .join(CandidateSegment, RenderedClip.candidate_segment_id == CandidateSegment.id)
            .all()
        )
        for clip in clips:
            rating, decision = _latest_rating_and_decision(db, clip.id)
            if rating is None or rating < args.min_rating or decision != "approved":
                continue

            candidate = clip.candidate_segment
            if candidate is None or candidate.score_breakdown is None:
                skipped_no_annotation += 1
                continue

            transcript = db.query(Transcript).filter_by(stream_job_id=clip.stream_job_id).one_or_none()
            if transcript is None or not transcript.segments:
                skipped_no_transcript += 1
                continue

            title = clip.caption_title
            hashtags = clip.caption_hashtags
            if not title or not hashtags:
                # A clip that never got an LLM/heuristic title+hashtags at
                # all (shouldn't happen for a 'rendered' clip, but a
                # dataset export is exactly the wrong place to guess) --
                # skip rather than emit a training example with holes in it.
                skipped_no_annotation += 1
                continue

            clip_text = window_text(transcript.segments, float(candidate.start_seconds), float(candidate.end_seconds))
            hint = heuristic_explanation(candidate.score_breakdown)
            prompt = build_caption_prompt(clip_text, hint, clip.caption_text or "")

            assistant_reply = json.dumps(
                {
                    "title": title,
                    # Strip the leading '#' back off -- the prompt's own
                    # instructions ask for hashtags WITHOUT it (see
                    # app.core.caption_logic.build_caption_prompt), and a
                    # fine-tuned model should learn to reply in the exact
                    # format it'll actually be called with.
                    "hashtags": [tag.lstrip("#") for tag in hashtags],
                    "caption": clip.caption_text or "",
                    "explanation": clip.caption_explanation or "",
                }
            )

            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_reply},
                    ]
                }
            )
    finally:
        db.close()

    with open(args.out, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {len(examples)} training examples to {args.out}")
    if skipped_no_transcript:
        print(f"Skipped {skipped_no_transcript} clip(s) with no transcript found")
    if skipped_no_annotation:
        print(f"Skipped {skipped_no_annotation} clip(s) missing a score breakdown or title/hashtags")
    if len(examples) < 10:
        print(
            "\nHeads up: OpenAI's fine-tuning API requires at least 10 examples, and recommends "
            "50-100+ before you'll actually see a quality improvement -- see README's "
            "'Training on your own approved clips' section. Keep rating clips 4-5 stars as you "
            "review them, then re-run this script later."
        )


if __name__ == "__main__":
    main()
