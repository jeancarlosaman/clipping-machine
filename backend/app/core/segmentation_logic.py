"""Pure logic for building candidate clip windows from a transcript + scene
cuts -- separated from app/workers/segmentation.py so it's unit-testable
without a real video file or DB session.

Combines two signals per the architecture doc's ranking guidance:
  - visual scene cuts (PySceneDetect's content-aware detector, via
    detect_scene_cuts())
  - transcript-derived speech/silence structure (merge_into_speech_blocks())

A continuous speech block becomes one candidate window if its length is
within [min_len, max_len]. A too-long block produces *several overlapping*
max_len-ish candidate windows via a sliding pass across it
(_sliding_subwindows), each snapping its boundaries to a nearby scene cut
when one exists -- falling back to a hard cut otherwise (the same "snap to
a nearby marker, else hard cut" shape as app/core/stt/chunking.py's
silence-boundary chunking, applied to a different signal). This
intentionally over-generates candidates for scoring to choose among, rather
than committing to one rigid partition here -- see _sliding_subwindows'
docstring for why, and app.core.scoring_logic.select_top_non_overlapping
for the non-max-suppression step downstream that turns overlapping
candidates back into a diverse final selection.

Known simplification: a standalone block shorter than min_len is dropped,
not merged across its neighboring silence into a longer candidate -- see
build_candidate_windows()'s docstring for when to revisit that.
"""
from __future__ import annotations


def detect_scene_cuts(video_path: str, frame_skip: int = 0) -> list[float]:
    """Interior scene-cut timestamps (seconds) via PySceneDetect.

    Local import: scenedetect pulls in OpenCV, only worth loading when a
    video is actually being segmented.

    Per-frame analysis already runs on a downscaled copy of each decoded
    frame -- SceneManager auto-downscales to ~256-384px effective width by
    default, so the pixel-diff math itself is already cheap. The remaining
    cost for a long VOD is decoding every single source frame in the first
    place (`frame_skip=0`, the PySceneDetect default). `frame_skip` (see
    settings.scene_detect_frame_skip) analyzes only every (frame_skip+1)th
    frame -- PySceneDetect's own decode loop calls the cheap `grab()`
    (advance the demuxer) instead of `grab()+retrieve()` (full decode) for
    every frame it skips, so this is a real decode-cost reduction, not just
    doing the same work and throwing results away. Trade-off: a cut is
    detected up to `frame_skip` frames later than its true boundary --
    irrelevant here since these timestamps only ever get *snapped to* by
    segmentation's window-boundary logic and contribute one modest-weight
    scoring signal (SCORE_WEIGHT_SCENE_CHANGES), never used for
    frame-accurate anything.
    """
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.detect_scenes(video, frame_skip=frame_skip)
    scene_list = manager.get_scene_list()
    # Interior cuts only: the start of every scene after the first IS a cut
    # point. The very first scene's start (0.0) and the last scene's end
    # (the video's duration) are the clip's own boundaries, not cuts.
    return [start.get_seconds() for start, _end in scene_list[1:]]


def merge_into_speech_blocks(
    segments: list[dict], max_gap_seconds: float = 1.2
) -> list[tuple[float, float]]:
    """Merge transcript segments into continuous speech blocks.

    Splits into a new block wherever the gap to the next segment exceeds
    max_gap_seconds -- a longer pause is treated as a natural break, a
    shorter one as just a breath/beat within the same block.
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s["start"])
    blocks: list[list[float]] = [[ordered[0]["start"], ordered[0]["end"]]]
    for seg in ordered[1:]:
        gap = seg["start"] - blocks[-1][1]
        if gap <= max_gap_seconds:
            blocks[-1][1] = max(blocks[-1][1], seg["end"])
        else:
            blocks.append([seg["start"], seg["end"]])
    return [(b[0], b[1]) for b in blocks]


def _snap(cursor: float, target: float, limit: float, scene_cuts: list[float], search_window: float) -> float:
    """The nearest scene cut to `target` within search_window, if any exist
    strictly between `cursor` and `limit`; otherwise `target` itself (a hard
    cut). Shared by _sliding_subwindows below."""
    candidates = [c for c in scene_cuts if cursor + 1.0 < c < limit and abs(c - target) <= search_window]
    return min(candidates, key=lambda c: abs(c - target)) if candidates else target


# Safety valve on _sliding_subwindows -- bounds how many candidate rows one
# unbroken block of speech can generate (e.g. a long uninterrupted monologue
# with no pauses long enough to be its own block boundary). Sized generously
# (a 60-candidate block is already a ~30+ minute monologue at the default
# step), not expected to bind in normal use -- see that function's docstring.
_MAX_SUBWINDOWS_PER_BLOCK = 60


def _sliding_subwindows(
    start: float, end: float, scene_cuts: list[float], min_len: float, max_len: float, search_window: float
) -> list[tuple[float, float]]:
    """A long speech block [start, end) -> several *overlapping* max_len-ish
    candidate windows spanning it, instead of one rigid partition.

    The previous approach (still visible in git history as
    `_split_long_block`) cut a long block into fixed max_len-ish pieces
    starting from `start` -- so whatever moment happened to fall at
    multiples of max_len became a clip boundary, whether or not that was
    actually the best place to cut. A 6-minute rant with one great 40s
    stretch in the middle would get split blind to where that stretch was.

    Sliding a max_len-sized window across the block in overlapping steps and
    letting app.workers.scoring's scoring + non-max-suppression pick the
    best-scoring, least-overlapping subset lets the *content* determine
    which sub-window wins, not its position relative to the block's start.
    Each window's boundaries still snap to a nearby scene cut when one
    exists, same "prefer a real visual cut, else hard cut" behavior as
    before.

    Step size is max_len/3 (each window overlaps its neighbors by ~2/3) --
    tight enough to actually find a peak inside the block, loose enough that
    a block doesn't produce an unreasonable number of near-duplicate
    candidates for scoring to wade through.
    """
    step = max(max_len / 3.0, min_len)  # never step finer than the shortest allowed clip
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor + max_len < end and len(windows) < _MAX_SUBWINDOWS_PER_BLOCK:
        window_end = _snap(cursor, cursor + max_len, end, scene_cuts, search_window)
        windows.append((cursor, window_end))
        cursor += step
    # Always include the block's own tail end so the last stretch (which the
    # loop above may step past without landing on exactly) is covered too.
    tail_start = max(start, end - max_len)
    if not windows or abs(windows[-1][1] - end) > 1.0:
        windows.append((tail_start, end))
    return windows


def build_candidate_windows(
    transcript_segments: list[dict],
    scene_cuts: list[float],
    *,
    min_len: float = 15.0,
    max_len: float = 90.0,
    max_gap_seconds: float = 1.2,
    scene_search_window: float = 5.0,
) -> list[tuple[float, float]]:
    """Transcript segments + scene cuts -> candidate clip windows (start, end).

    If real VODs turn out heavily fragmented (many short blocks getting
    dropped for being under min_len), that's the thing to revisit first --
    e.g. bridging blocks across a short silence -- not this function's
    overall structure.
    """
    windows: list[tuple[float, float]] = []
    for start, end in merge_into_speech_blocks(transcript_segments, max_gap_seconds):
        duration = end - start
        if duration > max_len:
            windows.extend(_sliding_subwindows(start, end, scene_cuts, min_len, max_len, scene_search_window))
        elif duration >= min_len:
            windows.append((start, end))
        # else: standalone block shorter than min_len -- dropped, see docstring above.
    return windows
