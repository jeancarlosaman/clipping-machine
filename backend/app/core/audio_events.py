"""Open-source audio-event detection (laughter/cheering/crowd reaction) as
an additional deterministic scoring signal -- see app.core.scoring_logic's
module docstring for how this plugs into scoring (`audio_events` param on
score_window). Uses PANNs (Pretrained Audio Neural Networks,
https://github.com/qiuqiangkong/audioset_tagging_cnn via the `panns-inference`
package), an open-source classifier trained on Google's AudioSet taxonomy --
a fixed pretrained model's output, not a generative/LLM signal, same
"deterministic feature" principle as every other scoring feature.

Opt-in (ENABLE_AUDIO_EVENT_SCORING, default False): this is a genuinely new
class of dependency for this project -- torch + a ~300MB pretrained
checkpoint, where previously the heaviest dependency was faster-whisper's
CTranslate2 runtime (no torch/tensorflow at all). Kept fully optional,
lazy-imported, and lazy-loaded (nothing here is touched unless the flag is
on and a caller actually constructs AudioEventDetector) so a normal run
with the flag off never even imports torch.

Two real Windows-breaking bugs found and fixed here (verified via a real,
non-mocked HTTP round trip in development -- see this project's test
history), not just theoretical: panns_inference's OWN package code shells
out to `wget` via os.system() in two places (its config.py, fetching the
AudioSet label list; its inference.py, fetching the model checkpoint from
Zenodo) to download its data files on first use. `wget` is not a built-in
Windows binary -- there's no bundled wget.exe -- so on a default Windows
install that os.system() call fails silently (os.system() doesn't raise on
a failed command, it just returns a non-zero status nothing checks), and
the very next line in panns_inference's own code (opening the file it just
tried to fetch) crashes with FileNotFoundError. _ensure_panns_data() below
pre-fetches both files ourselves via `requests` (already a project
dependency, identical code path on every platform, no external binary
required) before panns_inference is ever imported -- its own
`if not os.path.isfile(...)` / `if not os.path.exists(...)` guards then
find the files already present and skip the wget calls entirely.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

from app.workers.common import logger, run_subprocess

_PANNS_DATA_DIR = Path.home() / "panns_data"
_LABELS_CSV_PATH = _PANNS_DATA_DIR / "class_labels_indices.csv"
_LABELS_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
_CHECKPOINT_PATH = _PANNS_DATA_DIR / "Cnn14_mAP=0.431.pth"
_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
# panns_inference's own inference.py treats a checkpoint file under 3e8
# bytes as "not really downloaded, re-fetch it" (a corrupt/truncated
# previous attempt) -- matched here so our pre-fetch and its own guard
# agree on what "already have it" means.
_CHECKPOINT_MIN_BYTES = 300_000_000

SAMPLE_RATE = 32000  # PANNs' fixed input rate -- see panns_inference.config.sample_rate

# AudioSet class names we care about, grouped into two composite signals,
# looked up BY NAME against whatever class_labels_indices.csv actually
# loads -- not hardcoded numeric indices. AudioSet's class ordering is a
# fixed, unchanging taxonomy, but looking up by name is self-verifying
# against the real file's contents rather than us needing to trust a
# hand-copied index number is still correct.
LAUGHTER_LABELS: tuple[str, ...] = (
    "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
)
CROWD_REACTION_LABELS: tuple[str, ...] = (
    "Cheering", "Applause", "Crowd", "Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Children shouting",
)


def _download(path: Path, url: str, *, min_bytes: int) -> None:
    if path.is_file() and path.stat().st_size >= min_bytes:
        return
    logger.info("audio_events.downloading", url=url, dest=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    logger.info("audio_events.downloaded", dest=str(path), bytes=len(resp.content))


def _ensure_panns_data() -> None:
    _download(_LABELS_CSV_PATH, _LABELS_CSV_URL, min_bytes=1_000)
    _download(_CHECKPOINT_PATH, _CHECKPOINT_URL, min_bytes=_CHECKPOINT_MIN_BYTES)


def extract_window_waveform(audio_path: str, start: float, end: float) -> np.ndarray:
    """ffmpeg-extract [start, end] from a source audio file as mono float32
    PCM at PANNs' required 32kHz sample rate, returned as a 1-D numpy array
    in [-1, 1]. Shells out to ffmpeg (already a hard project dependency)
    rather than adding a second audio-decoding library on top of PANNs'
    own torch dependency.
    """
    import tempfile
    import wave

    duration = max(end - start, 0.05)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        run_subprocess([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-t", str(duration), "-i", audio_path,
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
            tmp_path,
        ])
        with wave.open(tmp_path, "rb") as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class AudioEventDetector:
    """Loads the PANNs model once (a real, multi-second cost) -- construct
    a single instance per stream_job and reuse it across every candidate
    window, never per-window. Not thread-safe to construct concurrently
    (torch model load), but this project's workers process one job at a
    time per process anyway.
    """

    def __init__(self) -> None:
        _ensure_panns_data()
        import panns_inference  # local import: keep torch optional until this class is actually constructed

        self._tagging = panns_inference.AudioTagging(checkpoint_path=str(_CHECKPOINT_PATH), device="cpu")
        label_to_index = {name: i for i, name in enumerate(panns_inference.labels)}

        self._laughter_indices = [label_to_index[label] for label in LAUGHTER_LABELS if label in label_to_index]
        self._crowd_indices = [
            label_to_index[label] for label in CROWD_REACTION_LABELS if label in label_to_index
        ]
        missing = [
            label for label in (*LAUGHTER_LABELS, *CROWD_REACTION_LABELS) if label not in label_to_index
        ]
        if missing:
            # Not fatal -- degrade to "that composite signal is always 0"
            # rather than crashing a whole scoring job over a label-name
            # mismatch (e.g. a differently-shaped class_labels_indices.csv).
            logger.warning("audio_events.labels_not_found", missing=missing)

    def score_window(self, audio_path: str, start: float, end: float) -> dict[str, float]:
        """Returns {"laughter": 0..1, "crowd_reaction": 0..1} -- each the
        max probability across that composite's member classes for this
        window (max, not mean/sum: we care whether the moment contains ANY
        strong hit of that category, not the average across many unrelated
        classes). Never raises -- an extraction/inference failure logs and
        returns zeros, same "optional enrichment can't crash scoring"
        posture as app.core.caption_generation's LLM-call fallback.
        """
        try:
            waveform = extract_window_waveform(audio_path, start, end)
            if waveform.size == 0:
                return {"laughter": 0.0, "crowd_reaction": 0.0}
            clipwise_output, _ = self._tagging.inference(waveform[np.newaxis, :])
            probs = clipwise_output[0]
            laughter = float(max((probs[i] for i in self._laughter_indices), default=0.0))
            crowd_reaction = float(max((probs[i] for i in self._crowd_indices), default=0.0))
            return {"laughter": laughter, "crowd_reaction": crowd_reaction}
        except Exception as exc:
            logger.warning("audio_events.score_window_failed", start=start, end=end, error=str(exc))
            return {"laughter": 0.0, "crowd_reaction": 0.0}


_detector: AudioEventDetector | None = None


def get_audio_event_detector() -> AudioEventDetector:
    """Lazy singleton -- the model load only happens once per worker
    process, on first actual use, not at import time (so a worker that
    never scores a job with ENABLE_AUDIO_EVENT_SCORING on never pays the
    torch-import/model-load cost at all).
    """
    global _detector
    if _detector is None:
        _detector = AudioEventDetector()
    return _detector
