"""The whole pipeline in one place: audio -> labeled transcript -> extraction.

run() is a generator. It yields Progress events while a step is working (so a UI can
show a live status and keep the connection open), then one final Result.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import SPEED_FACTORS
from .extract import DEFAULT_MODEL, UNKNOWN, extract_to_dict
from .transcribe import DEFAULT_DIARIZATION_MODEL, assign_speakers, diarize, load_audio, pick_device, transcribe

ROOT = Path(__file__).resolve().parent.parent
TIMING_LOG = Path(os.environ.get("MEETINGMATE_LOG_DIR", ROOT / "logs")) / "timings.jsonl"
TICK_SECONDS = 2.0

STEPS = [
    ("transcribing", "Transcribing speech"),
    ("diarizing", "Identifying speakers"),
    ("extracting", "Extracting decisions and action items"),
]


@dataclass
class Progress:
    step: int  # 1-based position in STEPS
    stage: str
    label: str
    elapsed: float  # seconds spent on this step so far


@dataclass
class Result:
    segments: list  # labeled transcript lines
    extraction: dict
    timings: dict = field(default_factory=dict)  # seconds per step; empty for cached results
    cached: bool = False


def estimate_seconds(audio_seconds, device=None):
    """Rough total run time for a recording of this length."""
    f = SPEED_FACTORS[device or pick_device()]
    return audio_seconds * (f["transcribe"] + f["diarize"]) + f["fixed"]


def name_speakers(segments, speaker_map):
    """Swap SPEAKER_00 for a real name where the model found one."""
    def name(label):
        found = speaker_map.get(label, UNKNOWN)
        return label if found.lower() == UNKNOWN else found
    return [{**s, "speaker": name(s["speaker"])} for s in segments]


def _log_timing(audio_seconds, device, timings):
    """Append one line per run so the wait-time estimates can be checked against reality."""
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "audio_seconds": round(audio_seconds, 1),
        **{k: round(v, 1) for k, v in timings.items()},
    }
    try:
        TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TIMING_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # a read-only disk should never fail a run
    print("timing:", entry)


def _step(index, fn, *args):
    """Run fn in a worker thread, yielding a Progress every few seconds until it finishes.

    Whisper and pyannote each block for minutes and report nothing. The heartbeat is what
    keeps the browser connection from looking idle. Use as `value = yield from _step(...)`.
    """
    stage, label = STEPS[index]
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args)
    start = time.time()
    try:
        while True:
            try:
                return future.result(timeout=TICK_SECONDS)
            except TimeoutError:
                yield Progress(index + 1, stage, label, time.time() - start)
    finally:
        executor.shutdown(wait=False)  # if the user cancels, don't block on the running step


def run(audio_path, token, participants=None, num_speakers=None,
        diarization_model=DEFAULT_DIARIZATION_MODEL, llm=DEFAULT_MODEL):
    device = pick_device()
    waveform, sr = load_audio(audio_path)
    audio_seconds = waveform.shape[1] / sr
    timings = {}

    def timed(index, fn, *args):
        started = time.time()
        yield Progress(index + 1, *STEPS[index], 0.0)
        value = yield from _step(index, fn, *args)
        timings[STEPS[index][0]] = time.time() - started
        return value

    words = yield from timed(0, transcribe, waveform, sr, device)
    turns = yield from timed(1, diarize, waveform, sr, device, token, num_speakers, diarization_model)
    segments = assign_speakers(words, turns)
    for seg in segments:
        seg["start"], seg["end"] = round(seg["start"], 2), round(seg["end"], 2)
    extraction = yield from timed(2, extract_to_dict, segments, llm, token, participants)

    timings["total"] = sum(timings.values())
    _log_timing(audio_seconds, device, timings)
    yield Result(segments, extraction, timings)


def load_cached(transcript_path, extraction_path):
    """Build a Result from saved files, for the sample meeting on a Space."""
    return Result(
        json.loads(Path(transcript_path).read_text()),
        json.loads(Path(extraction_path).read_text()),
        cached=True,
    )
