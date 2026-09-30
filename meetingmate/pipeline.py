"""The whole pipeline in one place: audio -> labeled transcript -> extraction.

run() is a generator. It yields Progress events while a step is working (so a UI can
show a live status and keep the connection open), then one final Result.
"""

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from . import config
from .config import SPEED_FACTORS
from .extract import DEFAULT_MODEL, LLMUnavailable
from .summarize import SUMMARY_UNAVAILABLE, extract_with_summary
from .transcribe import DEFAULT_DIARIZATION_MODEL, assign_speakers, pick_device
from .worker import WORKER, RunControl, WorkerDied

ROOT = Path(__file__).resolve().parent.parent
DEV_SAVED_RESULT = ROOT / "test_data" / "meeting_01_extracted_cc1_4spk_participants.json"
TIMING_LOG = Path(os.environ.get("MEETINGMATE_LOG_DIR", ROOT / "logs")) / "timings.jsonl"
TICK_SECONDS = 2.0  # how often a Progress event goes out
STOP_POLL_SECONDS = 0.1  # how often a run looks at its stop switch

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


def dev_extraction(segments, llm, token, participants, stopped=None):
    """Stands in for extract_with_summary in dev mode (config.DEV_LLM): saved output, or a pretend outage."""
    if config.DEV_LLM == "outage":
        raise LLMUnavailable()
    result = json.loads(DEV_SAVED_RESULT.read_text())
    if config.DEV_LLM == "summary-outage":
        result["summary"], result["summary_error"] = None, SUMMARY_UNAVAILABLE
    return result


def estimate_seconds(audio_seconds, device=None):
    """Rough total run time for a recording of this length."""
    f = SPEED_FACTORS[device or pick_device()]
    return audio_seconds * (f["transcribe"] + f["diarize"]) + f["fixed"]


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


def _wait_for_worker(index, expected, control):
    """Wait for the worker's next message, yielding a Progress every few seconds.

    Checks the stop switch every fraction of a second. Use as `value = yield from _wait_for_worker(...)`.
    """
    stage, label = STEPS[index]
    start = last_tick = time.time()
    while True:
        control.check()
        try:
            message = WORKER.poll(STOP_POLL_SECONDS)
        except WorkerDied:
            control.check()  # a stop kills the worker on purpose, which looks the same from here
            raise RuntimeError("The transcription process ended unexpectedly. It may have run out of memory.")
        if message:
            kind, value = message
            if kind == "error":
                raise RuntimeError(value)
            assert kind == expected, f"worker sent {kind!r}, expected {expected!r}"
            return value
        if time.time() - last_tick >= TICK_SECONDS:
            last_tick = time.time()
            yield Progress(index + 1, stage, label, last_tick - start)


def _step(index, control, fn, *args):
    """Run fn in a worker thread, yielding a Progress every few seconds until it finishes.

    Used for the LLM step, which waits on the network and can't be killed, only abandoned: if the
    run is stopped the thread is left to finish and its answer is thrown away.
    """
    stage, label = STEPS[index]
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args)
    start = last_tick = time.time()
    try:
        while True:
            try:
                return future.result(timeout=STOP_POLL_SECONDS)
            except TimeoutError:
                control.check()
                if time.time() - last_tick >= TICK_SECONDS:
                    last_tick = time.time()
                    yield Progress(index + 1, stage, label, last_tick - start)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)  # if the user cancels, don't block on the running step


def run(audio_path, token, participants=None, num_speakers=None,
        diarization_model=DEFAULT_DIARIZATION_MODEL, llm=DEFAULT_MODEL, control=None):
    """Yield Progress events, then one Result. Raises Stopped if control.stop() is called mid-run."""
    control = control or RunControl()
    device = pick_device()
    audio_seconds = sf.info(str(audio_path)).duration
    timings = {}
    workdir = tempfile.mkdtemp(prefix="meetingmate_")  # the worker's scratch space, deleted however the run ends
    finished = False
    proc = None
    worker_busy = [True]  # True from submit until the worker has sent everything. Only a busy worker is worth killing.
    try:
        control.check()
        proc = WORKER.submit({
            "audio": str(audio_path), "device": device, "token": token, "num_speakers": num_speakers,
            "diarization_model": diarization_model, "workdir": workdir,
        })
        control.on_stop(lambda: WORKER.kill(proc) if worker_busy[0] else None)  # the Stop button kills it straight away, from its own thread

        started = time.time()
        yield Progress(1, *STEPS[0], 0.0)
        words = yield from _wait_for_worker(0, "words", control)
        timings[STEPS[0][0]] = time.time() - started

        started = time.time()
        yield Progress(2, *STEPS[1], 0.0)
        turns = yield from _wait_for_worker(1, "turns", control)
        timings[STEPS[1][0]] = time.time() - started
        worker_busy[0] = False  # idle again and still holding the loaded models

        segments = assign_speakers(words, turns)
        for seg in segments:
            seg["start"], seg["end"] = round(seg["start"], 2), round(seg["end"], 2)

        control.check()  # stopped before this point means the LLM is never called
        started = time.time()
        yield Progress(3, *STEPS[2], 0.0)
        extraction = yield from _step(2, control, dev_extraction if config.DEV_LLM else extract_with_summary, segments, llm, token, participants, lambda: control.stopped)
        timings[STEPS[2][0]] = time.time() - started

        timings["total"] = sum(timings.values())
        _log_timing(audio_seconds, device, timings)
        finished = True
        yield Result(segments, extraction, timings)
    finally:
        if not finished and worker_busy[0]:  # stopped, failed or abandoned mid-step: start a fresh worker next time
            WORKER.kill(proc)
        shutil.rmtree(workdir, ignore_errors=True)


def load_cached(transcript_path, extraction_path):
    """Build a Result from saved files, for the sample meeting on a Space."""
    return Result(
        json.loads(Path(transcript_path).read_text()),
        json.loads(Path(extraction_path).read_text()),
        cached=True,
    )
