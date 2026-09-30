"""Stop a real run during each stage and check what is left behind.

Slow (a few minutes): it runs the real Whisper and pyannote models. The LLM is faked (tests/fake_llm.py), so it
costs no credits. Needs HF_TOKEN in .env for the gated pyannote model.
Run with: .venv/bin/python tests/test_stop.py

For each stage (transcribing, identifying speakers, extracting) it stops the run from another
thread, the way the Stop button does, and checks that:
  - the run ends almost at once, with Stopped
  - the worker process and its process group are gone (except after the heavy steps: an idle worker is kept)
  - the run's temp directory is deleted
  - the LLM was not called for stages before extraction, and the second (summary) call never happens
  - the uploaded file is still there
Finally it runs a complete analysis to show the next run works after a stop.
"""

import glob
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
load_dotenv(ROOT / ".env")

import fake_llm  # noqa: E402  (tests/fake_llm.py)
from meetingmate.pipeline import Progress, Result, run  # noqa: E402
from meetingmate.worker import WORKER, RunControl, Stopped  # noqa: E402

AUDIO = ROOT / "test_data" / "meeting_01.mp3"
TOKEN = os.environ["HF_TOKEN"]
STOP_AFTER_SECONDS = 3  # into the stage, so the stop lands in the middle of the heavy work
MAX_STOP_DELAY = 1.5  # seconds from pressing Stop to the run ending
llm_calls = fake_llm.calls
fake_llm.install()
fake_llm.mode.update(kind="slow", delay=10.0)  # an LLM call that takes 10s, so a stop can land in the middle of it


def leftover_dirs():
    return glob.glob(os.path.join(tempfile.gettempdir(), "meetingmate_*"))


def stop_during(step):
    """Start a run, stop it STOP_AFTER_SECONDS into the given step (1 to 3). Returns a dict of measurements."""
    llm_calls.clear()
    control = RunControl()
    seen = {}
    stop_thread = None
    t_stop = None
    outcome = "finished"
    t_end = None
    try:
        for event in run(str(AUDIO), TOKEN, ["Priya", "Rahul", "Meera", "Arjun"], 4, control=control):
            if isinstance(event, Progress) and event.step == step and event.elapsed >= (STOP_AFTER_SECONDS if step < 3 else 0) and not stop_thread:
                seen["pid"] = WORKER._proc.pid if WORKER._proc else None
                t_stop = time.time()
                stop_thread = threading.Thread(target=control.stop)  # like the Stop button's handler
                stop_thread.start()
            if isinstance(event, Result):
                break
    except Stopped:
        outcome = "stopped"
        t_end = time.time()
    stop_thread.join()
    seen.update(outcome=outcome, delay=(t_end - t_stop) if t_end else None)
    time.sleep(12 if step == 3 else 0)  # give the abandoned LLM call time to finish, to see if a second one starts
    seen["llm_calls"] = len(llm_calls)
    return seen


def process_gone(pid):
    for probe in (os.kill, os.killpg):
        try:
            probe(pid, 0)
            return False
        except ProcessLookupError:
            pass
    return True


def main():
    failures = []

    def check(ok, what):
        print(f"    {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            failures.append(what)

    for step, name, llm_expected in [(1, "transcribing", 0), (2, "identifying speakers", 0), (3, "extracting", 1)]:
        print(f"\nStop during step {step}: {name}")
        r = stop_during(step)
        check(r["outcome"] == "stopped", "run ended with Stopped")
        check(r["delay"] is not None and r["delay"] < MAX_STOP_DELAY, f"ended {r['delay']:.2f}s after Stop (limit {MAX_STOP_DELAY}s)" if r["delay"] is not None else "ended")
        if step < 3:
            check(r["pid"] is not None and process_gone(r["pid"]), f"worker process {r['pid']} and its group are gone")
            check(WORKER._proc is None, "no worker left registered")
        else:  # the worker had finished and was only idling, so the stop leaves it alone and the models stay loaded
            check(WORKER._proc is not None and WORKER._proc.poll() is None, "idle worker kept alive (models stay loaded)")
        check(not leftover_dirs(), f"temp directory removed ({leftover_dirs() or 'none left'})")
        check(r["llm_calls"] == llm_expected, f"LLM called {r['llm_calls']} time(s), expected {llm_expected}" + (" (the summary call was skipped)" if step == 3 else ""))
        check(AUDIO.exists(), "uploaded file still there")

    print("\nFull run after the stops (proves the queue and worker recover)")
    fake_llm.mode.update(kind="instant", delay=0.0)
    t = time.time()
    result = [e for e in run(str(AUDIO), TOKEN, ["Priya", "Rahul", "Meera", "Arjun"], 4) if isinstance(e, Result)]
    check(len(result) == 1 and result[0].extraction["summary"]["overview"] == "Fake summary.", f"complete result in {time.time() - t:.0f}s")
    check(not leftover_dirs(), "temp directory removed after a normal run too")
    WORKER.kill()

    print("\n" + ("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
