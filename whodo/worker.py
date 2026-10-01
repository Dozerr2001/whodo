"""Run the heavy steps (Whisper, pyannote) in a separate process so a run can be killed mid-step.

A Python thread can't be interrupted, and Whisper and pyannote each block for minutes. A child
process can be killed at any moment. One worker stays alive between runs, so the models stay loaded;
only after a kill does the next run pay to start a new one and reload them.

The child is started with `python -m whodo.worker` (not multiprocessing's "spawn", which would
re-import the whole app in the child). It talks to the parent over two pipes, and the parent starts
it in its own process group so one SIGKILL takes down anything it started.
"""

import atexit
import os
import signal
import subprocess
import sys
import tempfile
import threading
from multiprocessing.connection import Connection
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Stopped(Exception):
    """The user stopped the run."""


class WorkerDied(Exception):
    """The worker process ended without being asked to."""


class RunControl:
    """One run's stop switch. stop() can be called from any thread, for example the Stop button's handler."""

    def __init__(self):
        self._event = threading.Event()
        self._on_stop = []

    @property
    def stopped(self):
        return self._event.is_set()

    def check(self):
        if self.stopped:
            raise Stopped()

    def on_stop(self, callback):
        """Call callback when the run is stopped (straight away if it already was)."""
        self._on_stop.append(callback)
        if self.stopped:
            callback()

    def stop(self):
        self._event.set()
        for callback in self._on_stop:
            callback()


class Worker:
    """The parent's handle on the child process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._to_child = None
        self._from_child = None

    def submit(self, job):
        """Send a job, starting the child first if needed. Returns the process, to pass to kill()."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            self._to_child.send(job)
            return self._proc

    def poll(self, timeout):
        """The child's next message, or None if nothing arrived within timeout seconds."""
        inbox = self._from_child
        try:
            if inbox is not None and inbox.poll(timeout):
                return inbox.recv()
        except (EOFError, OSError, ValueError):  # the child ended, or kill() closed the pipe under us
            raise WorkerDied()
        if inbox is None:
            raise WorkerDied()
        return None

    def kill(self, proc=None):
        """Kill the child now. Given a proc, only if that is still the current one (a later run may own a new child)."""
        with self._lock:
            if self._proc is None or (proc is not None and proc is not self._proc):
                return
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)  # the whole process group
            except (ProcessLookupError, PermissionError):
                pass
            self._proc.kill()
            self._proc.wait()
            for conn in (self._to_child, self._from_child):
                try:
                    conn.close()
                except OSError:
                    pass
            self._proc = self._to_child = self._from_child = None

    def _start(self):
        job_r, job_w = os.pipe()
        result_r, result_w = os.pipe()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "whodo.worker", str(job_r), str(result_w)],
            cwd=ROOT, pass_fds=(job_r, result_w), stdin=subprocess.DEVNULL, start_new_session=True,
        )
        os.close(job_r)
        os.close(result_w)
        self._to_child = Connection(job_w, readable=False)
        self._from_child = Connection(result_r, writable=False)


WORKER = Worker()
atexit.register(WORKER.kill)


def _child_main(job_fd, result_fd):
    """Runs in the child: wait for jobs, send back the words, then the speaker turns."""
    jobs = Connection(job_fd, writable=False)
    results = Connection(result_fd, readable=False)
    from .transcribe import diarize, load_audio, transcribe  # the slow imports, only here

    while True:
        try:
            job = jobs.recv()
        except EOFError:  # the parent is gone
            return
        try:
            tempfile.tempdir = job["workdir"]  # anything this run writes goes where the parent will delete it
            waveform, sr = load_audio(job["audio"])
            results.send(("words", transcribe(waveform, sr, job["device"])))
            turns = diarize(waveform, sr, job["device"], job["token"], job["num_speakers"], job["diarization_model"])
            results.send(("turns", turns))
        except Exception as e:
            results.send(("error", f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    _child_main(int(sys.argv[1]), int(sys.argv[2]))
