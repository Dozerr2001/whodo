"""Settings that differ between running on a laptop and running on a Hugging Face Space.

Everything can be overridden with an environment variable.
"""

import os

# Hugging Face sets SPACE_ID inside every Space.
ON_SPACES = bool(os.environ.get("SPACE_ID"))


def _minutes(value, default):
    """Parse a minutes setting. Empty or 0 means no limit."""
    if value is None:
        return default
    try:
        minutes = float(value)
    except ValueError:
        return default
    return minutes if minutes > 0 else None


# Longest recording the app accepts, in minutes. None = no limit (local); 10 on a Space.
MAX_AUDIO_MINUTES = _minutes(os.environ.get("MEETINGMATE_MAX_AUDIO_MINUTES"), 10 if ON_SPACES else None)

# "live" runs the real pipeline on the sample recording; "cached" shows saved results.
SAMPLE_MODE = os.environ.get("MEETINGMATE_SAMPLE_MODE", "cached" if ON_SPACES else "live")

# Seconds of processing per second of audio, plus a fixed cost for the LLM call and model loading.
# "mps" is measured (Mac, 3-minute sample: Whisper 72s, pyannote 18s, LLM 2s). "cuda" and "cpu" are
# unmeasured guesses. Every run appends its real timings to logs/timings.jsonl so these can be corrected.
SPEED_FACTORS = {
    "mps": {"transcribe": 0.40, "diarize": 0.10, "fixed": 10},
    "cuda": {"transcribe": 0.03, "diarize": 0.03, "fixed": 30},
    "cpu": {"transcribe": 0.8, "diarize": 0.4, "fixed": 30},
}
