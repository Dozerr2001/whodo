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
MAX_AUDIO_MINUTES = _minutes(os.environ.get("WHODO_MAX_AUDIO_MINUTES"), 10 if ON_SPACES else None)

# "live" runs the real pipeline on the sample recording; "cached" shows saved results.
SAMPLE_MODE = os.environ.get("WHODO_SAMPLE_MODE", "cached" if ON_SPACES else "live")

# Dev mode: keep the real Whisper and pyannote steps but never call the LLM, so working on the UI costs no credits.
#   "saved"          the saved sample results stand in for the LLM's answer
#   "outage"         pretend the LLM is unavailable (to see the error message)
#   "summary-outage" saved results, but pretend only the summary call failed
# Ignored on a Space, so a stray setting can never put fake results in front of visitors.
DEV_LLM = "" if ON_SPACES else os.environ.get("WHODO_DEV_LLM", "").strip().lower()
if DEV_LLM not in ("", "saved", "outage", "summary-outage"):
    raise SystemExit(f"WHODO_DEV_LLM must be saved, outage or summary-outage, not {DEV_LLM!r}")

# How hard gpt-oss thinks on Groq before answering: "low", "medium", "high", or "default" to leave it to Groq.
# The hidden thinking counts as output tokens, and on the free tier (8,000 tokens a minute) it was most of each call.
# Off ("default") because "low" made meeting_01 results worse in all 3 runs: a missed decision, a failed trap,
# owners at 3/5, and summaries that contradict the extraction. See CLAUDE.md.
GROQ_REASONING_EFFORT = os.environ.get("WHODO_GROQ_REASONING_EFFORT", "default").strip().lower()
if GROQ_REASONING_EFFORT not in ("low", "medium", "high", "default"):
    raise SystemExit(f"WHODO_GROQ_REASONING_EFFORT must be low, medium, high or default, not {GROQ_REASONING_EFFORT!r}")

# Seconds of processing per second of audio, plus a fixed cost for the LLM call and model loading.
# "mps" is measured (Mac, 3-minute sample: Whisper 72s, pyannote 18s, LLM 2s). "cuda" and "cpu" are
# unmeasured guesses. Every run appends its real timings to logs/timings.jsonl so these can be corrected.
SPEED_FACTORS = {
    "mps": {"transcribe": 0.40, "diarize": 0.10, "fixed": 10},
    "cuda": {"transcribe": 0.03, "diarize": 0.03, "fixed": 30},
    "cpu": {"transcribe": 0.8, "diarize": 0.4, "fixed": 30},
}
