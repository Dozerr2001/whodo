"""Transcribe a meeting recording and label who spoke when.

Usage:
    .venv/bin/python scripts/transcribe.py test_data/meeting_01.wav
    .venv/bin/python scripts/transcribe.py test_data/meeting_01.wav -o out.json

Steps: Whisper (what was said) + pyannote (who spoke when) -> merged segments.
The logic lives in meetingmate/transcribe.py; this is just the command line.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-exported so other scripts (eval_diarization.py) can keep importing from here.
from meetingmate.transcribe import (  # noqa: E402, F401
    DEFAULT_DIARIZATION_MODEL,
    DIARIZATION_MODELS,
    ROOT,
    assign_speakers,
    diarize,
    load_audio,
    pick_device,
    speaker_at,
    transcribe,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="default: test_data/<audio name>_transcript.json")
    parser.add_argument("--num-speakers", type=int, help="exact number of speakers, if known (default: pyannote guesses)")
    parser.add_argument("--diarization-model", choices=DIARIZATION_MODELS, default=DEFAULT_DIARIZATION_MODEL, help=f"pyannote pipeline (default: {DEFAULT_DIARIZATION_MODEL})")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not found. Put it in .env.")

    device = pick_device()
    print(f"Device: {device}")
    waveform, sr = load_audio(args.audio)

    print("Transcribing with Whisper...")
    words = transcribe(waveform, sr, device)
    print(f"  {len(words)} words")

    print("Diarizing with pyannote...")
    turns = diarize(waveform, sr, device, token, args.num_speakers, args.diarization_model)
    print(f"  {len(turns)} speaker turns, {len({t[2] for t in turns})} speakers")

    labeled = assign_speakers(words, turns)
    for seg in labeled:
        seg["start"], seg["end"] = round(seg["start"], 2), round(seg["end"], 2)

    out = args.output or ROOT / "test_data" / f"{args.audio.stem}_transcript.json"
    out.write_text(json.dumps(labeled, indent=2, ensure_ascii=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
