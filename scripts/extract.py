"""Pull decisions, action items and open questions out of a labeled transcript.

Usage:
    .venv/bin/python scripts/extract.py test_data/meeting_01_transcript.json
    .venv/bin/python scripts/extract.py test_data/meeting_01_transcript_4spk.json --model Qwen/Qwen3-235B-A22B-Instruct-2507

Reads the output of transcribe.py and asks an open model on Hugging Face
Inference Providers to return structured JSON.
The logic lives in meetingmate/extract.py; this is just the command line.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetingmate.extract import DEFAULT_MODEL, ExtractionError, default_output, extract_to_dict  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--participants", help='comma-separated attendee names, e.g. "Priya,Rahul,Meera,Arjun"')
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not found. Put it in .env.")

    participants = [n.strip() for n in args.participants.split(",") if n.strip()] if args.participants else None
    segments = json.loads(args.transcript.read_text())
    print(f"Extracting from {args.transcript.name} with {args.model}...")
    try:
        result = extract_to_dict(segments, args.model, token, participants)
    except ExtractionError as e:
        sys.exit(str(e))
    stray = result.pop("stray_names", None)
    if stray:
        print(f"  warning: names outside the participant list: {stray}")

    out = args.output or default_output(args.transcript, args.model, participants)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"Saved {out}: {len(result['action_items'])} action items, {len(result['decisions'])} decisions, "
        f"{len(result['unassigned_items'])} unassigned, {len(result['open_questions'])} open questions"
    )


if __name__ == "__main__":
    main()
