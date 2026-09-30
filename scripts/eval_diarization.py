"""Compare speaker labeling setups with a word-level check.

Usage:
    .venv/bin/python scripts/eval_diarization.py test_data/meeting_01.wav

Runs Whisper once, then each pyannote pipeline with and without the speaker-count hint.
Every word is compared with the ground truth in <audio>_speakers.json, so a line that mixes
two speakers is penalized for each wrong word instead of counting as one correct line.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment

from transcribe import DIARIZATION_MODELS, ROOT, diarize, load_audio, pick_device, speaker_at, transcribe

# Stretch of meeting_01 where two speakers trade short turns. Pass --window to change it.
DEFAULT_WINDOW = (94.0, 121.5)


def true_speaker(word, truth):
    """Ground-truth speaker with the most overlap with the word, or None."""
    overlap = defaultdict(float)
    for t in truth:
        o = min(word["end"], t["end"]) - max(word["start"], t["start"])
        if o > 0:
            overlap[t["speaker"]] += o
    return max(overlap, key=overlap.get) if overlap else None


def word_accuracy(words, hyp, truth_labels, names):
    """Best one-to-one mapping of pyannote labels to names, then the share of words that match."""
    labels = sorted(set(hyp))
    grid = np.zeros((len(labels), len(names)))
    for h, t in zip(hyp, truth_labels):
        grid[labels.index(h), names.index(t)] += 1
    rows, cols = linear_sum_assignment(-grid)
    mapping = {labels[r]: names[c] for r, c in zip(rows, cols)}
    correct = [mapping.get(h) == t for h, t in zip(hyp, truth_labels)]
    return mapping, correct


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--num-speakers", type=int, default=4, help="value for the hinted runs")
    parser.add_argument("--window", type=float, nargs=2, default=DEFAULT_WINDOW, metavar=("START", "END"))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not found. Put it in .env.")
    truth = json.loads(args.audio.with_name(args.audio.stem + "_speakers.json").read_text())

    device = pick_device()
    waveform, sr = load_audio(args.audio)
    print("Transcribing once with Whisper...")
    words = transcribe(waveform, sr, device)

    scored = [(w, true_speaker(w, truth)) for w in words]
    scored = [(w, t) for w, t in scored if t]  # drop words that fall in silence between turns
    words = [w for w, _ in scored]
    truth_labels = [t for _, t in scored]
    names = sorted(set(truth_labels))
    in_window = [args.window[0] <= (w["start"] + w["end"]) / 2 <= args.window[1] for w in words]
    print(f"{len(words)} words scored; window {args.window[0]:.0f}-{args.window[1]:.0f}s has {sum(in_window)}\n")

    rows = []
    for model in DIARIZATION_MODELS:
        for hint in (None, args.num_speakers):
            print(f"Diarizing: {model}, num_speakers={hint}")
            turns = diarize(waveform, sr, device, token, hint, model)
            hyp = [speaker_at(w["start"], w["end"], turns) for w in words]
            mapping, correct = word_accuracy(words, hyp, truth_labels, names)
            window_correct = [c for c, inside in zip(correct, in_window) if inside]
            rows.append(
                (model, str(hint or "auto"), str(len(set(hyp))), f"{100 * np.mean(correct):.1f}%",
                 f"{100 * np.mean(window_correct):.1f}%")
            )

    headers = ("pipeline", "hint", "speakers", "word accuracy", "busy stretch")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print("\n'busy stretch' is word accuracy inside the --window range, where speakers trade short turns.")


if __name__ == "__main__":
    main()
