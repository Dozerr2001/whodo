"""Transcribe a meeting recording and label who spoke when.

Usage:
    .venv/bin/python scripts/transcribe.py test_data/meeting_01.wav
    .venv/bin/python scripts/transcribe.py test_data/meeting_01.wav -o out.json

Steps: Whisper (what was said) + pyannote (who spoke when) -> merged segments.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_SR = 16000


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(path):
    """Return a mono float32 tensor of shape (1, samples) and its sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.mean(axis=1)).unsqueeze(0), sr


def transcribe(waveform, sr, device):
    """Whisper -> list of {start, end, text}, one entry per word."""
    audio_16k = AF.resample(waveform, sr, WHISPER_SR).squeeze(0).numpy()
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    result = asr(
        {"raw": audio_16k, "sampling_rate": WHISPER_SR},
        return_timestamps="word",
        chunk_length_s=30,
        generate_kwargs={"language": "english"},
    )
    duration = len(audio_16k) / WHISPER_SR
    words = []
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        if end is None:  # Whisper sometimes leaves the last end open
            end = duration
        text = chunk["text"].strip()
        if text:
            words.append({"start": start, "end": end, "text": text})
    return words


def diarize(waveform, sr, device, token):
    """pyannote -> list of (start, end, speaker) turns."""
    pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    pipe.to(torch.device(device))
    output = pipe({"waveform": waveform, "sample_rate": sr})
    # Newer pyannote versions wrap the result; older ones return the Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def speaker_at(start, end, turns):
    """Speaker with the most overlap with [start, end]; nearest turn if none."""
    overlap = {}
    for t_start, t_end, speaker in turns:
        o = min(end, t_end) - max(start, t_start)
        if o > 0:
            overlap[speaker] = overlap.get(speaker, 0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]


def assign_speakers(words, turns):
    """Label every word, then group words into lines.

    A new line starts when the speaker changes or the previous word ended a sentence.
    """
    lines = []
    for w in words:
        speaker = speaker_at(w["start"], w["end"], turns)
        prev = lines[-1] if lines else None
        if prev and prev["speaker"] == speaker and not prev["text"].endswith((".", "?", "!")):
            prev["text"] += " " + w["text"]
            prev["end"] = w["end"]
        else:
            lines.append({"speaker": speaker, **w})
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="default: test_data/<audio name>_transcript.json")
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
    turns = diarize(waveform, sr, device, token)
    print(f"  {len(turns)} speaker turns, {len({t[2] for t in turns})} speakers")

    labeled = assign_speakers(words, turns)
    for seg in labeled:
        seg["start"], seg["end"] = round(seg["start"], 2), round(seg["end"], 2)

    out = args.output or ROOT / "test_data" / f"{args.audio.stem}_transcript.json"
    out.write_text(json.dumps(labeled, indent=2, ensure_ascii=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
