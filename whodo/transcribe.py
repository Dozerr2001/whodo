"""Whisper (what was said) + pyannote (who spoke when), merged into labeled lines.

The command-line entry point is scripts/transcribe.py.
"""

import functools
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_DIARIZATION_MODEL = "community-1"
DIARIZATION_MODELS = {
    "3.1": "pyannote/speaker-diarization-3.1",
    "community-1": "pyannote/speaker-diarization-community-1",
}
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


@functools.cache
def load_whisper(device):
    """Load Whisper once per process; later calls reuse it."""
    return hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )


@functools.cache
def load_diarizer(model, token, device):
    """Load a pyannote pipeline once per process; later calls reuse it."""
    pipe = Pipeline.from_pretrained(DIARIZATION_MODELS[model], token=token)
    pipe.to(torch.device(device))
    return pipe


def transcribe(waveform, sr, device):
    """Whisper -> list of {start, end, text}, one entry per word."""
    audio_16k = AF.resample(waveform, sr, WHISPER_SR).squeeze(0).numpy()
    asr = load_whisper(device)
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


def diarize(waveform, sr, device, token, num_speakers=None, model=DEFAULT_DIARIZATION_MODEL):
    """pyannote -> list of (start, end, speaker) turns."""
    pipe = load_diarizer(model, token, device)
    output = pipe({"waveform": waveform, "sample_rate": sr}, num_speakers=num_speakers)
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
