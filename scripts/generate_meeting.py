"""
Turn a meeting script into multi-speaker audio using Kokoro TTS
(hexgrad/Kokoro-82M on Hugging Face).

Also saves a speaker timeline (who spoke when) that you can use later
to check how accurate your speaker labeling is.

Setup:
    pip install kokoro soundfile numpy
    You may also need espeak-ng installed on your system:
      - Windows: download the .msi installer from the espeak-ng GitHub releases page
      - Mac: brew install espeak-ng
      - Linux: sudo apt-get install espeak-ng

Run:
    python generate_meeting.py meeting_01.txt

Output:
    meeting_01.wav            the meeting audio
    meeting_01_speakers.json  exact start and end time of every line
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000      # Kokoro outputs 24 kHz audio
PAUSE_SECONDS = 0.6      # silence between speakers

# One voice per speaker. Kokoro American English voices:
# af_* are female voices, am_* are male voices.
VOICES = {
    "PRIYA": "af_heart",
    "MEERA": "af_bella",
    "RAHUL": "am_adam",
    "ARJUN": "am_michael",
}


def parse_script(path):
    """Read lines like 'PRIYA: some text' into (speaker, text) pairs."""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        match = re.match(r"^([A-Z]+):\s*(.+)$", raw)
        if not match:
            print(f"Skipping line with no speaker: {raw}")
            continue
        lines.append((match.group(1), match.group(2)))
    return lines


def to_numpy(audio):
    """Kokoro may return a torch tensor, so convert it to a numpy array."""
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def main():
    script_path = Path(sys.argv[1] if len(sys.argv) > 1 else "meeting_01.txt")
    lines = parse_script(script_path)
    if not lines:
        sys.exit("No lines found in the script.")

    unknown = {speaker for speaker, _ in lines if speaker not in VOICES}
    if unknown:
        sys.exit(f"No voice set for: {', '.join(unknown)}. Add them to VOICES.")

    print("Loading Kokoro (first run downloads the model, give it a minute)...")
    pipeline = KPipeline(lang_code="a")  # "a" = American English

    pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype=np.float32)
    chunks = []
    timeline = []
    cursor = 0.0

    for i, (speaker, text) in enumerate(lines, start=1):
        print(f"[{i}/{len(lines)}] {speaker}: {text[:60]}")

        parts = [to_numpy(audio) for _, _, audio in pipeline(text, voice=VOICES[speaker])
                 if audio is not None]
        line_audio = np.concatenate(parts)

        start = cursor
        end = start + len(line_audio) / SAMPLE_RATE
        timeline.append({
            "speaker": speaker.title(),
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
        })

        chunks.extend([line_audio, pause])
        cursor = end + PAUSE_SECONDS

    full_audio = np.concatenate(chunks)
    wav_path = script_path.with_suffix(".wav")
    timeline_path = script_path.with_name(script_path.stem + "_speakers.json")

    sf.write(wav_path, full_audio, SAMPLE_RATE)
    timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    minutes = len(full_audio) / SAMPLE_RATE / 60
    print(f"\nDone. {minutes:.1f} minutes of audio.")
    print(f"Audio:    {wav_path}")
    print(f"Timeline: {timeline_path}")


if __name__ == "__main__":
    main()
