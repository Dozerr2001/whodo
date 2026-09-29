# MeetingMate

Generates synthetic meeting audio from a text transcript, with a speaker timeline.

## Setup

`requirements.txt` only covers Python packages. You also need:

- **Python 3.12** (3.11 also works)
- **espeak-ng**: `brew install espeak-ng`

Then create the virtualenv and install the packages:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```sh
.venv/bin/python scripts/generate_meeting.py test_data/meeting_01.txt
```

This writes `test_data/meeting_01.wav` and `test_data/meeting_01_speakers.json`.
