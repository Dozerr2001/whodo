# MeetingMate

Upload a meeting recording, get back the decisions made, the action items with owners and deadlines, and the questions still left open.

## The problem

Meetings end, and the stuff that actually matters, who is doing what by when, lives in someone's scribbled notes or nowhere at all. Existing note takers give you a summary, but a summary is not a task list. Somebody still has to read it, pull out the action items, and chase people.

## Who it's for

Chiefs of staff, project managers, and team leads who run recurring meetings and are on the hook for follow-through.

## How it works

MeetingMate runs on three pieces:

1. **Whisper** turns speech into text. *What was said.*
2. **pyannote** figures out who spoke when. *Who said it.*
3. **An LLM** reads the labeled transcript and pulls out decisions, action items, owners, and deadlines. *What it means.*

Pipeline: Audio → Whisper + pyannote → labeled transcript → LLM → action item table

## What v1 does

- Takes an uploaded meeting recording
- Transcribes it and labels who said what
- Returns decisions, action items (task, owner, deadline), and open questions
- Flags action items with no clear owner as unassigned instead of guessing
- Shows results as a table in the app, with a CSV download

## Out of scope for v1

- Joining live calls as a bot
- Google Sheets or Notion sync (planned for v1.5)
- Non-English meetings
- Reminders and follow-ups

## How we know it works

Tested against meetings where the correct answers are known. Targets:

- Catches at least 80% of real action items
- Gets the owner right on at least 90% of the ones it catches
- Invents no more than one fake action item per meeting
- Processes a 10-minute recording in under 5 minutes on a laptop

## Project structure

```
MeetingMate/
├── app/                  the app itself (coming soon)
├── docs/
│   └── brief.md          project brief
├── scripts/
│   └── generate_meeting.py   turns a meeting script into multi-voice audio
├── test_data/
│   ├── meeting_01.txt              synthetic meeting script
│   └── meeting_01_answer_key.json  correct answers for scoring
├── requirements.txt
├── .gitignore
└── .env                  Hugging Face token (never committed)
```

## Setup

Needs macOS with Homebrew, and Python 3.12.

1. Install espeak-ng (used by the text-to-speech model):
   ```
   brew install espeak-ng
   ```
2. Create and activate a virtual environment with Python 3.12:
   ```
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the Python packages:
   ```
   pip install -r requirements.txt
   ```
4. On Hugging Face, accept the terms for these two gated models:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
5. Create a read-access token on Hugging Face (Settings → Access Tokens) and add it to a `.env` file in the project root:
   ```
   HF_TOKEN=your_token_here
   ```

## Usage

Generate a synthetic test meeting:

```
python scripts/generate_meeting.py test_data/meeting_01.txt
```

This creates `test_data/meeting_01.wav` (the audio) and `test_data/meeting_01_speakers.json` (the exact start and end time of every line, used to check speaker labeling accuracy).

## Test data

`meeting_01` is a 3-minute, 4-person meeting about launching a referral program. It has traps planted on purpose to test whether the app actually understands the conversation:

- **Deadline change:** launch moves from November 3 to November 10
- **Reassignment:** the FAQ page is offered to one person, then taken by another
- **Cancelled item:** an email campaign is suggested, then dropped
- **Vague owner:** "someone should check" with nobody assigned
- **Non-decision:** WhatsApp vs SMS is explicitly pushed to next week

The answer key lists exactly what the app should and should not return.

## Status

- [x] Synthetic meeting generator (Kokoro text-to-speech)
- [ ] Transcription with speaker labels (Whisper + pyannote)
- [ ] Action item extraction (LLM)
- [ ] Scoring against the answer key
- [ ] App interface (Gradio)
- [ ] Deploy to Hugging Face Spaces
