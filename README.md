# WhoDo

**Turn a meeting recording into who does what, by when.**

Upload a meeting recording, get back the decisions made, the action items with owners and deadlines, and the questions still left open.

## The problem

Meetings end, and the stuff that actually matters, who is doing what by when, lives in someone's scribbled notes or nowhere at all. Existing note takers give you a summary, but a summary is not a task list. Somebody still has to read it, pull out the action items, and chase people.

## Who it's for

Chiefs of staff, project managers, and team leads who run recurring meetings and are on the hook for follow-through.

## How it works

WhoDo runs on three pieces:

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
WhoDo/
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

## Known issues

- **pyannote merges similar voices and fast turns.** On `meeting_01`, pyannote finds 3 speakers instead of 4 and merges Meera into Priya. An earlier line-by-line check reported 92.3% accuracy, but that was too generous: a line holding words from two speakers counted as correct if one speaker dominated it. Checked word by word with `scripts/eval_diarization.py`, the unhinted result is **79.0%** of words with the right speaker, and only **43.9%** in the busy stretch (94-122s) where Priya and Meera trade short turns. Wrong speakers mean wrong owners in the action item table.
- **Telling it the speaker count helps a lot.** With `--num-speakers 4`, word accuracy rises to 89.6% (81.8% in the busy stretch). The real app may not know the speaker count, so it could be an optional field on the upload form.
- **`pyannote/speaker-diarization-community-1` is no better here.** Same 79.0% unhinted and 90.4% with 4 speakers, a difference of about 3 words, which is within run-to-run noise. `transcribe.py` supports both through `--diarization-model`.
- **Owner accuracy depended on the speaker labels and names, not on the LLM.** Reading the transcripts alone, gpt-oss-120b caught all 5 action items and passed 4 of 5 traps, but got only 2 of 5 owners right. Priya's name is never spoken in the meeting, so it can't be inferred, and Whisper writes "Meera" as "Mira". The failed trap was the FAQ reassignment, which happens inside the busy stretch above.
- **A participant list fixes most of that.** Running `extract.py` with `--participants "Priya,Rahul,Meera,Arjun"` restricts names to that list, corrects misspellings, and uses cues like "Meera, design?" to correct speaker labels. On the best transcript (community-1, 4 speakers) it raised owners from 2/5 to 4/5 and all 5 traps passed. The remaining miss is a mockups task left as `unknown` even though Meera was the only attendee left over. Each number is one run on one synthetic meeting, and LLM output can vary between runs, so treat the results as rough. In the app, the participant list could be another optional upload field.
- **Meetings over roughly 10 minutes exceed Groq's free tier per-request limit and fall back to HF.** Groq's free tier allows 8,000 tokens per request for `gpt-oss-120b`, and a request is the transcript plus the instructions plus room for the answer. The first 10 minutes of a real recording fit and completed on Groq; the full 19 minutes asked for about 9,700 and was refused with a 413. The app estimates the size first and goes straight to Hugging Face Inference Providers when it is over, which uses HF credits. Planned fix: split long transcripts into chunks.

## Status

- [x] Synthetic meeting generator (Kokoro text-to-speech)
- [x] Transcription with speaker labels (Whisper + pyannote)
- [x] Action item extraction (LLM)
- [x] Scoring against the answer key
- [ ] App interface (Gradio)
- [ ] Deploy to Hugging Face Spaces
