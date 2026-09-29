# MeetingMate - context for Claude

## What this project is

MeetingMate takes a meeting recording and returns decisions, action items (task, owner, deadline), and open questions. It is a learning project: the goal is to learn how to build and ship AI products using models from Hugging Face, ending with a public app on Hugging Face Spaces.

## How it works

1. **Whisper** turns speech into text. *What was said.*
2. **pyannote** figures out who spoke when. *Who said it.*
3. **An LLM** reads the labeled transcript and pulls out decisions, action items, owners, and deadlines. *What it means.*

Pipeline: Audio → Whisper + pyannote → labeled transcript → LLM → action item table

## Project structure

- `scripts/` helper scripts (generate_meeting.py makes synthetic test audio with Kokoro TTS)
- `test_data/` meeting scripts, generated audio, speaker timelines, and answer keys
- `app/` the actual app (not built yet)
- `docs/brief.md` project brief with scope and success targets

## Environment notes

- Always run Python with `.venv/bin/python`, not the system or conda Python.
- `.venv` was built from a conda env called `py312-base` (Python 3.12). Do not delete that conda env or the venv breaks.
- espeak-ng is installed with brew and is not listed in requirements.txt.
- `HF_TOKEN` lives in `.env`. Load it from there. Never print it, log it, or hardcode it.
- When adding a package, add it to requirements.txt.

## v1 scope

In: upload a recording, transcript with speaker labels, decisions, action items, open questions, unassigned items flagged, table plus CSV download.

Out: live meeting bots, Sheets or Notion sync, non-English meetings, reminders.

## Testing

Each test meeting in `test_data/` has an answer key JSON. Score output against it: action items caught, correct owners, invented items. The planted traps (deadline change, reassignment, cancelled item, vague owner, non-decision) are the real test.

## How to work with me

- I am learning. After building something, explain the key parts in plain language.
- Keep changes small and focused on the task I asked for.
- Ask before installing large packages or making big changes to the environment.
- Suggest a git commit when something new starts working.
