# WhoDo - context for Claude

## What this project is

WhoDo takes a meeting recording and returns decisions, action items (task, owner, deadline), and open questions. It is a learning project: the goal is to learn how to build and ship AI products using models from Hugging Face, ending with a public app on Hugging Face Spaces.

## How it works

1. **Whisper** turns speech into text. *What was said.*
2. **pyannote** figures out who spoke when. *Who said it.*
3. **An LLM** reads the labeled transcript and pulls out decisions, action items, owners, and deadlines. *What it means.*

Pipeline: Audio → Whisper + pyannote → labeled transcript → LLM → action item table

## Project structure

- `whodo/` the pipeline as importable code: `transcribe.py` (Whisper + pyannote), `extract.py` (LLM), `summarize.py` (separate LLM call for the meeting summary; kept apart because adding it to the extraction prompt made extraction worse), `pipeline.py` (both, with progress events and a timing log), `worker.py` (Whisper and pyannote run in a separate process so Stop can kill them mid-step; the LLM call can only be abandoned), `config.py` (local vs Space settings), `dates.py` (deadline phrase -> calendar date, in code), `quotes.py` (finds the transcript line behind an LLM quote), `render.py` (HTML tables and summary line)
- `scripts/` command-line wrappers around `whodo/` (transcribe.py, extract.py), plus score.py, eval_diarization.py and generate_meeting.py (makes synthetic test audio with Kokoro TTS)
- `tests/` plain-assert checks for dates and quotes: `.venv/bin/python tests/test_dates.py`; `tests/test_stop.py` stops a real run at each stage (slow, uses real models and the LLM)
- `test_data/` meeting scripts, generated audio (`meeting_01.mp3` is the small copy the app plays and is committed; `*.wav` is gitignored), speaker timelines, and answer keys
- `app/app.py` the Gradio app. Run with `.venv/bin/python app/app.py`. On a Space (`SPACE_ID` set) it caps audio at 8 minutes and shows cached sample results; override with `WHODO_MAX_AUDIO_MINUTES` and `WHODO_SAMPLE_MODE=live|cached`.
- `logs/timings.jsonl` (gitignored) real per-step run times, used to check the wait estimates in `config.py`

## Environment notes

- Always run Python with `.venv/bin/python`, not the system or conda Python.
- `.venv` was built from a conda env called `py312-base` (Python 3.12). Do not delete that conda env or the venv breaks.
- espeak-ng is installed with brew and is not listed in requirements.txt.
- `HF_TOKEN` lives in `.env`. Load it from there. Never print it, log it, or hardcode it.
- When adding a package, add it to requirements.txt.

## Best config so far (from meeting_01 experiments)

- **Diarization:** `pyannote/speaker-diarization-community-1` with a speaker count hint (`--num-speakers`). Word-level accuracy is about 90% with the hint and 79% without. community-1 and 3.1 are about equal, so the hint is what matters.
- **Participants:** pass `--participants "A,B,C"` to `extract.py`. It fixes misspelled names ("Mira" for "Meera"), names people who are never addressed, and corrects speaker label errors. Owners went from 2/5 to 4/5 and all 5 traps passed.
- **LLM:** `openai/gpt-oss-120b` through Groq (`GROQ_API_KEY` in `.env`), falling back to HF Inference Providers. It beat `Qwen/Qwen3-235B-A22B-Instruct-2507`, which missed a task.
- **Groq free tier is 8,000 tokens a minute, and gpt-oss's hidden thinking counts.** One 19-minute extraction used about 9,700 tokens (5.7K prompt, 3.6K of 4K output was thinking), so the summary call right after hits a 429. The fix is waiting for the window to reset (`x-ratelimit-reset-tokens`, up to 75s). `max_tokens` does not count against the limit. `WHODO_GROQ_REASONING_EFFORT=low` cuts the thinking but made meeting_01 worse in 3 of 3 runs, so it is off by default. A single Groq request over 8,000 tokens gets a 413 that waiting cannot fix, so `ask_any` estimates the size first (prompt chars / 3.85 plus 4,000 for the answer, which matched Groq's own "Requested 9724") and goes straight to HF when it is over (`WHODO_GROQ_TOKEN_LIMIT` raises the limit for a paid plan). The first 10 minutes of the GitLab recording fit (about 7,770, thin margin) and the full 19 do not (about 9,730). Planned fix for long meetings: split the transcript into chunks.
- **Measure speakers word by word** with `scripts/eval_diarization.py`. Line-level accuracy hid merged turns and read about 13 points too high.
- **The LLM copies deadline phrases as spoken; code turns them into dates** using the meeting date (the sample uses a fixed one, 2026-10-26). Quotes are checked against the transcript in code, and speaker and time come from the matched line.
- **Dev mode saves credits.** `WHODO_DEV_LLM=saved` keeps the real Whisper and pyannote steps but uses the saved sample results instead of calling the LLM (`outage` and `summary-outage` simulate a failed LLM or summary, to see the error messages). Ignored on a Space. A failed LLM call shows "The AI service is temporarily unavailable. Try the example instead."; a failed summary shows a notice above the results.
- **Keep the extraction prompt general.** Never put anything from the answer key in it. The trap keywords live in `scripts/score.py`.
- One run per setup on one synthetic meeting, and LLM output varies between runs, so treat these numbers as rough.

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
