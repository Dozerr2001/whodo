# WhoDo

**Turn a meeting recording into who does what, by when.**

> **Live demo:** _add the Hugging Face Space link here after deploying._
> **Screenshot:** _add a screenshot of the results page here after deploying._

Upload a meeting recording, get back a summary, the decisions made, the action items with owners and deadlines, and the questions still left open.

## The problem

Meetings end, and the stuff that actually matters, who is doing what by when, lives in someone's scribbled notes or nowhere at all. Existing note takers give you a summary, but a summary is not a task list. Somebody still has to read it, pull out the action items, and chase people.

## Who it's for

Chiefs of staff, project managers, and team leads who run recurring meetings and are on the hook for follow-through.

## How it works

WhoDo runs on three pieces:

1. **Whisper** (`openai/whisper-large-v3-turbo`) turns speech into text. *What was said.*
2. **pyannote** (`speaker-diarization-community-1`) figures out who spoke when. *Who said it.*
3. **An LLM** (`openai/gpt-oss-120b`, through Groq with Hugging Face Inference Providers as the fallback) reads the labeled transcript and pulls out decisions, action items, owners, and deadlines. *What it means.*

Pipeline: Audio → Whisper + pyannote → labeled transcript → LLM → action item table

## What v1 does

- Takes an uploaded meeting recording, with optional speaker count and participant names to improve accuracy
- Transcribes it and labels who said what, with the full transcript available to read and copy
- Returns decisions, action items (task, owner, deadline), and open questions
- Flags action items with no clear owner as unassigned instead of guessing
- Shows a **supporting quote** for every action item and decision, with who said it and when (a timestamp in the recording), checked against the transcript in code
- Writes a **meeting summary**: an overview plus the main topics with timestamps, the options considered, and the reasons behind each outcome
- Turns deadlines like "by this Friday" into **real calendar dates** from the meeting date, in code rather than by the LLM, and marks dates that need a second look
- Has a **Stop button** that cancels a run mid-step
- Has an **example meeting** to try without a recording of your own
- Shows results as tables in the app, with a CSV download

## Out of scope for v1

- Joining live calls as a bot
- Google Sheets or Notion sync (planned for v1.5)
- Non-English meetings
- Reminders and follow-ups

## How we know it works

Tested against `meeting_01`, a synthetic meeting where the correct answers are known (see [Test data](#test-data)). Scored with `scripts/score.py`:

| Setup | Action items caught | Owners right | Deadlines right | Invented items | Traps passed |
|---|---|---|---|---|---|
| Speaker count hint only | 5/5 | 2/5 | 5/5 | 0 | 4/5 |
| Speaker count hint + participant names | 5/5 | 5/5 | 5/5 | 0 | 5/5 |

With participant names it also found all 3 expected decisions and the open question, and the summary did not contradict the extraction. Each row is one saved run on one synthetic meeting, and LLM output varies between runs, so treat these numbers as rough.

**A real meeting.** I also ran a real 19-minute GitLab meeting recording (someone else's, so it is not in this repo). It has no answer key, so its results are not scored. On a Mac (Apple GPU) the full pipeline took **633 seconds** (10.6 minutes) for 1,149 seconds of audio: Whisper 498 s, pyannote 123 s, the LLM 12 s. The request was too large for Groq's free tier, so the LLM step fell back to Hugging Face (see Known issues). Running extraction again later on that transcript, without participant names, gave 5 action items, 0 decisions, 4 unassigned items and 5 open questions, with two of the four speakers left unnamed.

**Speed on the 3-minute meeting.** Nine of ten logged runs took 83 to 103 seconds. The tenth took 316 seconds because pyannote alone took 241.

## Project structure

```
WhoDo/
├── app/
│   └── app.py            the Gradio app
├── whodo/                the pipeline as importable code
│   ├── transcribe.py     Whisper + pyannote
│   ├── extract.py        LLM: decisions, action items, open questions
│   ├── summarize.py      a separate LLM call for the summary
│   ├── pipeline.py       runs both, with progress events and a timing log
│   ├── worker.py         runs Whisper and pyannote in a process that Stop can kill
│   ├── dates.py          deadline phrase -> calendar date
│   ├── quotes.py         finds the transcript line behind a quote
│   ├── render.py         HTML tables and summary
│   └── config.py         local vs Space settings
├── scripts/              command-line wrappers, scoring, and the test-audio generator
├── tests/                plain-assert checks (dates, quotes, rendering, LLM errors, Stop)
├── test_data/            meeting scripts, audio, speaker timelines, answer keys, saved results
├── requirements.txt
├── LICENSE
├── .gitignore
└── .env                  Hugging Face token and Groq key (never committed)
```

## Setup

Needs macOS with Homebrew, and Python 3.12.

1. Install espeak-ng (used by the text-to-speech model that makes the test audio):
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
4. On Hugging Face, accept the terms for the gated pyannote model `pyannote/speaker-diarization-community-1`. To use the older pipeline with `--diarization-model 3.1`, also accept `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
5. Create a read-access token on Hugging Face (Settings → Access Tokens), and optionally a free [Groq](https://console.groq.com) API key, and add them to a `.env` file in the project root:
   ```
   HF_TOKEN=your_token_here
   GROQ_API_KEY=your_key_here
   ```
   Without a Groq key, every LLM call goes through Hugging Face Inference Providers, which uses your HF credits.

## Usage

Run the app:

```
python app/app.py
```

Or use the pieces from the command line:

```
python scripts/transcribe.py recording.mp3 --num-speakers 4
python scripts/extract.py test_data/meeting_01_transcript.json --participants "Priya,Rahul,Meera,Arjun"
python scripts/score.py        # score saved results against the answer key
python tests/test_dates.py     # one of the fast checks in tests/
```

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

## What I learned

- **Telling pyannote the speaker count mattered most.** Checked word by word, speaker accuracy on `meeting_01` rose from 79.0% to 89.6% with `--num-speakers 4`, and swapping to a newer diarization model made no real difference. My first check, line by line, said 92.3%, which was too generous because a line with two speakers counted as right if one dominated it.
- **The summary contradicted the action items.** The summary is its own LLM call (asking one call to do both made each worse), and it could hand a task to the wrong person or call a listed task dropped. The fix was giving the summary call the finished extraction as established facts, and adding a check in `scripts/score.py` that flags summary sentences which disagree with it. The check is a word-matching heuristic, so it still needs a human eye.
- **Lower reasoning effort looked like a free win and wasn't.** `gpt-oss` spends most of its output tokens thinking, so I tried `reasoning_effort="low"` to fit Groq's limit. Over three runs on `meeting_01` every run got worse somewhere: a missed decision, a failed trap, owners at 3/5, and summaries contradicting the extraction. I left it off and kept the setting (`WHODO_GROQ_REASONING_EFFORT`) for retesting.
- **Groq's free tier shaped the product.** A request there is capped at 8,000 tokens. The first 10 minutes of the GitLab recording fit; the full 19 minutes asked for about 9,700 and was refused. So the app estimates the size before calling Groq, and the public Space accepts recordings up to 8 minutes to leave a buffer.

## Known issues

- **pyannote merges similar voices and fast turns.** On `meeting_01`, pyannote finds 3 speakers instead of 4 and merges Meera into Priya. Checked word by word with `scripts/eval_diarization.py`, the unhinted result is **79.0%** of words with the right speaker, and only **43.9%** in the busy stretch (94-122s) where Priya and Meera trade short turns. Wrong speakers mean wrong owners in the action item table.
- **Telling it the speaker count helps a lot.** With `--num-speakers 4`, word accuracy rises to 89.6% (81.8% in the busy stretch). The app has an optional Speakers field for this.
- **`pyannote/speaker-diarization-community-1` is no better here.** Same 79.0% unhinted and 90.4% with 4 speakers, a difference of about 3 words, which is within run-to-run noise. `transcribe.py` supports both through `--diarization-model`.
- **Owner accuracy depended on the speaker labels and names, not on the LLM.** Reading the transcripts alone, gpt-oss-120b caught all 5 action items but got only 2 of 5 owners right. Priya's name is never spoken in the meeting, so it can't be inferred, and Whisper writes "Meera" as "Mira".
- **A participant list fixes most of that.** Running `extract.py` with `--participants "Priya,Rahul,Meera,Arjun"` (or filling in the app's Participant names field) restricts names to that list, corrects misspellings, and uses cues like "Meera, design?" to correct speaker labels. On the community-1 transcript it raised owners from 2/5 to 5/5 and all 5 traps passed; on the older 3.1 transcript it reached 4/5 owners. One run each on one synthetic meeting.
- **Meetings over roughly 10 minutes exceed Groq's free tier per-request limit and fall back to HF.** Groq's free tier allows 8,000 tokens per request for `gpt-oss-120b`, and a request is the transcript plus the instructions plus room for the answer. The first 10 minutes of a real recording fit and completed on Groq; the full 19 minutes asked for about 9,700 and was refused with a 413. The app estimates the size first and goes straight to Hugging Face Inference Providers when it is over, which uses HF credits. The public Space accepts recordings up to 8 minutes, a buffer under that limit (change it with `WHODO_MAX_AUDIO_MINUTES`; locally there is no cap). Planned fix: split long transcripts into chunks.

## Status

- [x] Synthetic meeting generator (Kokoro text-to-speech)
- [x] Transcription with speaker labels (Whisper + pyannote)
- [x] Action item extraction (LLM)
- [x] Meeting summary
- [x] Scoring against the answer key
- [x] App interface (Gradio)
- [ ] Deploy to Hugging Face Spaces
- [ ] Split long transcripts into chunks
