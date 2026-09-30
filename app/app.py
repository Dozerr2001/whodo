"""MeetingMate web app: upload a meeting recording, get decisions, action items and open questions.

Run locally:  .venv/bin/python app/app.py
"""

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import gradio as gr
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetingmate import config  # noqa: E402
from meetingmate import render  # noqa: E402
from meetingmate.extract import name_speakers  # noqa: E402
from meetingmate.pipeline import Progress, estimate_seconds, load_cached, run  # noqa: E402
from meetingmate.render import fmt_duration  # noqa: E402
from meetingmate.transcribe import pick_device  # noqa: E402

load_dotenv(ROOT / ".env")  # does nothing on a Space, where HF_TOKEN is a secret env var

SAMPLE_AUDIO = ROOT / "test_data" / "meeting_01.wav"  # full quality, local only (*.wav is gitignored)
SAMPLE_PLAYER = ROOT / "test_data" / "meeting_01.mp3"  # small copy that ships with the app
SAMPLE_TRANSCRIPT = ROOT / "test_data" / "meeting_01_transcript_cc1_4spk.json"
SAMPLE_EXTRACTION = ROOT / "test_data" / "meeting_01_extracted_cc1_4spk_participants.json"
SAMPLE_PARTICIPANTS = "Priya, Rahul, Meera, Arjun"
SAMPLE_SPEAKERS = 4
SAMPLE_MEETING_DATE = date(2026, 10, 26)  # a Monday; fixed so "this Friday" resolves the same way for every visitor
SAMPLE_DESCRIPTION = (
    "A 3-minute product meeting with 4 people, including a reassigned task, a cancelled idea and a moved deadline."
)
INPUT_CSS = """
.mm-col { max-width: 720px; width: 100%; margin: 0 auto; gap: 12px; }
.mm-small, .mm-small p { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.mm-link { background: none; border: none; box-shadow: none; text-decoration: underline; color: var(--link-text-color); padding: 0; width: fit-content; align-self: flex-start; min-width: 0; }
.mm-group-title { padding: 8px 12px 0; }
"""
TODAY_JS = "() => new Date().toLocaleDateString('en-CA')"  # the visitor's own date, as YYYY-MM-DD

PRIVACY_SHORT = (
    "🔒 Your recording is processed on this server and the transcript is sent to a third-party language model. "
    "Don't upload confidential meetings."
)
PRIVACY = """\
**Privacy.** Your recording is processed on this server. The transcript text is then sent to a third-party \
language model through Hugging Face Inference Providers to pull out the action items. Uploaded files are \
deleted from the server within about an hour, and MeetingMate keeps no copy of your audio or transcript. \
Don't upload confidential meetings to a public demo."""

UNASSIGNED_HEAD = "### Unassigned items\nMentioned as needing to be done, but nobody took them on."
N_OUTPUTS = 11  # status, banner, results column, summary, three table blocks, unassigned heading, questions, transcript, csv


def fmt_minutes(seconds):
    """'about 6 min' for an estimate."""
    return "under 1 min" if seconds < 60 else f"about {round(seconds / 60)} min"


def audio_seconds(path):
    try:
        return sf.info(path).duration
    except Exception:
        raise gr.Error("Couldn't read that file. Please use wav, mp3, flac or ogg.")


def check_length(seconds):
    limit = config.MAX_AUDIO_MINUTES
    if limit and seconds > limit * 60:
        raise gr.Error(f"This demo accepts recordings up to {limit:g} minutes; yours is {seconds / 60:.0f} minutes.")


def estimate_note(path):
    """Shown as soon as a file is uploaded."""
    if not path:
        return ""
    try:
        seconds = audio_seconds(path)
        check_length(seconds)
    except gr.Error as e:
        return f"⚠️ {e.message}"
    device = pick_device()
    return (
        f"Recording length: **{fmt_duration(seconds)}**. Estimated wait on this machine ({device}): "
        f"**{fmt_minutes(estimate_seconds(seconds, device))}**. Keep this tab open while it runs."
    )


def get_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise gr.Error("HF_TOKEN is not set. Add it to .env locally, or as a secret in the Space settings.")
    return token


def parse_meeting_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise gr.Error("Please enter the meeting date as YYYY-MM-DD.")


def parse_speakers(text):
    """Blank means unknown. Anything else must be a whole number from 1 to 20."""
    text = str(text or "").strip()
    if not text:
        return None
    if not text.isdigit() or not 1 <= int(text) <= 20:
        raise gr.Error("Number of speakers must be a whole number from 1 to 20, or blank if you're not sure.")
    return int(text)


def parse_names(text):
    return [n.strip() for n in (text or "").split(",") if n.strip()] or None


def transcript_text(segments):
    return "\n".join(f"[{fmt_duration(s['start'])}] {s['speaker']}: {s['text']}" for s in segments)


CSV_COLUMNS = [
    "type", "text", "owner", "deadline_as_spoken", "deadline_date", "date_needs_check",
    "quote", "quote_speaker", "quote_time",
]


def csv_row(kind, text, owner="", deadline="", anchor=None, source=None):
    date_text, phrase, check = render.deadline_parts(deadline, anchor)
    source = source or {}
    return (
        kind, text, owner, phrase, date_text, "yes" if check else "",
        source.get("quote", ""), source.get("speaker", ""), fmt_duration(source["time"]) if source else "",
    )


def build_csv(ex, anchor):
    rows = (
        [csv_row("action item", i["task"], render.owner_label(i["owner"]), i["deadline"], anchor, i["source"]) for i in ex["action_items"]]
        + [csv_row("decision", d["decision"], source=d["source"]) for d in ex["decisions"]]
        + [csv_row("unassigned", i["task"], render.NEEDS_OWNER) for i in ex["unassigned_items"]]
        + [csv_row("open question", q) for q in ex["open_questions"]]
    )
    path = Path(tempfile.mkdtemp()) / "meetingmate_results.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)
    return str(path)


# Order of the outputs wired to both buttons.
def outputs_from(status, banner, result, anchor):
    """Turn a Result into values for the output components, in the order OUTPUTS lists them.

    With no result the results section is hidden, so a new run never shows the previous run's tables.
    """
    if result is None:
        return status, banner, gr.update(visible=False), "", "", "", "", "", "", "", None
    ex = result.extraction
    segments = name_speakers(result.segments, ex["speaker_map"])
    return (
        status,
        banner,
        gr.update(visible=True),
        f'<div class="mm-summary">{render.summary_line(ex)}</div>',
        render.action_table(ex, anchor),
        render.decisions_table(ex),
        UNASSIGNED_HEAD if ex["unassigned_items"] else "",  # no heading when there is nothing under it
        render.unassigned_table(ex),
        render.questions_table(ex),
        transcript_text(segments),
        build_csv(ex, anchor),
    )


def progress_status(p: Progress, audio_secs, device):
    total = estimate_seconds(audio_secs, device)
    return (
        f"⏳ **Step {p.step} of 3: {p.label}…** (this step: {fmt_duration(p.elapsed)}). "
        f"Total estimate {fmt_minutes(total)}. Keep this tab open."
    )


def finish_status(result):
    warning = result.extraction.get("stray_names")
    note = f" ⚠️ Names outside the participant list: {', '.join(warning)}." if warning else ""
    if result.cached:
        return "Done." + note
    return f"✅ Done in {fmt_duration(result.timings['total'])}." + note


def stream_run(audio_path, participants, num_speakers, anchor, banner):
    """Run the pipeline and yield UI updates. The status changes on every heartbeat; the tables only at the end."""
    token = get_token()
    secs = audio_seconds(audio_path)
    check_length(secs)
    device = pick_device()
    yield outputs_from(f"⏳ Starting… estimated {fmt_minutes(estimate_seconds(secs, device))}.", banner, None, anchor)
    result = None
    try:
        for event in run(audio_path, token, parse_names(participants), num_speakers):
            if isinstance(event, Progress):
                yield (progress_status(event, secs, device),) + (gr.skip(),) * (N_OUTPUTS - 1)
            else:
                result = event
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {str(e).replace(token, '***')}"[:500])
    yield outputs_from(finish_status(result), banner, result, anchor)


def analyze(audio_path, participants, num_speakers, meeting_date):
    if not audio_path:
        raise gr.Error("Please upload a recording first.")
    yield from stream_run(audio_path, participants, parse_speakers(num_speakers), parse_meeting_date(meeting_date), "")


def on_upload(path, names, num):
    """A new recording replaces the example, so drop the example's inputs but keep anything the user typed."""
    return estimate_note(path), ("" if names == SAMPLE_PARTICIPANTS else names), ("" if str(num).strip() == str(SAMPLE_SPEAKERS) else num)


def analyze_sample():
    """Fills the form and the audio player with the example too, so visitors can see and hear what it used."""
    fill = (SAMPLE_PARTICIPANTS, str(SAMPLE_SPEAKERS), SAMPLE_MEETING_DATE.isoformat(), str(SAMPLE_PLAYER), "")
    if config.SAMPLE_MODE == "cached":
        banner = (
            "📦 **Pre-computed results.** These were produced earlier by running the full pipeline on the sample "
            "meeting. Nothing was processed just now. Upload your own recording to run it live."
        )
        result = load_cached(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
        yield fill + outputs_from("Showing the saved sample results.", banner, result, SAMPLE_MEETING_DATE)
        return
    source = SAMPLE_AUDIO if SAMPLE_AUDIO.exists() else SAMPLE_PLAYER  # prefer the lossless original when it's here
    if not source.exists():
        raise gr.Error(f"Sample recording not found at {source}.")
    banner = "▶️ **Live run** on the sample meeting."
    skip = (gr.skip(),) * len(fill)  # after the first update the inputs are already filled in
    for n, values in enumerate(
        stream_run(str(source), SAMPLE_PARTICIPANTS, SAMPLE_SPEAKERS, SAMPLE_MEETING_DATE, banner)
    ):
        yield (fill if n == 0 else skip) + values


with gr.Blocks(title="MeetingMate", delete_cache=(3600, 3600)) as demo:
    with gr.Column(elem_classes="mm-col"):
        gr.Markdown("# MeetingMate\nTurn a meeting recording into a task list: who owes what, by when.")
        audio = gr.Audio(sources=["upload"], type="filepath", label="Meeting recording")
        estimate = gr.Markdown()
        sample_btn = gr.Button("No recording handy? See an example", variant="secondary", size="sm", elem_classes="mm-link")
        gr.Markdown(SAMPLE_DESCRIPTION, elem_classes="mm-small")

        with gr.Group():
            gr.Markdown("**Meeting details** (optional, but improves accuracy)", elem_classes="mm-group-title")
            with gr.Row():
                meeting_date = gr.DateTime(
                    label="Meeting date", include_time=False, type="string", value=date.today().isoformat(),
                    info="Deadlines like \"this Friday\" are turned into dates from this day.", scale=3,
                )
                num_speakers = gr.Textbox(  # not gr.Number: it shows 0 (and a red border) when empty
                    label="Speakers", value="", max_length=2, info="Leave blank if unsure", scale=1, min_width=120,
                )
            participants = gr.Textbox(
                label="Participant names",
                placeholder="Priya, Rahul, Meera, Arjun",
                info="Comma-separated. Fixes misspelled names and helps assign owners.",
            )

        analyze_btn = gr.Button("Analyze meeting", variant="primary", size="lg")
        status = gr.Markdown()
        banner = gr.Markdown()

        gr.Markdown(PRIVACY_SHORT, elem_classes="mm-small")
        with gr.Accordion("Full privacy notice", open=False):
            gr.Markdown(PRIVACY, elem_classes="mm-small")

    with gr.Column(visible=False, elem_classes="mm-col") as results:  # shown once a run finishes
        summary = gr.HTML()
        gr.Markdown("### Action items")
        actions = gr.HTML()
        gr.Markdown("### Decisions")
        decisions = gr.HTML()
        unassigned_head = gr.Markdown()
        unassigned = gr.HTML()
        gr.Markdown("### Open questions")
        questions = gr.HTML()
        csv_file = gr.DownloadButton("Download CSV", variant="secondary")
        with gr.Accordion("Full transcript", open=False):
            transcript = gr.Textbox(lines=15, max_lines=30, interactive=False, show_label=False, buttons=["copy"])

    OUTPUTS = [
        status, banner, results, summary, actions, decisions, unassigned_head, unassigned, questions, transcript, csv_file,
    ]

    demo.load(None, None, meeting_date, js=TODAY_JS)  # the server's clock may be in another time zone
    audio.upload(on_upload, [audio, participants, num_speakers], [estimate, participants, num_speakers]).then(
        None, meeting_date, meeting_date,
        js=f"(d) => d === '{SAMPLE_MEETING_DATE.isoformat()}' ? ({TODAY_JS})() : d",
    )
    audio.clear(lambda: "", None, estimate)
    analyze_btn.click(analyze, [audio, participants, num_speakers, meeting_date], OUTPUTS)
    sample_btn.click(analyze_sample, None, [participants, num_speakers, meeting_date, audio, estimate] + OUTPUTS)

demo.queue(max_size=5, default_concurrency_limit=1)  # one job at a time; the rest wait in line

if __name__ == "__main__":
    demo.launch(css=render.CSS + INPUT_CSS, allowed_paths=[str(SAMPLE_PLAYER)])  # lets the player serve the sample mp3, and nothing else in test_data
