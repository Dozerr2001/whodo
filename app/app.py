"""MeetingMate web app: upload a meeting recording, get decisions, action items and open questions.

Run locally:  .venv/bin/python app/app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetingmate import config  # noqa: E402
from meetingmate.pipeline import Progress, estimate_seconds, load_cached, name_speakers, run  # noqa: E402
from meetingmate.transcribe import pick_device  # noqa: E402

load_dotenv(ROOT / ".env")  # does nothing on a Space, where HF_TOKEN is a secret env var

SAMPLE_AUDIO = ROOT / "test_data" / "meeting_01.wav"
SAMPLE_TRANSCRIPT = ROOT / "test_data" / "meeting_01_transcript_cc1_4spk.json"
SAMPLE_EXTRACTION = ROOT / "test_data" / "meeting_01_extracted_cc1_4spk_participants.json"
SAMPLE_PARTICIPANTS = "Priya, Rahul, Meera, Arjun"
SAMPLE_SPEAKERS = 4

PRIVACY = """\
**Privacy.** Your recording is processed on this server. The transcript text is then sent to a third-party \
language model through Hugging Face Inference Providers to pull out the action items. Uploaded files are \
deleted from the server within about an hour, and MeetingMate keeps no copy of your audio or transcript. \
Don't upload confidential meetings to a public demo."""

ACTION_HEADERS = ["Task", "Owner", "Deadline"]
STATUS_IDLE = "Upload a recording (wav, mp3, flac or ogg) and press **Analyze**, or try the sample meeting."


def fmt_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


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


def parse_names(text):
    return [n.strip() for n in (text or "").split(",") if n.strip()] or None


def transcript_text(segments):
    return "\n".join(f"[{fmt_duration(s['start'])}] {s['speaker']}: {s['text']}" for s in segments)


def build_csv(ex):
    rows = (
        [("action item", i["task"], i["owner"], i["deadline"] or "") for i in ex["action_items"]]
        + [("decision", d, "", "") for d in ex["decisions"]]
        + [("unassigned", i["task"], "", i["deadline"] or "") for i in ex["unassigned_items"]]
        + [("open question", q, "", "") for q in ex["open_questions"]]
    )
    path = Path(tempfile.mkdtemp()) / "meetingmate_results.csv"
    pd.DataFrame(rows, columns=["type", "text", "owner", "deadline"]).to_csv(path, index=False)
    return str(path)


# Order of the outputs wired to both buttons.
def outputs_from(status, banner, result):
    """Turn a Result into values for the output components, in the order OUTPUTS lists them."""
    if result is None:
        return status, banner, [], [], [], [], "", None
    ex = result.extraction
    segments = name_speakers(result.segments, ex["speaker_map"])
    return (
        status,
        banner,
        [[i["task"], i["owner"], i["deadline"] or "—"] for i in ex["action_items"]],
        [[d] for d in ex["decisions"]],
        [[i["task"], i["deadline"] or "—"] for i in ex["unassigned_items"]],
        [[q] for q in ex["open_questions"]],
        transcript_text(segments),
        build_csv(ex),
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


def stream_run(audio_path, participants, num_speakers, banner):
    """Run the pipeline and yield UI updates. The status changes on every heartbeat; the tables only at the end."""
    token = get_token()
    secs = audio_seconds(audio_path)
    check_length(secs)
    device = pick_device()
    yield outputs_from(f"⏳ Starting… estimated {fmt_minutes(estimate_seconds(secs, device))}.", banner, None)
    result = None
    try:
        for event in run(audio_path, token, parse_names(participants), int(num_speakers) if num_speakers else None):
            if isinstance(event, Progress):
                yield (progress_status(event, secs, device),) + (gr.skip(),) * 7
            else:
                result = event
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {str(e).replace(token, '***')}"[:500])
    yield outputs_from(finish_status(result), banner, result)


def analyze(audio_path, participants, num_speakers):
    if not audio_path:
        raise gr.Error("Please upload a recording first.")
    yield from stream_run(audio_path, participants, num_speakers, "")


def analyze_sample():
    """Returns the sample inputs too, so the form shows what the sample used."""
    fill = (SAMPLE_PARTICIPANTS, SAMPLE_SPEAKERS)
    if config.SAMPLE_MODE == "cached":
        banner = (
            "📦 **Pre-computed results.** These were produced earlier by running the full pipeline on the sample "
            "meeting. Nothing was processed just now. Upload your own recording to run it live."
        )
        result = load_cached(SAMPLE_TRANSCRIPT, SAMPLE_EXTRACTION)
        yield fill + outputs_from("Showing the saved sample results.", banner, result)
        return
    if not SAMPLE_AUDIO.exists():
        raise gr.Error(f"Sample recording not found at {SAMPLE_AUDIO}.")
    banner = "▶️ **Live run** on the sample meeting."
    skip = (gr.skip(), gr.skip())  # after the first update the inputs are already filled in
    for n, values in enumerate(stream_run(str(SAMPLE_AUDIO), SAMPLE_PARTICIPANTS, SAMPLE_SPEAKERS, banner)):
        yield (fill if n == 0 else skip) + values


with gr.Blocks(title="MeetingMate", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# MeetingMate\nUpload a meeting recording and get the decisions, action items (with owners and deadlines) and open questions.")
    gr.Markdown(PRIVACY)

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload"], type="filepath", label="Meeting recording")
            estimate = gr.Markdown()
            participants = gr.Textbox(
                label="Participant names (optional)",
                placeholder="Priya, Rahul, Meera, Arjun",
                info="Comma-separated. Fixes misspelled names and helps assign owners.",
            )
            num_speakers = gr.Number(
                label="Number of speakers (optional)", precision=0, minimum=1, maximum=20, value=None,
                info="If you know it, speaker labels are noticeably more accurate.",
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                sample_btn = gr.Button("Try the sample meeting")
        with gr.Column():
            status = gr.Markdown(STATUS_IDLE)
            banner = gr.Markdown()

    gr.Markdown("### Action items")
    actions = gr.Dataframe(headers=ACTION_HEADERS, interactive=False, wrap=True)
    gr.Markdown("### Decisions")
    decisions = gr.Dataframe(headers=["Decision"], interactive=False, wrap=True)
    gr.Markdown("### Unassigned items\nMentioned as needing to be done, but nobody took them on.")
    unassigned = gr.Dataframe(headers=["Task", "Deadline"], interactive=False, wrap=True)
    gr.Markdown("### Open questions")
    questions = gr.Dataframe(headers=["Question"], interactive=False, wrap=True)
    csv_file = gr.File(label="Download everything as CSV")
    with gr.Accordion("Full transcript", open=False):
        transcript = gr.Textbox(lines=15, max_lines=30, interactive=False, show_label=False, buttons=["copy"])

    OUTPUTS = [status, banner, actions, decisions, unassigned, questions, transcript, csv_file]

    audio.change(estimate_note, audio, estimate)
    analyze_btn.click(analyze, [audio, participants, num_speakers], OUTPUTS)
    sample_btn.click(analyze_sample, None, [participants, num_speakers] + OUTPUTS)

demo.queue(max_size=5, default_concurrency_limit=1)  # one job at a time; the rest wait in line

if __name__ == "__main__":
    demo.launch()
