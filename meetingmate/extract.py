"""Pull decisions, action items and open questions out of a labeled transcript.

Asks an open model on Hugging Face Inference Providers to return structured JSON.
The command-line entry point is scripts/extract.py.
"""

import json
import re
from pathlib import Path

from huggingface_hub import InferenceClient
from pydantic import BaseModel, ConfigDict, ValidationError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "openai/gpt-oss-120b"
UNKNOWN = "unknown"

INSTRUCTIONS = f"""\
You read the transcript of a meeting and extract what matters. Each line looks like
"[start seconds] SPEAKER_NN: text". The speaker labels are anonymous and can be wrong:
a speaker with a similar voice may have been merged into another label, so use the
content of what is said, not only the label, when deciding who said or committed to something.

Return a JSON object with these fields.

1. speaker_map: one entry per speaker label in the transcript, mapping it to a real name.
   Infer names from how people address each other ("Thanks, Sam", "Sam, where are we on...").
   Use "{UNKNOWN}" if the transcript does not make the name clear. Never guess from a label's number.

2. decisions: things the group actually settled. Something is a decision only if the group
   clearly agreed or the person in charge clearly chose. Ideas floated, options compared, and
   topics postponed are not decisions. If a decision is changed later, report only the final version.

3. action_items: tasks that someone committed to do. Each has task, owner and deadline.
   - Count it only if a specific person accepted or volunteered for the task. A suggestion, a
     wish, or "we should..." is not a commitment.
   - If a task is offered to one person and someone else takes it, the owner is the person who took it.
   - If a task is cancelled or dropped later in the meeting, leave it out entirely.
   - If a deadline is changed, use the latest one. Keep the deadline in the speaker's own words
     (for example "Friday" or "end of day"); do not invent calendar dates you cannot know.
   - If a person committed but gave no deadline, use null.
   - owner is a real name from speaker_map, or "{UNKNOWN}" if you cannot tell.

4. unassigned_items: tasks that were mentioned as needing to be done, but nobody took them on
   (for example "someone should check..."). Do not repeat tasks that appear in action_items,
   and do not include cancelled tasks. Each has task and deadline (null if none).

5. open_questions: questions or issues that were raised and left unresolved, including topics
   explicitly postponed. Do not include questions that were answered later.

General rules:
- Use only what is in the transcript. Do not invent items, names, or deadlines.
- One entry per real item. Do not split one task into several or repeat it.
- Write each task or decision as a short, self-contained sentence.
- Respond with the JSON object only.
"""


PARTICIPANT_RULES = """\

The attendees of this meeting are: {names}.
- Every name you output (in speaker_map and as an owner) must be one of these names, or "{unknown}".
  Never output any other name.
- The transcript was produced by speech recognition, so names may be misspelled or misheard.
  Match a name in the transcript to the closest attendee name.
- Speaker labels come from a voice-separation step that makes mistakes. Use conversational cues
  to work out who is really speaking, and let those cues override the label when they conflict:
  * When someone is addressed by name ("Sam, can you...?"), the next reply usually comes from
    that person, and the person addressed is not the one asking.
  * A label can contain lines from two people who traded short turns. Look for a question
    followed by its answer, or a request followed by acceptance, inside one label, and credit
    each line to the person the conversation points to.
  * A person who says "I'll do it" or "I'll write it myself" owns that task, whichever
    label that sentence carries.
  * If every attendee but one has been matched to a label or a cue, the leftover attendee
    most likely owns the unmatched voice.
- If the cues still do not settle it, use "{unknown}" instead of guessing.
"""


class ExtractionError(Exception):
    """The model never returned valid output."""


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpeakerName(Model):
    speaker: str
    name: str


class Decision(Model):
    decision: str


class ActionItem(Model):
    task: str
    owner: str
    deadline: str | None


class UnassignedItem(Model):
    task: str
    deadline: str | None


class Extraction(Model):
    speaker_map: list[SpeakerName]
    decisions: list[Decision]
    action_items: list[ActionItem]
    unassigned_items: list[UnassignedItem]
    open_questions: list[str]


def format_transcript(segments):
    return "\n".join(f"[{s['start']:.0f}s] {s['speaker']}: {s['text']}" for s in segments)


def parse_json(text):
    """Load JSON, tolerating a ```json fence around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return json.loads(fenced.group(1) if fenced else text)


def ask(client, model, messages):
    """One chat call. Prefer strict JSON-schema output; fall back if the provider refuses it."""
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": Extraction.model_json_schema(), "strict": True},
    }
    try:
        resp = client.chat_completion(model=model, messages=messages, temperature=0, max_tokens=8000, response_format=schema)
    except Exception as e:  # provider may not support structured output for this model
        print(f"  structured output rejected ({type(e).__name__}); retrying with prompt only")
        resp = client.chat_completion(model=model, messages=messages, temperature=0, max_tokens=8000)
    return resp.choices[0].message.content or ""


def extract(segments, model, token, participants=None):
    client = InferenceClient(api_key=token)
    schema_hint = json.dumps(Extraction.model_json_schema())
    rules = INSTRUCTIONS
    if participants:
        rules += PARTICIPANT_RULES.format(names=", ".join(participants), unknown=UNKNOWN)
    messages = [
        {"role": "system", "content": rules + f"\nJSON schema:\n{schema_hint}"},
        {"role": "user", "content": "Transcript:\n\n" + format_transcript(segments)},
    ]
    error = None
    for attempt in range(2):  # second try shows the model what was wrong with its first answer
        raw = ask(client, model, messages)
        try:
            return Extraction.model_validate(parse_json(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            error = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That was not valid against the schema: {e}\nReturn corrected JSON only."},
            ]
    raise ExtractionError(f"Model returned invalid output twice: {error}")


def extract_to_dict(segments, model, token, participants=None):
    """extract() with the result flattened into plain dicts and lists, ready for JSON or a table."""
    result = extract(segments, model, token, participants).model_dump()
    result["speaker_map"] = {e["speaker"]: e["name"] for e in result["speaker_map"]}
    result["decisions"] = [d["decision"] for d in result["decisions"]]
    result["model"] = model
    if participants:
        result["participants"] = participants
        allowed = {n.lower() for n in participants} | {UNKNOWN}
        used = set(result["speaker_map"].values()) | {i["owner"] for i in result["action_items"]}
        result["stray_names"] = sorted(n for n in used if n.lower() not in allowed)
    return result


def default_output(transcript, model, participants):
    name = transcript.stem.replace("_transcript", "_extracted")
    if participants:
        name += "_participants"
    if model != DEFAULT_MODEL:
        name += "_" + model.split("/")[-1].lower()
    return transcript.with_name(name + ".json")
