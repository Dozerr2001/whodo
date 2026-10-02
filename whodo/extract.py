"""Pull decisions, action items and open questions out of a labeled transcript.

Asks an open model to return structured JSON: Groq first (GROQ_API_KEY), Hugging Face Inference Providers as the fallback.
The command-line entry point is scripts/extract.py.
"""

import json
import os
import re
import time
from pathlib import Path

from huggingface_hub import InferenceClient
from pydantic import BaseModel, ConfigDict, ValidationError

from . import config
from .quotes import find_source

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
   Each has decision and quote: the words, copied exactly, where it was settled.

3. action_items: tasks that someone committed to do. Each has task, owner and deadline.
   - Count it only if a specific person accepted or volunteered for the task. A suggestion, a
     wish, or "we should..." is not a commitment.
   - If a task is offered to one person and someone else takes it, the owner is the person who took it.
   - If a task is cancelled or dropped later in the meeting, leave it out entirely.
   - If a deadline is changed, use the latest one. Copy the deadline exactly as it was spoken
     (for example "by the 15th", "end of day", "after the review"). Never convert it to a calendar
     date, never work out which day a weekday or "tomorrow" falls on, and never add words to it.
   - If a person committed but gave no deadline, use null.
   - owner is a real name from speaker_map, or "{UNKNOWN}" if you cannot tell.
   - quote is the words, copied exactly, where the owner took the task on (or, if nobody did,
     where it was agreed).

4. unassigned_items: tasks that were mentioned as needing to be done, but nobody took them on
   (for example "someone should check..."). Do not repeat tasks that appear in action_items,
   and do not include cancelled tasks. Each has task only.

5. open_questions: questions or issues that were raised and left unresolved, including topics
   explicitly postponed. Do not include questions that were answered later.

General rules:
- Use only what is in the transcript. Do not invent items, names, or deadlines.
- One entry per real item. Do not split one task into several or repeat it.
- Write each task or decision as a short, self-contained sentence.
- A quote is one or two sentences at most, copied word for word from the transcript text. Leave
  out the "[12s] SPEAKER_00:" prefix and never join lines or change any words.
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


LLM_UNAVAILABLE = "The AI service is temporarily unavailable. Try the example instead."


class ExtractionError(Exception):
    """The model never returned valid output."""


class LLMUnavailable(ExtractionError):
    """The LLM could not be reached or refused the call (no credits, rate limit, outage). str() is the message for users."""

    def __init__(self):
        super().__init__(LLM_UNAVAILABLE)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpeakerName(Model):
    speaker: str
    name: str


class Decision(Model):
    decision: str
    quote: str


class ActionItem(Model):
    task: str
    owner: str
    deadline: str | None
    quote: str


class UnassignedItem(Model):
    task: str


class Extraction(Model):
    speaker_map: list[SpeakerName]
    decisions: list[Decision]
    action_items: list[ActionItem]
    unassigned_items: list[UnassignedItem]
    open_questions: list[str]


def format_transcript(segments):
    return "\n".join(f"[{s['start']:.0f}s] {s['speaker']}: {s['text']}" for s in segments)


def name_speakers(segments, speaker_map):
    """Swap SPEAKER_00 for a real name where the model found one."""
    def name(label):
        found = speaker_map.get(label, UNKNOWN)
        return label if found.lower() == UNKNOWN else found
    return [{**s, "speaker": name(s["speaker"])} for s in segments]


def parse_json(text):
    """Load JSON, tolerating a ```json fence around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return json.loads(fenced.group(1) if fenced else text)


def _provider_message(error):
    """The provider's own explanation of a 413 or 429 ("Limit 8000, Requested 9500"), or None.

    Only for those two statuses: they say which limit was hit and carry no request content. The organization id is cut out.
    """
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) not in (413, 429):
        return None
    try:
        message = str(response.json()["error"]["message"])
    except Exception:  # no body, not JSON, or a different shape
        return None
    return re.sub(r"org_\w+", "org_...", message)[:300]


def _describe(error):
    """Error type and HTTP status for the server log, plus the provider's limit message for a 413 or 429.
    Never any other message, which can carry request details."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    text = type(error).__name__ + (f" {status}" if status else "")
    message = _provider_message(error)
    return f"{text}: {message}" if message else text


RATE_LIMIT_RETRIES = 2
RATE_LIMIT_DEFAULT_WAIT = 5  # seconds, when the provider does not say how long
RATE_LIMIT_MAX_WAIT = 75  # a per-minute token limit needs most of a minute to refill; a longer wait means a daily limit, so fall back instead
RESET_MARGIN = 1  # seconds added to the reset time Groq reports, so the retry lands just after the window refills


def _seconds(text):
    """Seconds from a duration like "7", "44.257s", "250ms" or "1m2.5s", or None if it is missing or unreadable."""
    if text is None:
        return None
    match = re.fullmatch(r"\s*(?:(\d+(?:\.\d+)?)m(?!s))?\s*(?:(\d+(?:\.\d+)?)(ms|s)?)?\s*", str(text))
    if not match or not any(match.groups()[:2]):
        return None
    minutes, number, unit = match.groups()
    seconds = float(minutes or 0) * 60
    if number:
        seconds += float(number) / 1000 if unit == "ms" else float(number)
    return seconds


def _retry_wait(error):
    """Seconds to wait before retrying a 429, or None if it is not a 429 or the wait is too long.

    Uses the longer of retry-after and the time Groq says the token window takes to reset.
    """
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) != 429:
        return None
    headers = getattr(response, "headers", None) or {}
    hints = [_seconds(headers.get("retry-after"))]
    reset = _seconds(headers.get("x-ratelimit-reset-tokens"))
    hints.append(reset + RESET_MARGIN if reset is not None else None)
    hints = [h for h in hints if h is not None]
    wait = max(hints) if hints else RATE_LIMIT_DEFAULT_WAIT
    return wait if wait <= RATE_LIMIT_MAX_WAIT else None


def chat(client, **kwargs):
    """chat_completion, waiting out a rate limit (429) up to RATE_LIMIT_RETRIES times."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat_completion(**kwargs)
        except Exception as e:
            wait = _retry_wait(e)
            if wait is None or attempt == RATE_LIMIT_RETRIES:
                raise
            print(f"  rate limited ({_describe(e)}); waiting {wait:g}s, retry {attempt + 1} of {RATE_LIMIT_RETRIES}")
            time.sleep(wait)


def ask(client, model, messages, schema_model, extra_body=None):
    """One chat call. Prefer strict JSON-schema output; fall back if the provider refuses it.
    extra_body is for provider-specific settings."""
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": schema_model.model_json_schema(), "strict": True},
    }
    try:
        resp = chat(client, model=model, messages=messages, temperature=0, max_tokens=8000, response_format=schema, extra_body=extra_body)
    except Exception as e:  # provider may not support structured output for this model
        print(f"  structured output rejected ({_describe(e)}); retrying with prompt only")
        try:
            resp = chat(client, model=model, messages=messages, temperature=0, max_tokens=8000, extra_body=extra_body)
        except Exception as e2:  # no credits, rate limit, outage, no network: nothing more to try
            print(f"  LLM call failed ({_describe(e2)})")
            raise LLMUnavailable() from e2
    return (resp.choices[0].message.content or "").replace("\u2011", "-").replace("\u202f", " ")  # models sometimes emit non-breaking hyphens and narrow no-break spaces


def make_clients(token):
    """Groq first when GROQ_API_KEY is set (same model, called directly), then Hugging Face Inference Providers."""
    clients = []
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        clients.append(("Groq", InferenceClient(provider="groq", api_key=groq_key)))
    clients.append(("Hugging Face", InferenceClient(api_key=token)))
    return clients


def groq_settings(model):
    """Extra request settings for Groq: less hidden thinking, which would otherwise eat the per-minute token budget.
    Only gpt-oss models take this setting."""
    effort = config.GROQ_REASONING_EFFORT
    return None if effort == "default" or "gpt-oss" not in model else {"reasoning_effort": effort}


CHARS_PER_TOKEN = 3.85  # measured on a real request: 22,061 characters were 5,727 tokens
GROQ_OUTPUT_ALLOWANCE = 4000  # Groq's 413 on a 5,727-token prompt said "Requested 9724": the prompt plus about 4,000 for the answer


def estimated_request_tokens(messages):
    """What Groq will count for this request: the prompt (estimated from its length) plus room for the answer."""
    return round(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + GROQ_OUTPUT_ALLOWANCE


def ask_any(clients, model, messages, schema_model):
    """Try each provider in order; LLMUnavailable only when every one of them failed.
    Groq is skipped when the request would be over its per-request limit: a 413 cannot succeed, and trying costs a wait."""
    for i, (name, client) in enumerate(clients):
        if name == "Groq" and i + 1 < len(clients) and estimated_request_tokens(messages) > config.GROQ_TOKEN_LIMIT:
            print(f"  Groq skipped: about {estimated_request_tokens(messages)} tokens, over its {config.GROQ_TOKEN_LIMIT} limit; "
                  f"using {clients[i + 1][0]}")
            continue
        try:
            return ask(client, model, messages, schema_model, groq_settings(model) if name == "Groq" else None)
        except LLMUnavailable:
            if i + 1 < len(clients):
                print(f"  {name} failed; falling back to {clients[i + 1][0]}")
    raise LLMUnavailable()


def ask_structured(token, model, rules, transcript_text, schema_model):
    """Send rules plus a transcript and return the answer as a validated schema_model."""
    clients = make_clients(token)
    messages = [
        {"role": "system", "content": rules + f"\nJSON schema:\n{json.dumps(schema_model.model_json_schema())}"},
        {"role": "user", "content": transcript_text},
    ]
    error = None
    for attempt in range(2):  # second try shows the model what was wrong with its first answer
        raw = ask_any(clients, model, messages, schema_model)
        try:
            return schema_model.model_validate(parse_json(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            error = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That was not valid against the schema: {e}\nReturn corrected JSON only."},
            ]
    raise ExtractionError(f"Model returned invalid output twice: {error}")


def extract(segments, model, token, participants=None):
    rules = INSTRUCTIONS
    if participants:
        rules += PARTICIPANT_RULES.format(names=", ".join(participants), unknown=UNKNOWN)
    return ask_structured(token, model, rules, "Transcript:\n\n" + format_transcript(segments), Extraction)


def extract_to_dict(segments, model, token, participants=None):
    """extract() with the result flattened into plain dicts and lists, ready for JSON or a table."""
    result = extract(segments, model, token, participants).model_dump()
    result["speaker_map"] = {e["speaker"]: e["name"] for e in result["speaker_map"]}
    named = name_speakers(segments, result["speaker_map"])
    for item in result["decisions"] + result["action_items"]:
        item["source"] = find_source(item.pop("quote"), named)  # None when the quote isn't in the transcript
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
