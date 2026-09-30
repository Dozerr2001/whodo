"""Write the meeting summary: an overview plus the key discussion points, with the reasons behind outcomes.

This is a separate LLM call from extract.py. Asking one call to extract tasks and write a summary
made both worse, so each gets its own prompt. The tables come from extract.py; this explains the why.
"""

from .extract import (
    DEFAULT_MODEL, LLM_UNAVAILABLE, LLMUnavailable, Model, ask_structured, extract_to_dict, format_transcript, name_speakers,
)
from .quotes import find_source

INSTRUCTIONS = """\
You read the transcript of a meeting and write a short, factual summary of it. Each line looks like
"[start seconds] Name: text". Names were worked out by software and can occasionally be wrong, so
attribute something to a named person only when the words themselves make it clear.

Return a JSON object with these fields.

1. overview: two or three sentences on what the meeting was for and what it came to. Name the
   outcomes, not the list of who will do what.

2. topics: the subjects that got real discussion, in the order they came up. Fold small asides into
   the topic they belong to. Each topic has:
   - title: a few words.
   - discussed: what was talked about, in one or two sentences.
   - options_considered: alternatives or positions that people actually put forward and weighed,
     one short string each. Use an empty list if nothing was weighed. Never add an option nobody raised.
   - outcome_reasoning: what was settled or left open, and the reasons people gave for it, in one or
     two sentences. If nobody gave a reason, state the outcome and leave the reason out.
   - start_quote: the words, copied exactly, from the line where the topic begins.

Rules:
- Explain the why, using the reasons people gave.
- Keep the direction of every statement. Note who agreed, who objected, and what was preferred over
  what, exactly as said; if someone says X matters more than Y, do not turn it around.
- Keep tentative things tentative. An offer, suggestion or "I could..." is not something that was done
  or decided. Report decisions only when someone clearly made them.
- Say only what was said. Do not interpret, judge, or fill in. Do not call something dropped,
  cancelled, postponed or unnecessary unless someone said so.
- Action items are shown in a separate table. Do not list who will do what or by when, and do not make
  a topic out of task assignments, wrap-up or goodbyes. Mention a task only when it is the reason
  behind an outcome.
- No filler such as "a productive discussion". Keep every part short. A short meeting gives a short
  summary and a long meeting a longer one, with more topics.
- start_quote is copied word for word from the transcript text, without the "[12s] Name:" prefix.
- Respond with the JSON object only.
"""


SUMMARY_UNAVAILABLE = f"No summary this time. {LLM_UNAVAILABLE} The action items and decisions below are not affected."
SUMMARY_FAILED = "No summary this time: the AI service gave an unusable answer. The action items and decisions below are not affected."


class Topic(Model):
    title: str
    discussed: str
    options_considered: list[str]
    outcome_reasoning: str
    start_quote: str


class Summary(Model):
    overview: str
    topics: list[Topic]


def minutes_long(segments):
    return max(1, round(max((s.get("end", s["start"]) for s in segments), default=0) / 60))


def summarize(named_segments, model, token):
    """named_segments have real names in the speaker field. Returns {"overview", "topics"}; each topic gets a start time."""
    text = f"The meeting is about {minutes_long(named_segments)} minutes long.\n\nTranscript:\n\n{format_transcript(named_segments)}"
    summary = ask_structured(token, model, INSTRUCTIONS, text, Summary).model_dump()
    for topic in summary["topics"]:  # the model copies the opening words; code finds the real timestamp
        found = find_source(topic.pop("start_quote"), named_segments)
        topic["start"] = found["time"] if found else None
    return summary


def extract_with_summary(segments, model=DEFAULT_MODEL, token=None, participants=None, stopped=None):
    """extract_to_dict() plus a "summary".

    If only the summary fails, the action items are still returned: summary is None and summary_error says
    why, so the page can tell the user. A failure of the extraction itself raises, as before.
    stopped is an optional function that says whether the run was stopped; if so, the second LLM call is skipped.
    """
    result = extract_to_dict(segments, model, token, participants)
    result["summary"], result["summary_error"] = None, None
    if stopped and stopped():
        return result
    try:
        result["summary"] = summarize(name_speakers(segments, result["speaker_map"]), model, token)
    except LLMUnavailable:
        result["summary_error"] = SUMMARY_UNAVAILABLE
    except Exception as e:  # e.g. the model's answer was unusable twice
        print(f"  summary failed ({type(e).__name__})")
        result["summary_error"] = SUMMARY_FAILED
    return result
