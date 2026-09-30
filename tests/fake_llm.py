"""A stand-in for the LLM calls, so stop tests don't spend Inference Provider credits or depend on the network.

install() replaces ask_structured in meetingmate.extract and meetingmate.summarize.
  mode "instant": answers at once with an empty but valid result.
  mode "slow":    waits `delay` seconds, then answers (a stop during the wait must not lead to a second call).
"""

import time

import meetingmate.extract
import meetingmate.summarize

calls = []
mode = {"kind": "instant", "delay": 0.0}

EMPTY = {
    "Extraction": {"speaker_map": [], "decisions": [], "action_items": [], "unassigned_items": [], "open_questions": []},
    "Summary": {"overview": "Fake summary.", "topics": []},
}


def fake_ask_structured(token, model, rules, transcript_text, schema_model):
    calls.append(time.time())
    if mode["kind"] == "slow":
        time.sleep(mode["delay"])
    return schema_model.model_validate(EMPTY[schema_model.__name__])


def install():
    meetingmate.extract.ask_structured = fake_ask_structured
    meetingmate.summarize.ask_structured = fake_ask_structured
