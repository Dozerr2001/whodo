"""The summary consistency check in scripts/score.py, and the facts handed to the summary call.
Run with: .venv/bin/python tests/test_summary_check.py"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("score", ROOT / "scripts" / "score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

from whodo import extract, summarize  # noqa: E402

EX = {
    "speaker_map": {}, "participants": ["Ana", "Ben", "Cy"], "decisions": [{"decision": "Ship on May 2"}],
    "action_items": [
        {"task": "Design the share screen mock-ups", "owner": "Ben", "deadline": "next wednesday"},
        {"task": "Write the FAQ page", "owner": "Ana", "deadline": "May 7"},
    ],
    "unassigned_items": [{"task": "Check whether the store screenshots need updating"}], "open_questions": ["Which channel?"],
}


def topic(discussed, outcome=""):
    return {"summary": {"overview": "", "topics": [{"title": "t", "discussed": discussed, "outcome_reasoning": outcome, "options_considered": []}]}}


def flagged(text, outcome=""):
    return score.check_summary({**EX, **topic(text, outcome)})


def test_dropped_task_is_flagged():
    flags = flagged("Ben offered mock-ups, but Ana decided to write the FAQ herself and forgo the mock‑ups.")
    assert any("dropped" in why for _, why in flags), flags


def test_wrong_owner_is_flagged():
    assert any("Ana" in why for _, why in flagged("Ana agreed to look into the store screenshots."))
    assert any("Cy" in why for _, why in flagged("Cy will write the FAQ page."))


def test_consistent_statements_pass():
    assert flagged("Ben will design the mock-ups by next Wednesday.", "Ana took the FAQ page so Ben could focus on mock-ups.") == []
    assert flagged("Ana asked whether the store screenshots need updating.", "Nobody took it.") == []
    assert flagged("Ben offered to take the FAQ page, but Ana said the mock-ups mattered more and wrote the FAQ herself.") == []  # an offer is not an assignment
    assert score.check_summary({**EX, "summary": None}) == []


def test_summary_call_gets_the_extraction_as_facts():
    text = summarize.facts_text(EX)
    assert "Design the share screen mock-ups (owner: Ben, deadline: next wednesday)" in text
    assert "Ship on May 2" in text and "Check whether the store screenshots" in text and "Which channel?" in text
    assert "source of truth" in summarize.INSTRUCTIONS


def test_non_breaking_hyphens_and_narrow_spaces_are_replaced():
    class C:
        def chat_completion(self, **kw):
            m = type("M", (), {"content": "fraud\u2011check on November\u202f3"})()
            return type("R", (), {"choices": [type("Ch", (), {"message": m})()]})()

    assert extract.ask(C(), "m", [], extract.Extraction) == "fraud-check on November 3"


def test_quotes_stay_only_around_exact_transcript_words():
    segs = [{"start": 0, "end": 5, "speaker": "Ana", "text": "no mock-ups matter more forget it"}]
    fix = lambda t: summarize.unquote_paraphrases(t, segs)
    assert fix('Ana said "mock-ups matter more".') == 'Ana said "mock-ups matter more".'
    assert fix('Ana took the FAQ "so Ben could focus".') == "Ana took the FAQ so Ben could focus."
    assert fix("They \u201cagreed to skip it\u201d.") == "They agreed to skip it."


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
