"""Checks for the summary parts of whodo/render.py. Run with: .venv/bin/python tests/test_render.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whodo.render import summary_block, summary_text  # noqa: E402

EX = {
    "summary": {
        "overview": "They picked a date. <b>Two</b> options.",
        "topics": [
            {"title": "Date", "discussed": "Which day.", "options_considered": ["May 3", "May 10"],
             "outcome_reasoning": "May 10, because testing needs a week.", "start": 75.0},
            {"title": "Venue", "discussed": "Where.", "options_considered": [],
             "outcome_reasoning": "Left open.", "start": None},
        ],
    }
}


def test_plain_text_is_clean():
    text = summary_text(EX)
    assert text.startswith("Meeting summary\n\nThey picked a date.")
    assert "1. Date (1:15)\n   Discussed: Which day.\n   Options considered: May 3; May 10\n   Outcome and why: May 10, because testing needs a week." in text
    assert "2. Venue\n   Discussed: Where.\n   Outcome and why: Left open." in text  # no time, no options line
    assert "**" not in text and "<div" not in text


def test_html_is_escaped_and_optional_parts_left_out():
    html = summary_block(EX)
    assert "&lt;b&gt;Two&lt;/b&gt;" in html and "<b>" not in html
    assert html.count("Options considered") == 1  # only the topic that has options
    assert html.count("mm-time") == 1  # only the topic that has a start time


def test_no_summary_means_nothing_shown():
    assert summary_block({}) == "" and summary_text({"summary": None}) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
