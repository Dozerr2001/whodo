"""Checks for meetingmate/quotes.py. Run with: .venv/bin/python tests/test_quotes.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meetingmate.quotes import find_source  # noqa: E402

SEGMENTS = [
    {"speaker": "Priya", "start": 0.0, "text": "Okay, let's get started."},
    {"speaker": "Arjun", "start": 12.5, "text": "If I can have the fraud check done by November fifth, we're safe."},
    {"speaker": "Rahul", "start": 40.0, "text": "Yes, I'll have that done by this Friday."},
]


def test_exact_quote_is_located():
    src = find_source("I'll have that done by this Friday", SEGMENTS)
    assert src == {"quote": "I'll have that done by this Friday", "speaker": "Rahul", "time": 40.0}


def test_punctuation_and_case_do_not_matter():
    assert find_source("fraud check done by november fifth", SEGMENTS)["speaker"] == "Arjun"


def test_quote_spanning_two_lines_points_to_where_it_starts():
    src = find_source("let's get started If I can have the fraud check", SEGMENTS)
    assert src["speaker"] == "Priya"


def test_slightly_wrong_quote_shows_the_real_line():
    src = find_source("I will have that finished by this Friday", SEGMENTS)
    assert src["speaker"] == "Rahul" and src["quote"] == "Yes, I'll have that done by this Friday."


def test_invented_or_tiny_quote_is_dropped():
    assert find_source("We will hire three more engineers next quarter", SEGMENTS) is None
    assert find_source("Friday", SEGMENTS) is None
    assert find_source("", SEGMENTS) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
