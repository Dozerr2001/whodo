"""Checks for meetingmate/dates.py. Run with: .venv/bin/python tests/test_dates.py"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meetingmate.dates import resolve_deadline  # noqa: E402

WED = date(2026, 9, 30)  # a Wednesday
FRI = date(2026, 10, 2)

# (meeting date, phrase, expected date, expected check flag)
CASES = [
    # weekdays
    (WED, "this Friday", FRI, False),
    (WED, "Friday", FRI, False),
    (WED, "by Friday EOD", FRI, False),
    (WED, "end of day Friday", FRI, False),
    (WED, "next Wednesday", date(2026, 10, 7), True),
    (WED, "next Friday", date(2026, 10, 9), True),
    (WED, "Wednesday", date(2026, 9, 30), True),  # said on a Wednesday: today, or a week on?
    (FRI, "this Wednesday", date(2026, 10, 7), True),  # already past this week
    (date(2026, 10, 4), "next Wednesday", date(2026, 10, 7), True),  # Sunday still belongs to the old week
    (WED, "Friday or Monday", FRI, True),  # hedged
    (WED, "last Friday", None, False),
    # today and relative
    (WED, "end of day today", WED, False),
    (WED, "EOD", WED, False),
    (WED, "tonight", WED, False),
    (WED, "tomorrow", date(2026, 10, 1), False),
    (WED, "the day after tomorrow", FRI, False),
    (WED, "in 3 days", date(2026, 10, 3), False),
    (WED, "in two weeks", date(2026, 10, 14), False),
    (WED, "end of the week", FRI, True),
    (date(2026, 10, 3), "end of the week", date(2026, 10, 9), True),  # Saturday: the coming Friday
    (WED, "end of next week", date(2026, 10, 9), True),
    (WED, "end of the month", date(2026, 9, 30), False),
    (WED, "end of next month", date(2026, 10, 31), False),
    # calendar dates
    (WED, "November 5th", date(2026, 11, 5), False),
    (WED, "Nov 5", date(2026, 11, 5), False),
    (WED, "5th of November", date(2026, 11, 5), False),
    (WED, "November fifth", date(2026, 11, 5), False),
    (WED, "November twenty-first", date(2026, 11, 21), False),
    (WED, "November 5, 2027", date(2027, 11, 5), False),
    (WED, "the 15th", date(2026, 10, 15), True),  # the 15th of September has passed, so October
    (date(2026, 10, 1), "the 15th", date(2026, 10, 15), False),
    (date(2026, 12, 20), "January 10th", date(2027, 1, 10), True),  # rolled into next year
    (WED, "February 30th", None, False),  # not a real date
    # no date
    (WED, "before launch", None, False),
    (WED, "ASAP", None, False),
    (WED, "next week", None, False),
    (WED, "once the design is approved", None, False),
    (WED, "", None, False),
    (WED, None, None, False),
]


def test_resolve_deadline():
    failures = []
    for anchor, phrase, want_date, want_check in CASES:
        got = resolve_deadline(phrase, anchor)
        if (got.day, got.check) != (want_date, want_check if want_date else False):
            failures.append(f"{anchor:%a %d %b} {phrase!r}: got {got.day} check={got.check}, want {want_date} check={want_check}")
        assert got.phrase == (phrase or "").strip()
    assert not failures, "\n" + "\n".join(failures)


def test_date_label():
    d = resolve_deadline("this Friday", WED)
    assert d.date_label(WED) == "Fri, 2 Oct"
    assert resolve_deadline("November 5, 2027", WED).date_label(WED) == "Fri, 5 Nov 2027"
    assert resolve_deadline("ASAP", WED).date_label(WED) == ""


if __name__ == "__main__":
    test_resolve_deadline()
    test_date_label()
    print(f"ok, {len(CASES)} cases")
