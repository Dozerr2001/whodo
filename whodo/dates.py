"""Turn deadline phrases ("this Friday", "November 5th") into calendar dates, in code.

The LLM only copies the phrase as spoken. Doing the date arithmetic here keeps it
predictable and testable: same phrase and meeting date in, same date out.

resolve_deadline() returns a Deadline with one of three outcomes:
  - a date, sure of itself:            "this Friday"      -> Fri, 2 Oct
  - a date plus check=True:            "next Wednesday"   -> Wed, 7 Oct, worth a second look
  - no date, phrase kept as it was:    "before launch", "ASAP"
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}
TENS = {"twenty": 20, "thirty": 30}
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}

_WD = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_ORD_WORD = "|".join(sorted(ORDINAL_WORDS, key=len, reverse=True))
# A day of the month: "5", "5th", "fifth", "twenty first"
_DAY = rf"\d{{1,2}}(?:st|nd|rd|th)?|(?:(?:twenty|thirty)[ -])?(?:{_ORD_WORD})"

MONTH_FIRST = re.compile(rf"\b({_MONTH})\.? (?:the )?({_DAY})\b(?: (\d{{4}}))?")  # November 5th
DAY_FIRST = re.compile(rf"\b({_DAY})(?: of)? ({_MONTH})\b\.?(?: (\d{{4}}))?")  # 5th of November
DAY_ONLY = re.compile(rf"\bthe (\d{{1,2}})(?:st|nd|rd|th)\b")  # the 15th
HEDGE = re.compile(r"\b(or|maybe|perhaps|probably|around|about|roughly|approximately|sometime|ish)\b")


@dataclass(frozen=True)
class Deadline:
    phrase: str  # exactly as spoken
    day: date | None = None
    check: bool = False  # a date was found but the phrase could mean another one

    def date_label(self, anchor):
        """'Fri, 2 Oct', with the year added when it differs from the meeting's."""
        if not self.day:
            return ""
        label = f"{self.day:%a}, {self.day.day} {self.day:%b}"
        return label if self.day.year == anchor.year else f"{label} {self.day.year}"


def _day_number(text):
    """'5th' -> 5, 'fifth' -> 5, 'twenty first' -> 21."""
    text = text.replace("-", " ")
    if text[0].isdigit():
        return int(re.sub(r"\D", "", text))
    parts = text.split()
    return TENS.get(parts[0], 0) + ORDINAL_WORDS[parts[-1]] if len(parts) == 2 else ORDINAL_WORDS[parts[0]]


def _monday(d):
    return d - timedelta(days=d.weekday())


def _safe_date(year, month, day):
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month_end(year, month):
    return date(year, month, calendar.monthrange(year, month)[1])


def _add_months(d, n):
    index = d.year * 12 + d.month - 1 + n
    return index // 12, index % 12 + 1


def _calendar_date(t, anchor):
    """'November 5th', '5th of November', 'the 15th'. Returns (date, check) or None."""
    m = MONTH_FIRST.search(t)
    if m:
        month, day, year = MONTHS[m[1]], _day_number(m[2]), m[3]
    else:
        m = DAY_FIRST.search(t)
        if m:
            month, day, year = MONTHS[m[2]], _day_number(m[1]), m[3]
    if m:
        if year:
            d = _safe_date(int(year), month, day)
            return (d, False) if d else None
        d = _safe_date(anchor.year, month, day)
        if d and d < anchor:  # no year said and this year's date has passed, so it must mean next year
            d = _safe_date(anchor.year + 1, month, day)
            return (d, True) if d else None
        return (d, False) if d else None
    m = DAY_ONLY.search(t)
    if m:
        day = int(m[1])
        d = _safe_date(anchor.year, anchor.month, day)
        if d and d >= anchor:
            return d, False
        year, month = _add_months(anchor, 1)
        d = _safe_date(year, month, day)
        return (d, True) if d else None
    return None


def _weekday_date(t, anchor):
    m = re.search(rf"\b(?:(next|this|coming|last) )?({_WD})\b", t)
    if not m:
        return None
    which, index = m[1], WEEKDAYS[m[2]]
    if which == "last":
        return None  # a deadline in the past is not a deadline
    if which == "next":
        return _monday(anchor) + timedelta(days=7 + index), True  # "next" is often said of the coming one too
    ahead = (index - anchor.weekday()) % 7
    d = anchor + timedelta(days=ahead)
    crosses_week = which is not None and index < anchor.weekday()  # "this Wednesday", said on a Friday
    return d, ahead == 0 or crosses_week


def _relative_date(t, anchor):
    if "day after tomorrow" in t:
        return anchor + timedelta(days=2), False
    if re.search(r"\btomorrow\b", t):
        return anchor + timedelta(days=1), False
    m = re.search(rf"\bin (\d+|{'|'.join(NUMBER_WORDS)}) (day|week)s?\b", t)
    if m:
        n = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
        return anchor + timedelta(days=n * (7 if m[2] == "week" else 1)), False
    m = re.search(r"\bend of (?:the |this )?(next )?(week|month)\b", t)
    if m:
        if m[2] == "month":
            year, month = _add_months(anchor, 1 if m[1] else 0)
            return _month_end(year, month), False
        friday = _monday(anchor) + timedelta(days=4 + (7 if m[1] else 0))
        return (friday if friday >= anchor else friday + timedelta(days=7)), True  # the week may end on Sunday
    if re.search(r"\b(today|tonight|eod|end of (?:the )?day|close of business|cob)\b", t):
        return anchor, False
    return None


def resolve_deadline(phrase, anchor):
    """Resolve a deadline phrase against the meeting date. Never guesses silently: doubtful dates get check=True."""
    phrase = (phrase or "").strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", phrase.lower().replace("-", " ")) if phrase else ""
    t = re.sub(r"\s+", " ", t).strip()
    # A calendar date beats a weekday, which beats "today": in "EOD Friday" the day that counts is Friday.
    found = _calendar_date(t, anchor) or _weekday_date(t, anchor) or _relative_date(t, anchor)
    if not found:
        return Deadline(phrase)
    d, check = found
    return Deadline(phrase, d, check or bool(HEDGE.search(t)))
