"""Tie a quote from the LLM back to the transcript line it came from.

The LLM copies a short quote; it does not report who said it or when. Code finds the
line, so the speaker and timestamp shown are real, and a quote the model made up is dropped.
"""

import re
from bisect import bisect_right

MIN_OVERLAP = 0.7  # share of the quote's words that a line must contain to count as a fuzzy match
MIN_WORDS = 3


def _normalize(text):
    """Lowercase words only, so punctuation and spacing differences don't block a match."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def find_source(quote, segments):
    """Locate quote in the transcript. Returns {"quote", "speaker", "time"} or None.

    segments are lines with "speaker", "start" and "text". The quote is shown as the model
    gave it when it appears word for word; otherwise the closest line is shown instead.
    """
    wanted = _normalize(quote)
    words = wanted.split()
    if len(words) < MIN_WORDS:
        return None

    lines = [_normalize(s["text"]) for s in segments]
    starts, text = [], ""
    for line in lines:  # one searchable string, remembering where each line begins
        starts.append(len(text))
        text += line + " "

    pos = f" {text}".find(f" {wanted} ")
    if pos >= 0:
        index = bisect_right(starts, pos) - 1
        return _source(segments[index], quote.strip())

    best, best_score = None, 0.0
    for index, line in enumerate(lines):
        score = len(set(words) & set(line.split())) / len(set(words))
        if score > best_score:
            best, best_score = index, score
    if best is not None and best_score >= MIN_OVERLAP:
        return _source(segments[best], segments[best]["text"].strip())
    return None


def _source(segment, quote):
    return {"quote": quote, "speaker": segment["speaker"], "time": segment["start"]}
