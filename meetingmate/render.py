"""Turn an extraction into the HTML tables and summary line the app shows.

Everything the LLM wrote goes through html.escape before it reaches the page.
"""

from html import escape

from .dates import resolve_deadline
from .extract import UNKNOWN

NEEDS_OWNER = "Needs owner"


def fmt_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def needs_owner(owner):
    return not owner or owner.strip().lower() == UNKNOWN


def owner_label(owner):
    return NEEDS_OWNER if needs_owner(owner) else owner


def deadline_parts(phrase, anchor):
    """(date text, phrase, needs a check) for a deadline. The date text is empty when no date could be worked out."""
    if not phrase:
        return "", "", False
    resolved = resolve_deadline(phrase, anchor)
    return resolved.date_label(anchor), phrase, resolved.check


def summary_line(ex):
    actions, decisions, questions = len(ex["action_items"]), len(ex["decisions"]), len(ex["open_questions"])
    unowned = sum(needs_owner(i["owner"]) for i in ex["action_items"]) + len(ex["unassigned_items"])
    parts = [
        f"{actions} action item{'' if actions == 1 else 's'}",
        f"{decisions} decision{'' if decisions == 1 else 's'}",
    ]
    if unowned:
        parts.append("1 needs an owner" if unowned == 1 else f"{unowned} need an owner")
    parts.append(f"{questions} open question{'' if questions == 1 else 's'}")
    return ", ".join(parts)


def _table(headers, rows, empty):
    if not rows:
        return f'<p class="mm-empty">{escape(empty)}</p>'
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="mm-scroll"><table class="mm-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _quote(source):
    if not source:
        return '<span class="mm-muted">—</span>'
    return (
        f'<span class="mm-quote">“{escape(source["quote"])}”</span>'
        f'<span class="mm-muted mm-meta">{escape(source["speaker"])} · {fmt_duration(source["time"])}</span>'
    )


def _owner(owner):
    return f'<span class="mm-badge mm-needs">{NEEDS_OWNER}</span>' if needs_owner(owner) else escape(owner)


def _deadline(phrase, anchor):
    date_text, phrase, check = deadline_parts(phrase, anchor)
    if not phrase:
        return '<span class="mm-muted">—</span>'
    if not date_text:
        return escape(phrase)  # no calendar date to show, so the phrase stands alone
    flag = ' <span class="mm-badge mm-check" title="The phrase could mean a different date. Check it.">check</span>' if check else ""
    return f'<strong>{escape(date_text)}</strong> <span class="mm-muted">({escape(phrase)})</span>{flag}'


def action_table(ex, anchor):
    rows = [
        [escape(i["task"]), _owner(i["owner"]), _deadline(i["deadline"], anchor), _quote(i["source"])]
        for i in ex["action_items"]
    ]
    return _table(["Task", "Owner", "Deadline", "Supporting quote"], rows, "No action items found.")


def decisions_table(ex):
    rows = [[escape(d["decision"]), _quote(d["source"])] for d in ex["decisions"]]
    return _table(["Decision", "Supporting quote"], rows, "No decisions found.")


def unassigned_table(ex):
    """Empty string when there are none: the app hides the heading too, so nothing is left on screen."""
    if not ex["unassigned_items"]:
        return ""
    return _table(["Task"], [[escape(i["task"])] for i in ex["unassigned_items"]], "")


def questions_table(ex):
    return _table(["Question"], [[escape(q)] for q in ex["open_questions"]], "No open questions.")


CSS = """
.mm-summary { font-size: var(--text-lg); font-weight: 600; margin: 4px 0; }
.mm-scroll { overflow-x: auto; }
.mm-table { width: 100%; border-collapse: collapse; font-family: var(--font); font-size: var(--text-md); }
.mm-table th, .mm-table td { text-align: left; vertical-align: top; padding: 8px 12px; border-bottom: 1px solid var(--border-color-primary); }
.mm-table th { font-weight: 600; color: var(--body-text-color-subdued); white-space: nowrap; }
.mm-muted { color: var(--body-text-color-subdued); }
.mm-quote { display: block; font-style: italic; }
.mm-meta { display: block; font-size: var(--text-sm); margin-top: 2px; }
.mm-empty { color: var(--body-text-color-subdued); font-family: var(--font); }
.mm-badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: var(--text-sm); font-weight: 600; white-space: nowrap; }
.mm-needs { background: #fde68a; color: #78350f; border: 1px solid #f59e0b; }
.mm-check { background: #e0e7ff; color: #3730a3; border: 1px solid #818cf8; }
.dark .mm-needs { background: #78350f; color: #fde68a; }
.dark .mm-check { background: #312e81; color: #c7d2fe; }
"""
