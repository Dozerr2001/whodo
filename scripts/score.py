"""Score extract.py output against a meeting's answer key.

Usage:
    .venv/bin/python scripts/score.py                      # every test_data/meeting_01_extracted*.json
    .venv/bin/python scripts/score.py test_data/meeting_01_extracted.json

Reports action items caught, correct owners, invented items and the planted traps,
and prints extracted items next to expected ones so the matching can be checked by eye.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
MATCH_THRESHOLD = 0.5  # share of the shorter text's words that must appear in the other

STOPWORDS = {
    "a", "an", "the", "to", "of", "for", "and", "or", "by", "on", "in", "at", "with", "will", "be",
    "is", "are", "we", "this", "that", "it", "its", "from", "as", "up", "so", "not", "no",
}

# What each planted trap looks for. The answer key describes traps in prose, so the
# words to search for are written here. Only the scorer uses them, never the extraction prompt.
TRAP_CHECKS = {
    "deadline_change": {"required": ["november 10"], "forbidden": ["november 3"]},
    "reassignment": {"item": "faq"},
    "cancelled_item": {"forbidden_action": ["email", "campaign"]},
    "vague_owner": {"item": "screenshot"},
    "non_decision": {"forbidden_decision": ["whatsapp", "sms"]},
}


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"\bnov\.?(?=\W|$)", "november", text)
    return re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)


def words(text):
    out = set()
    for w in re.findall(r"[a-z0-9]+", normalize(text)):
        if w in STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):  # crude stemming: "terms" == "term"
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def similarity(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def match(expected, found, expected_text, found_text):
    """One-to-one pairing of expected and found items, best overall similarity.

    Returns {expected index: found index} for pairs above the threshold.
    """
    if not expected or not found:
        return {}
    sim = np.array([[similarity(expected_text(e), found_text(f)) for f in found] for e in expected])
    rows, cols = linear_sum_assignment(-sim)
    return {int(r): int(c) for r, c in zip(rows, cols) if sim[r, c] >= MATCH_THRESHOLD}


def same_person(a, b):
    return (a or "").strip().lower() == (b or "").strip().lower()


def deadline_ok(expected, found):
    if not expected:
        return not found
    return similarity(expected, found or "") >= 0.6


def mentions(text, keywords):
    text = normalize(text)
    return any(re.search(rf"\b{re.escape(k)}s?\b", text) for k in keywords)


def line(mark, text):
    return f"   {mark} {text}"


def fmt_item(item):
    owner = item.get("owner") or "nobody"
    return f"{item['task']} [{owner}, {item.get('deadline') or 'no deadline'}]"


def score_file(path, key):
    ex = json.loads(path.read_text())
    ex["decisions"] = [d if isinstance(d, str) else d["decision"] for d in ex["decisions"]]  # older files hold plain strings
    print("=" * 78)
    print(f"{path.name}  (model: {ex.get('model', '?')})")
    print("=" * 78)
    print("speaker map:", ", ".join(f"{k}={v}" for k, v in ex["speaker_map"].items()))

    # Action items
    exp_items, got_items = key["action_items"], ex["action_items"]
    pairs = match(exp_items, got_items, lambda e: e["task"], lambda f: f["task"])
    print("\nACTION ITEMS  (expected -> extracted)")
    caught = owners = deadlines = 0
    for i, e in enumerate(exp_items):
        print(f" expected: {fmt_item(e)}")
        if i not in pairs:
            print(line("MISSED", "no matching extracted item"))
            continue
        f = got_items[pairs[i]]
        caught += 1
        owner_good, deadline_good = same_person(e["owner"], f["owner"]), deadline_ok(e["deadline"], f["deadline"])
        owners += owner_good
        deadlines += deadline_good
        print(line("caught", f"{fmt_item(f)}   owner {'OK' if owner_good else 'WRONG'}, deadline {'OK' if deadline_good else 'WRONG'}"))
    invented = [f for j, f in enumerate(got_items) if j not in pairs.values()]
    for f in invented:
        print(f" extra:    {fmt_item(f)}\n" + line("INVENTED", "matches no expected action item"))

    # Unassigned items
    exp_un, got_un = key["unassigned_items"], ex["unassigned_items"]
    un_pairs = match(exp_un, got_un, lambda e: e["task"], lambda f: f["task"])
    print("\nUNASSIGNED ITEMS")
    for i, e in enumerate(exp_un):
        print(f" expected: {e['task']}")
        print(line("caught" if i in un_pairs else "MISSED", got_un[un_pairs[i]]["task"] if i in un_pairs else "not flagged as unassigned"))
    for j, f in enumerate(got_un):
        if j not in un_pairs.values():
            print(f" extra:    {f['task']}\n" + line("extra", "not in the answer key"))

    # Decisions and open questions
    decisions_caught = report_list("DECISIONS", key["decisions"], ex["decisions"])
    questions_caught = report_list("OPEN QUESTIONS", key["open_questions"], ex["open_questions"])

    # Traps
    print("\nTRAPS")
    results = {}
    every_text = " ".join(
        ex["decisions"] + [i["task"] + " " + (i.get("deadline") or "") for i in got_items + got_un] + ex["open_questions"]
    )
    for trap in key["traps"]:
        kind, check = trap["type"], TRAP_CHECKS.get(trap["type"], {})
        passed, why = True, []
        if "required" in check and not mentions(" ".join(ex["decisions"]), check["required"]):
            passed, why = False, [f"decisions never mention {check['required']}"]
        if "forbidden" in check and mentions(every_text, check["forbidden"]):
            passed, why = False, [f"old value {check['forbidden']} still appears"]
        if "forbidden_action" in check and any(mentions(i["task"], check["forbidden_action"]) for i in got_items):
            passed, why = False, ["cancelled task is still an action item"]
        if "forbidden_decision" in check and any(mentions(d, check["forbidden_decision"]) for d in ex["decisions"]):
            passed, why = False, ["deferred topic listed as a decision"]
        if "item" in check:
            kw = [check["item"]]
            in_actions = [i for i in got_items if mentions(i["task"], kw)]
            in_unassigned = [i for i in got_un if mentions(i["task"], kw)]
            expected_owner = next((e["owner"] for e in exp_items if mentions(e["task"], kw)), None)
            if expected_owner:  # reassignment: exactly the right owner, once
                passed = len(in_actions) == 1 and same_person(in_actions[0]["owner"], expected_owner)
                why = [] if passed else [f"owners found: {[i['owner'] for i in in_actions] or 'none'}, expected {expected_owner}"]
            else:  # vague owner: flagged unassigned and not handed to a person
                passed = bool(in_unassigned) and not in_actions
                why = [] if passed else ["not flagged as unassigned" if not in_unassigned else "also given an owner"]
        results[kind] = passed
        print(f" {'PASS' if passed else 'FAIL'}  {kind}: {trap['detail']}" + ("" if passed else f"\n        -> {'; '.join(why)}"))

    return {
        "file": path.name,
        "caught": f"{caught}/{len(exp_items)}",
        "owners": f"{owners}/{len(exp_items)}",
        "deadlines": f"{deadlines}/{len(exp_items)}",
        "invented": str(len(invented)),
        "traps": f"{sum(results.values())}/{len(results)}",
    }


def report_list(title, expected, found):
    pairs = match(expected, found, lambda e: e, lambda f: f)
    print(f"\n{title}")
    for i, e in enumerate(expected):
        print(f" expected: {e}")
        print(line("caught" if i in pairs else "MISSED", found[pairs[i]] if i in pairs else "no matching entry"))
    for j, f in enumerate(found):
        if j not in pairs.values():
            print(f" extra:    {f}\n" + line("extra", "not in the answer key (may be a fair addition)"))
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("extracted", nargs="*", type=Path, help="default: all test_data/meeting_01_extracted*.json")
    parser.add_argument("--key", type=Path, default=ROOT / "test_data" / "meeting_01_answer_key.json")
    args = parser.parse_args()

    files = args.extracted or sorted((ROOT / "test_data").glob("meeting_01_extracted*.json"))
    if not files:
        sys.exit("No extracted files found. Run scripts/extract.py first.")
    key = json.loads(args.key.read_text())

    rows = [score_file(f, key) for f in files]

    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    headers = ["file", "caught", "owners", "deadlines", "invented", "traps"]
    widths = [max(len(h), *(len(r[h]) for r in rows)) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(r[h].ljust(w) for h, w in zip(headers, widths)))
    print("\ncaught/owners/deadlines are out of the expected action items; invented = extra action items.")


if __name__ == "__main__":
    main()
