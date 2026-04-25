# -*- coding: utf-8 -*-
"""
Test: Practical hybrid length control — 12 tests (5 required + 7 unseen inputs).
Validates word count tolerance ±2, punctuation, capitalization, no duplicate words.
"""
import os
import sys
import io
import nltk

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

from modules.length_control import analyze_length_and_rewrite

TOLERANCE = 2

EXAMPLES = [
    # === 5 REQUIRED TEST CASES ===
    {
        "name": "1. AI communication -> compress to 12",
        "text": (
            "Artificial intelligence is transforming communication systems by helping "
            "users summarize messages and remove ambiguity."
        ),
        "target": 12,
    },
    {
        "name": "2. Meeting fragment -> expand to 8",
        "text": "Meeting tomorrow.",
        "target": 8,
    },
    {
        "name": "3. Project fragment -> expand to 10",
        "text": "Project submission today.",
        "target": 10,
    },
    {
        "name": "4. College festival -> compress to 14",
        "text": (
            "The annual college innovation festival includes coding events, "
            "robotics contests, lectures, and networking sessions."
        ),
        "target": 14,
    },
    {
        "name": "5. AI short -> expand to 12",
        "text": "AI improves communication.",
        "target": 12,
    },
    # === 7 UNSEEN / RANDOM INPUTS ===
    {
        "name": "6. Interview fragment -> expand to 9",
        "text": "Interview next week.",
        "target": 9,
    },
    {
        "name": "7. Workshop -> expand to 8",
        "text": "Workshop on Friday.",
        "target": 8,
    },
    {
        "name": "8. Report -> expand to 10",
        "text": "Report submission tomorrow.",
        "target": 10,
    },
    {
        "name": "9. AI delays -> expand to 12",
        "text": "AI reduces delays in companies.",
        "target": 12,
    },
    {
        "name": "10. Tech fest long -> compress to 12",
        "text": (
            "The annual tech fest includes coding, robotics, and startup events "
            "with expert panels and cultural performances."
        ),
        "target": 12,
    },
    {
        "name": "11. Long business email -> compress to 18",
        "text": (
            "Dear team, I wanted to reach out to all of you regarding the upcoming "
            "quarterly review meeting that is scheduled for next Friday at 3 PM in "
            "the main conference room on the fifth floor of our headquarters building, "
            "and I would like everyone to prepare their departmental reports."
        ),
        "target": 18,
    },
    {
        "name": "12. Short note -> expand to 15",
        "text": "Fix the bug immediately.",
        "target": 15,
    },
]


def check_quality(text):
    """Returns (punct_ok, cap_ok, no_dup_ok)."""
    if not text:
        return False, False, False
    punct = text.rstrip()[-1] in '.!?'
    cap = text[0].isupper()
    words = text.lower().split()
    no_dup = True
    for i in range(len(words) - 1):
        a = words[i].strip('.,!?;:')
        b = words[i + 1].strip('.,!?;:')
        if a and a == b:
            no_dup = False
            break
    return punct, cap, no_dup


def run_tests():
    total = len(EXAMPLES)
    print("=" * 80)
    print(f"  HYBRID LENGTH CONTROL TEST  ({total} cases, ±{TOLERANCE} tolerance)")
    print("=" * 80)

    results = []
    pass_count = 0

    for ex in EXAMPLES:
        text = ex["text"]
        target = ex["target"]
        orig_wc = len(text.split())

        print(f"\n--- [{ex['name']}] ---")
        print(f"  Input ({orig_wc}w): {text}")
        print(f"  Target: {target} words")

        r = analyze_length_and_rewrite(text, target)
        out = r["rewritten_text"]
        fwc = r["final_word_count"]
        dev = r.get("deviation", fwc - target)

        p, c, d = check_quality(out)
        ok = abs(dev) <= TOLERANCE

        if ok:
            pass_count += 1

        print(f"  Output ({fwc}w): {out}")
        print(f"  Dev: {dev:+d} | Status: {r['status']} | Punct={p} Cap={c} NoDup={d}")
        print(f"  Verdict: [{'PASS' if ok else 'FAIL'}]")

        results.append({
            "name": ex["name"], "orig": orig_wc, "target": target,
            "final": fwc, "dev": dev, "ok": ok, "qual": p and c and d,
        })

    # Summary
    print("\n" + "=" * 80)
    print(f"{'#':<4} {'Test':<46} {'Orig':>5} {'Tgt':>5} {'Out':>5} {'Dev':>5} {'Result':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['name'][:3]:<4} {r['name'][3:49]:<46} {r['orig']:>5} "
              f"{r['target']:>5} {r['final']:>5} {r['dev']:+5d} "
              f"{'PASS' if r['ok'] else 'FAIL':>7}")
    print("-" * 80)
    fail = total - pass_count
    print(f"  Passed: {pass_count}/{total}  |  Failed: {fail}/{total}")
    print("=" * 80)

    if fail == 0:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[PARTIAL] {fail} test(s) need review.")

    return fail == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
