# -*- coding: utf-8 -*-
"""
Test: Gemini-powered Length Control — 6 required test cases from spec.
Validates word count tolerance ±2, quality, and API integration.
"""
import os
import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import nltk
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

from modules.length_control import analyze_length_and_rewrite

TOLERANCE = 2

EXAMPLES = [
    # === 6 REQUIRED TEST CASES FROM SPEC ===
    {
        "name": "1. Small Compression",
        "text": "Artificial intelligence improves communication in organizations worldwide.",
        "target": 5,
    },
    {
        "name": "2. Medium Compression",
        "text": (
            "The annual college innovation festival includes coding hackathons, "
            "robotics contests, startup talks and networking sessions for students."
        ),
        "target": 12,
    },
    {
        "name": "3. Large Compression",
        "text": (
            "The rapidly evolving landscape of artificial intelligence and machine learning "
            "technologies is fundamentally reshaping how modern organizations approach complex "
            "business challenges, enabling them to automate repetitive tasks, analyze vast "
            "amounts of data with unprecedented speed and accuracy, and develop innovative "
            "solutions that were previously considered impossible to achieve."
        ),
        "target": 25,
    },
    {
        "name": "4. Small Expansion",
        "text": "Meeting tomorrow.",
        "target": 10,
    },
    {
        "name": "5. Medium Expansion",
        "text": "Submit report today.",
        "target": 12,
    },
    {
        "name": "6. Large Expansion",
        "text": "AI helps business.",
        "target": 20,
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
    print(f"  GEMINI LENGTH CONTROL TEST  ({total} cases, ±{TOLERANCE} tolerance)")
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
        tag = r['name'][:3]
        name = r['name'][3:49]
        print(f"{tag:<4} {name:<46} {r['orig']:>5} "
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
