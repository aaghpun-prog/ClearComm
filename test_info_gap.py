# -*- coding: utf-8 -*-
"""
Test: Smart Hybrid Info Gap Detection — 25 test cases.
Covers all 8 intent types, multi-gap detection, false positive control,
and unseen inputs.
"""
import os
import sys
import io
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from modules.info_gap_detector import check_info_gaps

# ==============================================================================
# BENCHMARK: 25 test cases
# ==============================================================================

BENCHMARK = [
    # --- CASUAL / GENERAL (should have NO gaps) ---
    {"id": 1,  "text": "Good morning everyone!", "expect_gap": False, "cat": "Casual"},
    {"id": 2,  "text": "Thank you so much.", "expect_gap": False, "cat": "Casual"},
    {"id": 3,  "text": "The earth revolves around the sun.", "expect_gap": False, "cat": "General"},
    {"id": 4,  "text": "Rome is the capital of Italy.", "expect_gap": False, "cat": "General"},
    {"id": 5,  "text": "I like cricket.", "expect_gap": False, "cat": "General"},

    # --- MEETING ---
    {"id": 6,  "text": "Team meeting tomorrow.",
     "expect_gap": True, "cat": "Meeting", "must_contain": ["Time"]},
    {"id": 7,  "text": "Catch up call at 2 PM.",
     "expect_gap": True, "cat": "Meeting", "must_contain": ["Venue", "Location"]},
    {"id": 8,  "text": "Project sync meeting tomorrow at 10am in Room 4B.",
     "expect_gap": False, "cat": "Meeting"},
    {"id": 9,  "text": "Let's meet on Zoom at 3pm.",
     "expect_gap": False, "cat": "Meeting"},

    # --- EVENT ---
    {"id": 10, "text": "Annual conference next month.",
     "expect_gap": True, "cat": "Event", "must_contain": ["Time", "Venue"]},
    {"id": 11, "text": "Party tonight at 8pm at the main hall.",
     "expect_gap": False, "cat": "Event"},

    # --- TASK ---
    {"id": 12, "text": "Submit the project report.",
     "expect_gap": True, "cat": "Task", "must_contain": ["Deadline"]},
    {"id": 13, "text": "Please update the report.",
     "expect_gap": True, "cat": "Task", "must_contain": ["Deadline"]},
    {"id": 14, "text": "John, please submit the assignment by Friday 5pm.",
     "expect_gap": False, "cat": "Task"},

    # --- INTERVIEW ---
    {"id": 15, "text": "Interview scheduled next week.",
     "expect_gap": True, "cat": "Interview", "must_contain": ["Time"]},
    {"id": 16, "text": "Job interview on Monday.",
     "expect_gap": True, "cat": "Interview", "must_contain": ["Time", "Venue"]},
    {"id": 17, "text": "Interview tomorrow at 10am via Google Meet.",
     "expect_gap": False, "cat": "Interview"},

    # --- WORKSHOP ---
    {"id": 18, "text": "Workshop tomorrow on AI.",
     "expect_gap": True, "cat": "Workshop", "must_contain": ["Time", "Venue"]},
    {"id": 19, "text": "Seminar next Monday on AI.",
     "expect_gap": True, "cat": "Workshop", "must_contain": ["Time", "Venue"]},

    # --- TRAVEL ---
    {"id": 20, "text": "Visit client tomorrow.",
     "expect_gap": True, "cat": "Travel", "must_contain": ["Time"]},
    {"id": 21, "text": "Visit branch office tomorrow.",
     "expect_gap": True, "cat": "Travel", "must_contain": ["Time"]},

    # --- PAYMENT ---
    {"id": 22, "text": "Pay hostel fees today.",
     "expect_gap": True, "cat": "Payment", "must_contain": ["Amount"]},
    {"id": 23, "text": "Pay fees soon.",
     "expect_gap": True, "cat": "Payment", "must_contain": ["Amount", "Deadline"]},

    # --- EMERGENCY ---
    {"id": 24, "text": "Server outage! Fix immediately.",
     "expect_gap": True, "cat": "Emergency", "must_contain": ["Contact"]},

    # --- MULTI-SENTENCE ---
    {"id": 25, "text": "Meeting tomorrow. Submit the report by Friday.",
     "expect_gap": True, "cat": "Multi", "must_contain": ["Time"]},
]


def check_result(gaps, tc):
    """Check if result matches expectations."""
    if tc["expect_gap"]:
        if not gaps:
            return False, "False Negative — no gaps found"
        if "must_contain" in tc:
            all_missing = ' '.join(g.get('missing', '') for g in gaps).lower()
            for req in tc["must_contain"]:
                if req.lower() not in all_missing:
                    return False, f"Missing expected field: {req}"
        return True, "Correct — gap detected"
    else:
        if gaps:
            return False, f"False Positive — found: {gaps[0].get('missing', '?')}"
        return True, "Correct — no gap"


def run_tests():
    total = len(BENCHMARK)
    print("=" * 80)
    print(f"  INFO GAP DETECTION TEST  ({total} cases)")
    print("=" * 80)

    results = []
    correct = 0
    fp = 0
    fn = 0
    total_time = 0

    for tc in BENCHMARK:
        t0 = time.time()
        try:
            res = check_info_gaps(tc["text"])
            gaps = res.get("gaps", [])
        except Exception as e:
            print(f"  ERROR on #{tc['id']}: {e}")
            gaps = []
        elapsed = time.time() - t0
        total_time += elapsed

        ok, msg = check_result(gaps, tc)

        if ok:
            correct += 1
        elif tc["expect_gap"] and not gaps:
            fn += 1
        elif not tc["expect_gap"] and gaps:
            fp += 1
        else:
            fn += 1  # wrong fields detected

        gap_str = gaps[0]['missing'] if gaps else "(none)"
        verdict = "PASS" if ok else "FAIL"

        print(f"  [{verdict}] #{tc['id']:>2} [{tc['cat']:<10}] {tc['text'][:55]:<56} -> {gap_str}")
        if not ok:
            print(f"         ^ {msg}")

        results.append({"id": tc["id"], "ok": ok, "cat": tc["cat"]})

    # Summary
    acc = correct / total * 100
    avg_ms = (total_time / total) * 1000

    print("\n" + "=" * 80)
    print(f"  SUMMARY")
    print("=" * 80)
    print(f"  Accuracy:        {acc:.1f}% ({correct}/{total})")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Avg Time/sent:   {avg_ms:.1f} ms")
    print("=" * 80)

    if correct == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[PARTIAL] {total - correct} test(s) need review.")

    return correct == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
