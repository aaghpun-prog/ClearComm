# -*- coding: utf-8 -*-
"""
Test script: Strict length control with 10 diverse examples.
Validates that every rewrite lands within +/-2 words of the target.
"""
import os
import sys
import io
import nltk

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set NLTK data path before any imports
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

from modules.length_control import analyze_length_and_rewrite

TOLERANCE = 2

# =====================================================================
# 10 TEST EXAMPLES (diverse lengths, styles, and target word counts)
# =====================================================================

EXAMPLES = [
    {
        "name": "1. Long business email -> short",
        "text": (
            "Dear team, I wanted to reach out to all of you regarding the upcoming quarterly "
            "review meeting that is scheduled for next Friday at 3 PM in the main conference room "
            "on the fifth floor of our headquarters building, and I would like everyone to prepare "
            "their departmental reports including budget summaries and project status updates."
        ),
        "target": 20,
    },
    {
        "name": "2. Short note -> expand",
        "text": "Fix the bug immediately.",
        "target": 15,
    },
    {
        "name": "3. Technical paragraph -> compress",
        "text": (
            "Machine learning models require large datasets for training. The quality of the data "
            "directly impacts the model performance. Data preprocessing steps include normalization, "
            "feature extraction, and handling missing values. Transfer learning can reduce the amount "
            "of data needed by leveraging pre-trained models."
        ),
        "target": 25,
    },
    {
        "name": "4. News style -> tighten",
        "text": (
            "The government announced a new policy to combat climate change by investing heavily "
            "in renewable energy infrastructure and setting strict carbon emission targets for "
            "all major industries across the country."
        ),
        "target": 18,
    },
    {
        "name": "5. Multi-sentence -> medium compress",
        "text": (
            "Artificial intelligence is transforming healthcare by enabling early disease detection "
            "through medical imaging analysis. Researchers at Stanford University developed a deep "
            "learning algorithm that can identify skin cancer with accuracy comparable to dermatologists. "
            "The technology uses convolutional neural networks trained on thousands of clinical images."
        ),
        "target": 30,
    },
    {
        "name": "6. Very long -> aggressive compress",
        "text": (
            "The annual technology conference brought together over five thousand developers, "
            "designers, and product managers from around the world to discuss the latest trends "
            "in software development, artificial intelligence, cloud computing, and cybersecurity. "
            "Keynote speakers included industry leaders from major tech companies who shared their "
            "vision for the future of technology and its impact on society, education, and the global economy."
        ),
        "target": 22,
    },
    {
        "name": "7. Simple sentence -> slight expand",
        "text": "The cat sat on the mat and watched the birds outside.",
        "target": 18,
    },
    {
        "name": "8. Legal/formal text -> compress",
        "text": (
            "Pursuant to the terms and conditions outlined in the aforementioned agreement, "
            "the party of the first part shall be obligated to provide written notice of "
            "termination no fewer than thirty calendar days prior to the effective date of "
            "such termination, failing which the agreement shall automatically renew for an "
            "additional period of twelve months."
        ),
        "target": 25,
    },
    {
        "name": "9. Conversational text -> tighten",
        "text": (
            "Hey, so I was thinking about what you said yesterday about the project deadline "
            "and I totally agree that we should probably push it back by at least a week because "
            "honestly there is just too much work left to do and the quality would really suffer "
            "if we tried to rush it."
        ),
        "target": 20,
    },
    {
        "name": "10. Academic text -> moderate compress",
        "text": (
            "The study employed a mixed-methods research design combining quantitative surveys "
            "with qualitative interviews to investigate the relationship between social media "
            "usage and academic performance among undergraduate students at three major universities "
            "during the spring semester of the academic year."
        ),
        "target": 25,
    },
]


def run_tests():
    """Run all 10 tests and report results."""
    print("=" * 72)
    print("  STRICT LENGTH CONTROL TEST SUITE  (10 examples, +/-2 tolerance)")
    print("=" * 72)

    results = []
    pass_count = 0
    fail_count = 0

    for i, ex in enumerate(EXAMPLES):
        text = ex["text"]
        target = ex["target"]
        original_wc = len(text.split())

        print(f"\n--- [{ex['name']}] ---")
        print(f"  Input:   {original_wc} words")
        print(f"  Target:  {target} words")

        result = analyze_length_and_rewrite(text, target)

        final_wc = result["final_word_count"]
        deviation = result.get("deviation", final_wc - target)
        status = result["status"]
        msg = result.get("message", "")

        within_tolerance = abs(deviation) <= TOLERANCE

        if within_tolerance:
            verdict = "PASS"
            pass_count += 1
        else:
            verdict = "FAIL"
            fail_count += 1

        print(f"  Output:  {final_wc} words (deviation: {deviation:+d})")
        print(f"  Status:  {status}")
        print(f"  Verdict: [{verdict}]")
        print(f"  Preview: {result['rewritten_text'][:100]}...")

        results.append({
            "name": ex["name"],
            "original": original_wc,
            "target": target,
            "final": final_wc,
            "deviation": deviation,
            "verdict": verdict,
        })

    # Summary table
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"{'#':<4} {'Test':<42} {'Orig':>5} {'Tgt':>5} {'Out':>5} {'Dev':>5} {'Result':>7}")
    print("-" * 72)
    for r in results:
        dev_str = f"{r['deviation']:+d}"
        print(f"{r['name'][:3]:<4} {r['name'][3:45]:<42} {r['original']:>5} {r['target']:>5} {r['final']:>5} {dev_str:>5} {r['verdict']:>7}")

    print("-" * 72)
    print(f"  Passed: {pass_count}/10  |  Failed: {fail_count}/10  |  Tolerance: +/-{TOLERANCE} words")
    print("=" * 72)

    if fail_count == 0:
        print("\n[SUCCESS] All 10 tests passed - strict length control is working!")
    else:
        print(f"\n[PARTIAL] {fail_count} test(s) exceeded tolerance. Review needed.")

    return fail_count == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
