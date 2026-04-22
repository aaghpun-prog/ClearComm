"""
ClearComm Homonym Accuracy Evaluation Script
============================================
Tests HYBRID system (Layer 1 curated + SBERT) vs BASELINE (SBERT-only).
Safe: does NOT modify any project files. Patches at runtime only.
Run: python eval_homonym_accuracy.py
"""

import os, sys, json, time, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import nltk
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

# ─────────────────────────────────────────────
# BENCHMARK DATASET  (67 labeled test cases)
# ─────────────────────────────────────────────
BENCHMARK = [
    # BANK (easy)
    {"sentence": "I deposited money in the bank.",            "word": "bank", "expected": "financial",  "category": "easy"},
    {"sentence": "I need to withdraw cash from the bank.",    "word": "bank", "expected": "financial",  "category": "easy"},
    {"sentence": "She applied for a loan at the bank.",       "word": "bank", "expected": "financial",  "category": "easy"},
    {"sentence": "We sat near the river bank.",               "word": "bank", "expected": "land",       "category": "easy"},
    {"sentence": "The river bank was covered in mud.",        "word": "bank", "expected": "land",       "category": "easy"},
    {"sentence": "He fished from the bank of the stream.",   "word": "bank", "expected": "land",       "category": "medium"},

    # BAT (easy)
    {"sentence": "He hit the ball with a bat.",               "word": "bat",  "expected": "club",       "category": "easy"},
    {"sentence": "The cricket bat was broken.",               "word": "bat",  "expected": "club",       "category": "easy"},
    {"sentence": "The bat flew out of the cave at night.",   "word": "bat",  "expected": "mammal",     "category": "easy"},
    {"sentence": "A bat was hanging upside down in the cave.","word": "bat",  "expected": "mammal",     "category": "easy"},

    # FILE (easy/medium)
    {"sentence": "Please file the report.",                   "word": "file", "expected": "document",   "category": "easy"},
    {"sentence": "He saved the data in the computer file.",  "word": "file", "expected": "document",   "category": "easy"},
    {"sentence": "Use a file to smooth the sharp metal edge.","word": "file", "expected": "tool",       "category": "easy"},
    {"sentence": "She used a nail file to shape her nails.", "word": "file", "expected": "tool",       "category": "medium"},

    # LEFT (easy/medium/hard)
    {"sentence": "Turn left at the signal.",                  "word": "left", "expected": "side",       "category": "easy"},
    {"sentence": "He turned left at the corner.",             "word": "left", "expected": "side",       "category": "easy"},
    {"sentence": "She left the office early.",                "word": "left", "expected": "leave",      "category": "easy"},
    {"sentence": "He left home yesterday.",                   "word": "left", "expected": "leave",      "category": "easy"},
    {"sentence": "She left without saying goodbye.",          "word": "left", "expected": "leave",      "category": "medium"},
    {"sentence": "Take the left lane on the highway.",       "word": "left", "expected": "side",       "category": "medium"},

    # PARK (easy)
    {"sentence": "Children played in the park.",              "word": "park", "expected": "area",       "category": "easy"},
    {"sentence": "We had a picnic in the park.",             "word": "park", "expected": "area",       "category": "easy"},
    {"sentence": "Please park the car outside.",              "word": "park", "expected": "vehicle",    "category": "easy"},
    {"sentence": "He could not find a space to park.",       "word": "park", "expected": "vehicle",    "category": "easy"},

    # RING (easy/medium)
    {"sentence": "She wore a beautiful diamond ring.",        "word": "ring", "expected": "band",       "category": "easy"},
    {"sentence": "He gave her a gold ring at the wedding.",  "word": "ring", "expected": "band",       "category": "easy"},
    {"sentence": "I heard the phone ring.",                   "word": "ring", "expected": "sound",      "category": "easy"},
    {"sentence": "The doorbell rang loudly.",                 "word": "ring", "expected": "sound",      "category": "medium"},
    {"sentence": "The bell will ring at noon.",              "word": "ring", "expected": "sound",      "category": "medium"},

    # LIGHT (easy/medium/hard)
    {"sentence": "Turn on the light.",                        "word": "light","expected": "illuminat",  "category": "easy"},
    {"sentence": "The lamp gave a bright light.",             "word": "light","expected": "illuminat",  "category": "easy"},
    {"sentence": "The suitcase was very light.",              "word": "light","expected": "weight",     "category": "easy"},
    {"sentence": "She carried a light bag to the gym.",      "word": "light","expected": "weight",     "category": "medium"},
    {"sentence": "The feather felt light in his hand.",      "word": "light","expected": "weight",     "category": "hard"},

    # WATCH (easy/medium)
    {"sentence": "He checked his watch for the time.",        "word": "watch","expected": "timepiece",  "category": "easy"},
    {"sentence": "She wore a gold watch on her wrist.",      "word": "watch","expected": "timepiece",  "category": "easy"},
    {"sentence": "Let us watch the movie together.",          "word": "watch","expected": "observe",    "category": "easy"},
    {"sentence": "We watched the birds on the screen.",      "word": "watch","expected": "observe",    "category": "medium"},

    # SEAL (easy/medium)
    {"sentence": "The seal was resting on the ice.",          "word": "seal", "expected": "mammal",     "category": "easy"},
    {"sentence": "The seal swam through the arctic water.",  "word": "seal", "expected": "mammal",     "category": "easy"},
    {"sentence": "Break the seal to open the package.",      "word": "seal", "expected": "close",      "category": "easy"},
    {"sentence": "The wax seal kept the envelope closed.",   "word": "seal", "expected": "close",      "category": "medium"},

    # KEY (easy/medium/hard)
    {"sentence": "I lost the house key.",                     "word": "key",  "expected": "lock",       "category": "easy"},
    {"sentence": "He used the car key to start the engine.", "word": "key",  "expected": "lock",       "category": "easy"},
    {"sentence": "Hard work is the key to success.",         "word": "key",  "expected": "crucial",    "category": "easy"},
    {"sentence": "Communication is the key factor here.",    "word": "key",  "expected": "crucial",    "category": "medium"},
    {"sentence": "She found the key under the mat.",         "word": "key",  "expected": "lock",       "category": "medium"},

    # SPRING (easy/medium/hard)
    {"sentence": "Flowers bloom in the spring.",              "word": "spring","expected": "season",    "category": "easy"},
    {"sentence": "We hiked in the forest during spring.",    "word": "spring","expected": "season",    "category": "easy"},
    {"sentence": "The mattress spring is broken.",            "word": "spring","expected": "coil",      "category": "easy"},
    {"sentence": "We drank from a mountain spring.",         "word": "spring","expected": "water",     "category": "medium"},
    {"sentence": "The spring in the mechanism lost tension.","word": "spring","expected": "coil",      "category": "hard"},

    # CURRENT (medium/hard)
    {"sentence": "The ocean current was very strong.",       "word": "current","expected": "flow",     "category": "easy"},
    {"sentence": "An electric current flows through the wire.","word":"current","expected":"flow",     "category": "medium"},
    {"sentence": "Current events are concerning.",            "word": "current","expected": "present",  "category": "easy"},
    {"sentence": "The current situation requires attention.","word": "current","expected": "present",  "category": "medium"},

    # DUCK (easy/medium)
    {"sentence": "The duck swam across the pond.",            "word": "duck", "expected": "bird",      "category": "easy"},
    {"sentence": "The duck quacked loudly by the lake.",     "word": "duck", "expected": "bird",      "category": "easy"},
    {"sentence": "Duck down to avoid the low beam.",         "word": "duck", "expected": "avoid",     "category": "easy"},
    {"sentence": "He had to duck quickly to dodge the ball.","word": "duck", "expected": "avoid",     "category": "medium"},

    # CRANE (easy/medium)
    {"sentence": "The construction crane lifted the steel beams.","word":"crane","expected":"machine", "category": "easy"},
    {"sentence": "A tall crane was used to build the tower.","word": "crane","expected": "machine",   "category": "easy"},
    {"sentence": "The crane flew gracefully over the lake.", "word": "crane","expected": "bird",      "category": "easy"},
    {"sentence": "We spotted a crane nesting in the wetlands.","word":"crane","expected":"bird",      "category": "medium"},

    # RIGHT (easy/medium/hard)
    {"sentence": "Take a right turn at the intersection.",   "word": "right","expected": "direction", "category": "easy"},
    {"sentence": "The right answer was correct.",            "word": "right","expected": "correct",   "category": "medium"},
    {"sentence": "You made the right decision.",             "word": "right","expected": "correct",   "category": "hard"},
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def meaning_matches(predicted_meaning: str, expected_keyword: str) -> bool:
    """Case-insensitive substring match between predicted definition and expected keyword."""
    return expected_keyword.lower() in predicted_meaning.lower()


def run_pipeline(sentence: str, word: str, pipeline_fn) -> dict:
    """Run detection pipeline and extract result for target word."""
    t0 = time.time()
    try:
        result = pipeline_fn(sentence)
        elapsed = time.time() - t0
        homonyms = result.get("homonyms", [])
        for h in homonyms:
            if h.get("word", "").lower() == word.lower():
                return {"detected": True, "meaning": h.get("meaning", ""), "elapsed": elapsed}
        return {"detected": False, "meaning": "", "elapsed": elapsed}
    except Exception as e:
        return {"detected": False, "meaning": f"ERROR: {e}", "elapsed": time.time() - t0}


def evaluate(pipeline_fn, label: str):
    """Run all benchmark cases against a pipeline and compute metrics."""
    print(f"\n{'='*64}")
    print(f"  Evaluating: {label}")
    print(f"{'='*64}")

    correct = wrong = no_detect = 0
    total_time = 0.0
    per_word = {}
    per_cat = {"easy": {"c":0,"t":0}, "medium": {"c":0,"t":0}, "hard": {"c":0,"t":0}}
    errors = []

    for tc in BENCHMARK:
        res = run_pipeline(tc["sentence"], tc["word"], pipeline_fn)
        total_time += res["elapsed"]
        w = tc["word"]
        cat = tc["category"]

        if w not in per_word:
            per_word[w] = {"correct": 0, "total": 0}
        per_word[w]["total"] += 1
        per_cat[cat]["t"] += 1

        if not res["detected"]:
            no_detect += 1
            errors.append({"sentence": tc["sentence"], "word": w,
                           "expected": tc["expected"], "got": "(not detected)", "cat": cat})
        elif meaning_matches(res["meaning"], tc["expected"]):
            correct += 1
            per_word[w]["correct"] += 1
            per_cat[cat]["c"] += 1
        else:
            wrong += 1
            errors.append({"sentence": tc["sentence"], "word": w,
                           "expected": tc["expected"], "got": res["meaning"][:80], "cat": cat})

    total = len(BENCHMARK)
    accuracy = correct / total * 100
    avg_time = total_time / total * 1000  # ms per sentence

    # Precision / Recall estimates
    precision = correct / max(correct + wrong, 1) * 100
    recall = correct / max(correct + no_detect + wrong, 1) * 100

    print(f"\n  Total Cases  : {total}")
    print(f"  Correct      : {correct}")
    print(f"  Wrong        : {wrong}")
    print(f"  Not Detected : {no_detect}")
    print(f"  Accuracy     : {accuracy:.1f}%")
    print(f"  Precision    : {precision:.1f}%")
    print(f"  Recall       : {recall:.1f}%")
    print(f"  Avg Time/sent: {avg_time:.1f} ms")

    print(f"\n  Per-Category:")
    for cat in ("easy", "medium", "hard"):
        c, t = per_cat[cat]["c"], per_cat[cat]["t"]
        pct = c/t*100 if t else 0
        print(f"    {cat.capitalize():8s}: {c}/{t}  ({pct:.0f}%)")

    print(f"\n  Per-Word Accuracy:")
    for w in sorted(per_word):
        c = per_word[w]["correct"]
        t = per_word[w]["total"]
        pct = c/t*100 if t else 0
        bar = "#" * int(pct/10) + "." * (10 - int(pct/10))
        print(f"    {w:10s}: {bar}  {c}/{t}  ({pct:.0f}%)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    [{e['cat'].upper()}] \"{e['sentence']}\"")
            print(f"          expected ~'{e['expected']}' | got: '{e['got'][:60]}'")

    return {
        "label": label,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "no_detect": no_detect,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "avg_time_ms": avg_time,
        "per_word": per_word,
        "per_cat": per_cat,
        "errors": errors,
    }


# ─────────────────────────────────────────────
# BASELINE PIPELINE (SBERT only – no curated JSON)
# ─────────────────────────────────────────────

def make_baseline_pipeline():
    """
    Returns a pipeline function that disables Layer 1 curated matching
    so only SBERT fallback (Layer 2) is used. Patches module at runtime;
    restores original after evaluation. Project files are never touched.
    """
    import modules.homonym_detector as hd

    original_try_curated = hd._try_curated_match

    def _disabled_curated(sentence, word):
        return None  # Force Layer 2 every time

    hd._try_curated_match = _disabled_curated

    def baseline_pipeline(text):
        return hd.analyze_homonyms_sbert_pipeline(text)

    def restore():
        hd._try_curated_match = original_try_curated

    return baseline_pipeline, restore


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    from modules.homonym_detector import analyze_homonyms_sbert_pipeline, _load_curated_dataset

    print("\n" + "="*64)
    print("  ClearComm Homonym Accuracy Evaluation")
    print("="*64)
    curated = _load_curated_dataset()
    print(f"  Curated dataset: {len(curated)} words loaded")
    print(f"  Benchmark cases: {len(BENCHMARK)}")

    # ── A. Hybrid System ──
    hybrid_result = evaluate(analyze_homonyms_sbert_pipeline, "HYBRID SYSTEM (Layer1 Curated + SBERT)")

    # ── B. Baseline System ──
    baseline_fn, restore = make_baseline_pipeline()
    baseline_result = evaluate(baseline_fn, "BASELINE SYSTEM (SBERT Only – no curated data)")
    restore()

    # ── Comparison ──
    print(f"\n{'='*64}")
    print("  COMPARISON: Hybrid vs Baseline")
    print(f"{'='*64}")
    acc_gain = hybrid_result["accuracy"] - baseline_result["accuracy"]
    speed_gain = baseline_result["avg_time_ms"] - hybrid_result["avg_time_ms"]
    fp_reduction = baseline_result["wrong"] - hybrid_result["wrong"]
    nd_reduction = baseline_result["no_detect"] - hybrid_result["no_detect"]

    print(f"\n  {'Metric':<30} {'Baseline':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy %':<30} {baseline_result['accuracy']:>9.1f}% {hybrid_result['accuracy']:>9.1f}% {acc_gain:>+9.1f}%")
    print(f"  {'Precision %':<30} {baseline_result['precision']:>9.1f}% {hybrid_result['precision']:>9.1f}%")
    print(f"  {'Recall %':<30} {baseline_result['recall']:>9.1f}% {hybrid_result['recall']:>9.1f}%")
    print(f"  {'Avg Time/sentence (ms)':<30} {baseline_result['avg_time_ms']:>9.1f}  {hybrid_result['avg_time_ms']:>9.1f}  {-speed_gain:>+9.1f}")
    print(f"  {'Wrong predictions':<30} {baseline_result['wrong']:>10} {hybrid_result['wrong']:>10} {fp_reduction:>+10}")
    print(f"  {'Not detected':<30} {baseline_result['no_detect']:>10} {hybrid_result['no_detect']:>10} {nd_reduction:>+10}")

    # ── Save results JSON ──
    results = {
        "hybrid": hybrid_result,
        "baseline": baseline_result,
        "comparison": {
            "accuracy_gain_pct": acc_gain,
            "speed_gain_ms": speed_gain,
            "false_positive_reduction": fp_reduction,
            "no_detect_reduction": nd_reduction,
        }
    }
    out_path = os.path.join(os.getcwd(), "docs", "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Raw results saved → {out_path}")
    print(f"\n{'='*64}")
    print("  Evaluation complete.")
    print(f"{'='*64}\n")

    return results


if __name__ == "__main__":
    main()
