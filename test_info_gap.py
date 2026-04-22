import os
import sys
import time
import json

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import get_spacy_doc
from models.transformer_loader import get_models
from modules.info_gap_detector import check_info_gaps as new_check_info_gaps

# ==============================================================================
# OLD BASELINE LOGIC (For comparison)
# ==============================================================================

def old_check_info_gaps(text: str) -> dict:
    """The original logic from before the upgrade."""
    gaps = []
    doc = get_spacy_doc(text)
    
    has_person = False
    has_time = False
    has_action = False
    
    if doc:
        has_person = any(ent.label_ in ['PERSON', 'ORG'] for ent in doc.ents)
        has_time = any(ent.label_ in ['DATE', 'TIME'] for ent in doc.ents)
        has_action = any(token.pos_ == 'VERB' for token in doc)
        
        has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
        if not has_subject and not has_person:
            gaps.append({"sentence": text, "missing": "responsible person"})
            
    models = get_models()
    if has_action:
        categories = ["contains a specific deadline or time", "has a condition or dependency", "is a general statement"]
        try:
            result = models.classify_zero_shot(text, categories)
            top_label = result['labels'][0]
            if top_label == "is a general statement" and not has_time:
                gaps.append({"sentence": text, "missing": "deadline"})
        except Exception:
            pass
            
    unique_gaps = []
    seen_missing = set()
    for gap in gaps:
        if gap['missing'] not in seen_missing:
            unique_gaps.append(gap)
            seen_missing.add(gap['missing'])
            
    return {"gaps": unique_gaps}

# ==============================================================================
# BENCHMARK DATASET
# ==============================================================================

BENCHMARK = [
    # Casual / Greetings (Should have NO gaps)
    {"text": "Good morning everyone!", "expected_gap": False, "category": "Casual"},
    {"text": "Nice weather today, isn't it?", "expected_gap": False, "category": "Casual"},
    {"text": "Thank you so much.", "expected_gap": False, "category": "Casual"},
    {"text": "Hello there.", "expected_gap": False, "category": "Casual"},
    
    # Meetings
    {"text": "Meeting tomorrow.", "expected_gap": True, "category": "Meeting", "missing_contains": ["Time", "Location"]},
    {"text": "Catch up call at 2pm.", "expected_gap": True, "category": "Meeting", "missing_contains": ["Location"]},
    {"text": "Project sync meeting tomorrow at 10am in Room 4B.", "expected_gap": False, "category": "Meeting"},
    {"text": "Let's meet.", "expected_gap": True, "category": "Meeting", "missing_contains": ["Time"]},
    {"text": "Let's meet on Zoom at 3pm.", "expected_gap": False, "category": "Meeting"},
    
    # Events
    {"text": "Annual conference next month.", "expected_gap": True, "category": "Event", "missing_contains": ["Time", "Venue"]},
    {"text": "Workshop on Sunday at 10am.", "expected_gap": True, "category": "Event", "missing_contains": ["Venue"]},
    {"text": "Party tonight at 8pm at the main hall.", "expected_gap": False, "category": "Event"},
    
    # Sales
    {"text": "Product available for sale.", "expected_gap": True, "category": "Sale", "missing_contains": ["Price", "Contact"]},
    {"text": "Selling my old bicycle for 50 dollars.", "expected_gap": True, "category": "Sale", "missing_contains": ["Contact"]},
    {"text": "Shoes on sale for $20. Contact 555-1234.", "expected_gap": False, "category": "Sale"},
    {"text": "We offer a 10% discount on all items. Buy now at www.example.com for $10.", "expected_gap": False, "category": "Sale"},
    
    # Job / Interview
    {"text": "Hiring for a new software role.", "expected_gap": True, "category": "Job", "missing_contains": ["Date", "Time", "Venue"]},
    {"text": "Job interview on Monday.", "expected_gap": True, "category": "Job", "missing_contains": ["Time", "Venue"]},
    {"text": "Interview tomorrow at 10am via Google Meet.", "expected_gap": False, "category": "Job"},
    
    # Tasks / Requests
    {"text": "Please update the report.", "expected_gap": True, "category": "Task", "missing_contains": ["Deadline"]},
    {"text": "Submit the assignment by Friday 5pm.", "expected_gap": True, "category": "Task", "missing_contains": ["Assignee"]},
    {"text": "John, please submit the assignment by Friday 5pm.", "expected_gap": False, "category": "Task"},
    {"text": "Need to finish this task.", "expected_gap": True, "category": "Task", "missing_contains": ["Deadline"]},
    
    # General Informational (Should have NO gaps)
    {"text": "The earth revolves around the sun.", "expected_gap": False, "category": "General"},
    {"text": "Rome is the capital of Italy.", "expected_gap": False, "category": "General"},
]

def check_match(gaps, expected_gap, missing_contains=None):
    if expected_gap:
        if not gaps:
            return False, "False Negative (Empty no-gap mistake)"
        if missing_contains:
            missing_str = gaps[0].get("missing", "").lower()
            for req in missing_contains:
                if req.lower() not in missing_str:
                    return False, f"Missing expected field: {req}"
        return True, "Correctly found gap"
    else:
        if gaps:
            return False, f"False Positive (Found gap: {gaps[0].get('missing')})"
        return True, "Correctly found NO gap"

def run_evaluation(name, pipeline_fn):
    print(f"\n{'='*60}")
    print(f" EVALUATING: {name}")
    print(f"{'='*60}")
    
    correct = 0
    false_positives = 0
    false_negatives = 0
    total_time = 0.0
    
    for i, tc in enumerate(BENCHMARK, 1):
        text = tc["text"]
        
        t0 = time.time()
        try:
            res = pipeline_fn(text)
            gaps = res.get("gaps", [])
        except Exception as e:
            print(f"Error on '{text}': {e}")
            gaps = []
        elapsed = time.time() - t0
        total_time += elapsed
        
        ok, msg = check_match(gaps, tc["expected_gap"], tc.get("missing_contains"))
        
        if ok:
            correct += 1
        else:
            if tc["expected_gap"] and not gaps:
                false_negatives += 1
            elif not tc["expected_gap"] and gaps:
                false_positives += 1
                
        # print(f"[{'PASS' if ok else 'FAIL'}] {text}")
        # if not ok:
        #     print(f"  -> {msg}")
            
    total = len(BENCHMARK)
    acc = correct / total * 100
    avg_time = (total_time / total) * 1000
    
    print(f"Accuracy:        {acc:.1f}% ({correct}/{total})")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives} (Empty no-gap mistakes)")
    print(f"Avg Time/sent:   {avg_time:.1f} ms")
    
    return {
        "accuracy": acc,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_time_ms": avg_time
    }

def main():
    print("Pre-loading models...")
    _ = get_models()
    get_spacy_doc("Warm up")
    
    print("Running Info Gap Benchmark...")
    old_res = run_evaluation("Old Baseline System", old_check_info_gaps)
    new_res = run_evaluation("New Hybrid System", new_check_info_gaps)
    
    print(f"\n{'='*60}")
    print(" COMPARISON")
    print(f"{'='*60}")
    print(f"Accuracy Improvement: +{new_res['accuracy'] - old_res['accuracy']:.1f}%")
    print(f"False Positives:      {old_res['false_positives']} -> {new_res['false_positives']}")
    print(f"False Negatives:      {old_res['false_negatives']} -> {new_res['false_negatives']}")
    print(f"Speed Improvement:    {old_res['avg_time_ms'] - new_res['avg_time_ms']:.1f} ms per sentence")
    
    # Save results
    results = {"old": old_res, "new": new_res}
    os.makedirs("docs", exist_ok=True)
    with open("docs/info_gap_eval.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
