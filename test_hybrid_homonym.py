"""
ClearComm Hybrid Homonym Detector — Verification Test Suite
Tests the 3-layer hybrid architecture with known homonym sentences.
"""
import os
import sys
import json

# Ensure we're running from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set NLTK data path
import nltk
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

from modules.homonym_detector import analyze_homonyms_sbert_pipeline, _load_curated_dataset

# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "sentence": "I deposited money in the bank.",
        "expected_word": "bank",
        "expected_meaning_contains": "financial"
    },
    {
        "sentence": "We sat near the river bank.",
        "expected_word": "bank",
        "expected_meaning_contains": "land"
    },
    {
        "sentence": "He hit the ball with a bat.",
        "expected_word": "bat",
        "expected_meaning_contains": "club"  
    },
    {
        "sentence": "The bat flew at night.",
        "expected_word": "bat",
        "expected_meaning_contains": "mammal"
    },
    {
        "sentence": "Please file the report.",
        "expected_word": "file",
        "expected_meaning_contains": "document"
    },
    {
        "sentence": "Use a file to smooth the edge.",
        "expected_word": "file",
        "expected_meaning_contains": "tool"
    },
    {
        "sentence": "Turn left at the signal.",
        "expected_word": "left",
        "expected_meaning_contains": "side"
    },
    {
        "sentence": "She left early.",
        "expected_word": "left",
        "expected_meaning_contains": "leave"
    },
    {
        "sentence": "Children played in the park.",
        "expected_word": "park",
        "expected_meaning_contains": "area"
    },
    {
        "sentence": "Please park the car outside.",
        "expected_word": "park",
        "expected_meaning_contains": "vehicle"
    },
]

def run_tests():
    print("=" * 70)
    print("  ClearComm Hybrid Homonym Detector — Test Suite")
    print("=" * 70)
    
    # Verify curated dataset loaded
    curated = _load_curated_dataset()
    print(f"\n[INFO] Curated dataset: {len(curated)} words loaded\n")
    
    passed = 0
    failed = 0
    
    for i, tc in enumerate(TEST_CASES, 1):
        sentence = tc["sentence"]
        expected_word = tc["expected_word"]
        expected_contains = tc["expected_meaning_contains"]
        
        print(f"--- Test {i}: \"{sentence}\"")
        
        result = analyze_homonyms_sbert_pipeline(sentence)
        homonyms = result.get("homonyms", [])
        
        # Find the expected word in results
        found = None
        for h in homonyms:
            if h["word"] == expected_word:
                found = h
                break
        
        if found:
            meaning = found["meaning"].lower()
            confidence = found.get("confidence", "unknown")
            
            if expected_contains.lower() in meaning:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL (wrong meaning)"
                failed += 1
            
            print(f"    Word: {found['word']}")
            print(f"    Meaning: {found['meaning']}")
            print(f"    Confidence: {confidence}")
            print(f"    Status: {status}")
        else:
            status = "FAIL (word not detected)"
            failed += 1
            all_words = [h["word"] for h in homonyms]
            print(f"    Expected '{expected_word}' but got: {all_words}")
            print(f"    Status: {status}")
        
        print()
    
    print("=" * 70)
    print(f"  Results: {passed} PASSED / {failed} FAILED / {len(TEST_CASES)} TOTAL")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
