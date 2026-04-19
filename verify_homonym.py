from modules.homonym_detector import analyze_homonyms_sbert_pipeline
from models.transformer_loader import get_models

import os
import nltk
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

test_sentences = [
    "I deposited money in the bank.",
    "We sat on the river bank.",
    "He hit the ball with a bat.",
    "The bat flew at night.",
    "Turn left at the signal.",
    "She left the office early.",
    "He wore a gold ring.",
    "I heard the phone ring.",
    "Please file the papers.",
    "Use a file to smooth the edge.",
    "Children played in the park.",
    "Please park the car outside."
]

for text in test_sentences:
    print(f"\nInput: {text}")
    report = analyze_homonyms_sbert_pipeline(text)
    if not report.get("homonyms"):
        print(" -> No ambiguous homonyms detected.")
    for item in report.get("homonyms", []):
        print(f" -> Detected: {item['word']}")
        print(f"    Meaning: {item['meaning']}")
        print(f"    Confidence: {item['confidence']} | Score: {item['score']:.2f} | Gap: {item['score_gap']:.2f}")
