import nltk
import os
from modules.homonym_detector import analyze_homonyms_sbert_pipeline, HOMONYM_DICT
from nltk.corpus import wordnet as wn

# Ensure environment is set up like the app
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

def debug_homonym_pipeline(text):
    print(f"DEBUGGING TEXT: {text}")
    print("-" * 30)
    
    # Check POS tags
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    print(f"NLTK POS TAGS: {tags}")
    
    # Check pipeline
    result = analyze_homonyms_sbert_pipeline(text)
    print(f"\nFINAL RESULT: {result}")
    
    # If empty, let's drill down into a specific word like 'bank'
    if not result.get("homonyms"):
        target_word = "bank"
        if target_word in text.lower():
            print(f"\nDrilling down into '{target_word}'...")
            synsets = wn.synsets(target_word)
            print(f"Synsets found: {len(synsets)}")
            
            # Simulate the detection
            from modules.model_utils import predict_meaning_wic
            candidates = []
            for data in HOMONYM_DICT[target_word].values():
                candidates.append({"meaning": data["definition"], "example": data.get("example", "")})
            
            res = predict_meaning_wic(text, target_word, candidates)
            if res:
                print(f"Prediction: {res}")
                print(f"Score: {res.get('score')}")
                print(f"Gap: {res.get('score_gap')}")
            else:
                print("Prediction failed.")

test_sentences = [
    "I went to the bank to deposit my money.",
    "The bat flew out of the cave.",
    "He hit the ball with a bat."
]

for s in test_sentences:
    debug_homonym_pipeline(s)
    print("\n" + "="*50 + "\n")
