import re
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import util
from models.transformer_loader import get_models
from modules.model_utils import predict_meaning_wic
from utils.preprocess import get_spacy_doc
import os

# Set local NLTK data path
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

# Ensure WordNet is available
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet', download_dir=NLTK_DATA_PATH)
    nltk.download('omw-1.4', download_dir=NLTK_DATA_PATH)
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)

# Predefined Homonym Dictionary mapping words to various meanings guided by context keywords
# This remains for backward compatibility and high-quality curated data
HOMONYM_DICT = {
    "bank": {
        "financial": {"definition": "A financial institution that accepts deposits.", "keywords": ["financial", "institution", "money", "cash", "deposit", "loan"], "example": "He deposited a check at the bank."},
        "river": {"definition": "Sloping land, especially beside a river.", "keywords": ["river", "edge", "water", "stream", "fishing", "mud", "shore", "sat"], "example": "The river bank was muddy."}
    },
    "bat": {
        "animal": {"definition": "A mouselike flying mammal.", "keywords": ["animal", "mammal", "fly", "flying", "night", "wings", "cave"], "example": "A bat flew at night."},
        "sports": {"definition": "A club used for hitting a ball in sports.", "keywords": ["sports", "equipment", "baseball", "hit", "ball", "swing", "game"], "example": "He hit the ball with a bat."}
    },
    "match": {
        "contest": {"definition": "A game or contest.", "keywords": ["game", "contest", "win", "lose", "football", "tennis", "score"], "example": "The football match was tied."},
        "fire": {"definition": "A stick tipped with combustible chemical for starting fire.", "keywords": ["fire", "starter", "burn", "light", "strike", "flame"], "example": "He struck a match to light the fire."}
    },
    "spring": {
        "season": {"definition": "The season of growth following winter.", "keywords": ["season", "summer", "bloom", "flowers", "warm"], "example": "Flowers bloom in the spring."},
        "coil": {"definition": "An elastic device, typically a metal coil.", "keywords": ["coil", "metal", "bounce", "bed", "mattress", "jump"], "example": "The mattress springs are broken."},
        "water": {"definition": "A natural flow of ground water.", "keywords": ["water", "source", "drink", "fresh", "mountain"], "example": "Fresh water from the spring."}
    },
    "light": {
        "illumination": {"definition": "Natural agent that makes things visible.", "keywords": ["bright", "sun", "shine", "dark", "see", "lamp", "bulb"], "example": "Turn on the light."},
        "weight": {"definition": "Of little weight; not heavy.", "keywords": ["heavy", "feather", "carry", "weight", "easy"], "example": "The box is light to carry."}
    },
    "left": {
        "direction": {"definition": "On or towards the side of the body to the west when facing north.", "keywords": ["right", "turn", "direction", "side", "hand"], "example": "Turn left at the signal."},
        "departed": {"definition": "Went away from a place.", "keywords": ["go", "leave", "depart", "office", "home", "early", "went"], "example": "She left the office early."}
    },
    "watch": {
        "observe": {"definition": "Look at or observe attentively.", "keywords": ["look", "see", "movie", "bird", "screen", "observe"], "example": "Watch the movie carefully."},
        "timepiece": {"definition": "A small timepiece worn typically on a strap on one's wrist.", "keywords": ["wrist", "time", "clock", "wear", "gold", "strap"], "example": "I wore my new watch today."}
    },
    "ring": {
        "jewelry": {"definition": "A small circular band, typically of precious metal, worn on a finger.", "keywords": ["finger", "gold", "diamond", "wear", "jewelry", "wedding"], "example": "He wore a gold ring."},
        "sound": {"definition": "Make a clear resonant or vibrating sound.", "keywords": ["bell", "phone", "sound", "hear", "doorbell", "call"], "example": "I heard the phone ring."}
    },
    "file": {
        "document": {"definition": "A folder or box for holding loose papers arranged in order.", "keywords": ["paper", "document", "cabinet", "folder", "computer", "data"], "example": "Please file the papers in the cabinet."},
        "tool": {"definition": "A tool with a roughened surface used for smoothing.", "keywords": ["tool", "metal", "smooth", "edge", "nail", "wood", "shape"], "example": "Use a file to smooth the sharp edge."}
    },
    "seal": {
        "animal": {"definition": "A fish-eating aquatic mammal.", "keywords": ["animal", "ocean", "swim", "fish", "bark", "water", "ice"], "example": "The seal was clapping its flippers."},
        "closure": {"definition": "A device or substance used to join two things to prevent coming apart.", "keywords": ["close", "envelope", "wax", "tight", "door", "container", "leak"], "example": "Please break the seal to open the letter."}
    },
    "key": {
        "lock": {"definition": "A shaped piece of metal used to open a lock.", "keywords": ["door", "open", "lock", "metal", "car", "start"], "example": "I lost the house key."},
        "crucial": {"definition": "Of crucial importance.", "keywords": ["important", "crucial", "success", "element", "factor", "main"], "example": "Communication is key to winning."}
    },
    "park": {
        "recreation": {"definition": "A large public green area in a town.", "keywords": ["green", "grass", "play", "children", "tree", "walk", "picnic"], "example": "Children played in the park."},
        "vehicle": {"definition": "Bring a vehicle to a halt and leave it temporarily.", "keywords": ["car", "vehicle", "drive", "lot", "garage", "outside", "space"], "example": "Please park the car outside."}
    },
    "duck": {
        "animal": {"definition": "A waterbird with a broad blunt bill.", "keywords": ["bird", "water", "quack", "pond", "feather", "animal"], "example": "The duck swam in the pond."},
        "action": {"definition": "Lower the head or the body quickly to avoid a blow.", "keywords": ["dodge", "head", "avoid", "low", "hit", "hide", "down"], "example": "You need to duck down to fit through the door."}
    },
    "current": {
        "time": {"definition": "Belonging to the present time.", "keywords": ["now", "present", "event", "affair", "today", "modern"], "example": "Current events are complicated."},
        "flow": {"definition": "A body of water or air moving in a definite direction.", "keywords": ["water", "river", "ocean", "flow", "electric", "air", "strong"], "example": "The ocean current was very strong today."}
    },
    "crane": {
        "bird": {"definition": "A tall, long-legged, long-necked bird.", "keywords": ["bird", "fly", "animal", "neck", "water", "tall"], "example": "The crane flew over the lake."},
        "machine": {"definition": "A large, tall machine used for moving heavy objects.", "keywords": ["machine", "heavy", "lift", "build", "construction", "raise", "weight"], "example": "The construction crane lifted the steel beams."}
    },
    "right": {
        "direction": {"definition": "On or toward the side of the body to the east when facing north.", "keywords": ["left", "turn", "direction", "side", "hand"], "example": "Take a right turn at the intersection."},
        "correct": {"definition": "Morally good, justified, or acceptable; true or correct.", "keywords": ["correct", "true", "wrong", "answer", "moral", "just"], "example": "You found the right answer."}
    }
}

# Precision settings
VERB_BLACKLIST = {"go", "do", "make", "get", "set", "run", "went", "is", "was", "are", "were", "has", "have", "had", "be", "been", "being", "will", "would", "shall", "should", "can", "could", "did", "does"}

# Dual-threshold Strategy for Precision vs Demo Reliability
CURATED_THRESHOLD = 0.15
CURATED_GAP = 0.01

GENERAL_THRESHOLD = 0.40
GENERAL_GAP = 0.04

def get_meanings(word: str) -> list:
    """Fetch possible meanings for a word using WordNet."""
    synsets = wn.synsets(word)
    if not synsets:
        return []
    
    meanings = []
    for syn in synsets:
        meaning_dict = {
            "definition": syn.definition(),
            "example": syn.examples()[0] if syn.examples() else ""
        }
        meanings.append(meaning_dict)
    return meanings

def detect_homonym_meaning_wic(sentence: str, word: str) -> dict:
    """
    Tries to use the fine-tuned WiC model for homonym detection.
    Falls back to SBERT if WiC model is not available or fails.
    """
    candidates = []
    source = "wic_model"

    # 1. Check curated dictionary first
    if word in HOMONYM_DICT:
        for data in HOMONYM_DICT[word].values():
            candidates.append({
                "meaning": data["definition"],
                "example": data.get("example", "")
            })
        source = "wic_dict"
    else:
        # 2. Use WordNet for all other words
        wn_meanings = get_meanings(word)
        if not wn_meanings:
             return {"word": word, "meaning": "unknown meaning", "confidence": "no_definitions_found"}
        
        for m in wn_meanings:
            candidates.append({
                "meaning": m["definition"],
                "example": m["example"]
            })
        source = "wic_wordnet"

    # 3. Predict meaning
    # FOR DEMO RELIABILITY: We only use the DistilBERT WiC model if it has been fine-tuned.
    # Otherwise, we use SBERT which has much better zero-shot similarity for this task.
    model_dir = "saved_models/wic_model"
    if os.path.exists(model_dir):
        result = predict_meaning_wic(sentence, word, candidates)
        if result:
             result["confidence"] = f"{source} (wic_fine_tuned)"
             return result
    
    # 4. Fallback to original SBERT method for best zero-shot reliability
    return detect_homonym_meaning_sbert_fallback(sentence, word)

def detect_homonym_meaning_sbert_fallback(sentence: str, word: str) -> dict:
    """
    Fallback method using SBERT cosine similarity (original logic).
    """
    clean_meanings = []
    enriched_meanings = []
    source = "sbert_fallback"
    
    if word in HOMONYM_DICT:
        for data in HOMONYM_DICT[word].values():
            clean_meanings.append(data["definition"])
            enriched_meanings.append(f"{' '.join(data['keywords'])}")
    else:
        wn_meanings = get_meanings(word)
        for m in wn_meanings:
            clean_meanings.append(m["definition"])
            enriched_meanings.append(m["definition"])
            
    if not clean_meanings:
         return {"word": word, "meaning": "unknown meaning", "confidence": "no_definitions_found"}

    models_loader = get_models()
    model = models_loader.sbert
    
    query = model.encode(sentence, convert_to_tensor=True)
    scores = []
    for enriched_m in enriched_meanings:
        emb = model.encode(enriched_m, convert_to_tensor=True)
        score = util.cos_sim(query, emb)
        scores.append(score.item())
        
    best_index = scores.index(max(scores))
    top_score = max(scores)
    
    # Calculate gap
    if len(scores) > 1:
        sorted_scores = sorted(scores, reverse=True)
        score_gap = sorted_scores[0] - sorted_scores[1]
    else:
        score_gap = 1.0

    return {
        "word": word,
        "meaning": clean_meanings[best_index],
        "confidence": source,
        "score": top_score,
        "score_gap": score_gap
    }

def analyze_homonyms_sbert_pipeline(text: str) -> dict:
    """
    Entry point for high-precision homonym detection.
    Uses spaCy for POS filtering, falling back to NLTK if needed.
    """
    doc = get_spacy_doc(text)
    
    tokens_to_process = []
    if doc:
        for token in doc:
            tokens_to_process.append({
                "text": token.text,
                "pos": token.pos_,
                "word": token.text.lower()
            })
    else:
        # Fallback to NLTK POS tagging
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        # NLTK uses Penn Treebank tags: NN (Noun), JJ (Adj), VB (Verb)
        # Mapping to simpler categories for consistency
        for word, tag in pos_tags:
            mapped_pos = "OTHER"
            if tag.startswith("NN"): mapped_pos = "NOUN"
            elif tag.startswith("JJ"): mapped_pos = "ADJ"
            elif tag.startswith("VB"): mapped_pos = "VERB"
            
            tokens_to_process.append({
                "text": word,
                "pos": mapped_pos,
                "word": word.lower()
            })
    
    results = []
    seen_words = set()

    for item in tokens_to_process:
        word = item["word"]
        pos = item["pos"]
        is_curated = word in HOMONYM_DICT
        
        # 1. POS Filtering: Allow NOUN, PROPN, ADJ, VERB. Always allow Curated words to bypass POS filter if missed.
        if pos not in ["NOUN", "PROPN", "ADJ", "VERB"] and not is_curated:
            continue
            
        # 2. Verb Blacklist (and common short words)
        if word in VERB_BLACKLIST or len(word) < 3 or word in seen_words:
            continue
            
        # 3. Use WordNet to check for lexical ambiguity
        synsets = wn.synsets(word)
        
        if is_curated or len(synsets) > 1:
            result = detect_homonym_meaning_wic(text, word)
            
            if result:
                score = result.get("score", 0)
                gap = result.get("score_gap", 0)
                
                # 4. Apply Dual-Threshold Strategy
                # Curated words (demo) are more lenient; WordNet (general) are strict.
                th = CURATED_THRESHOLD if is_curated else GENERAL_THRESHOLD
                gp = CURATED_GAP if is_curated else GENERAL_GAP
                
                if score >= th and gap >= gp:
                    results.append(result)
                    seen_words.add(word)
            
    return {"homonyms": results}

def analyze_homonyms_rule_based(text: str) -> dict:
    """Old rule-based method preserved for reference."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    results = []
    for i, word in enumerate(words):
        if word in HOMONYM_DICT:
            meanings = HOMONYM_DICT[word]
            start_idx = max(0, i - 10)
            end_idx = min(len(words), i + 11)
            context_window = words[start_idx:end_idx]
            best_match_meaning = "unknown meaning / ambiguous"
            max_matches = 0
            for meaning, data in meanings.items():
                match_count = sum(1 for kw in data["keywords"] if kw in context_window)
                if match_count > max_matches:
                    max_matches = match_count
                    best_match_meaning = meaning
            results.append({
                "word": word,
                "meaning": best_match_meaning,
                "confidence": "rule-based match" if max_matches > 0 else "rule-based default"
            })
    return {"homonyms": results}
