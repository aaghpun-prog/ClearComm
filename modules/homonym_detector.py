import re
import json
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

# ============================================================
# LAYER 1 — Curated Homonym Knowledge Base (loaded from JSON)
# ============================================================

# Module-level cache: loaded once, reused forever
_CURATED_CACHE = None

def _load_curated_dataset():
    """
    Loads the curated homonym dataset from data/homonyms.json.
    Uses module-level caching — file is read only once per process lifetime.
    Falls back to empty dict if file is missing (system still works via Layer 2).
    """
    global _CURATED_CACHE
    if _CURATED_CACHE is not None:
        return _CURATED_CACHE

    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'homonyms.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                _CURATED_CACHE = json.load(f)
            print(f"[ClearComm] Loaded curated homonym dataset: {len(_CURATED_CACHE)} words")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ClearComm] Warning: Could not load homonyms.json: {e}")
            _CURATED_CACHE = {}
    else:
        print("[ClearComm] Warning: data/homonyms.json not found. Layer 1 disabled, using SBERT fallback only.")
        _CURATED_CACHE = {}

    return _CURATED_CACHE

# Predefined Homonym Dictionary mapping words to various meanings guided by context keywords
# PRESERVED for backward compatibility — merged with JSON dataset at runtime
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

# Stopwords to skip during multi-occurrence detection
STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "it", "its", "this", "that", "i", "he", "she", "we", "they", "my", "his", "her", "our", "your", "their", "not", "no", "so", "if", "up", "out"}

# Context window size (tokens on each side) for multi-occurrence extraction
CONTEXT_WINDOW_SIZE = 5

# Minimum meaningful words in a context window to consider it valid
MIN_CONTEXT_WORDS = 3

# Dual-threshold Strategy for Precision vs Demo Reliability
CURATED_THRESHOLD = 0.15
CURATED_GAP = 0.01

GENERAL_THRESHOLD = 0.40
GENERAL_GAP = 0.04

# ============================================================
# LAYER 1 — Fast Curated Keyword-Scoring Engine
# ============================================================

# Confidence thresholds for Layer 1 curated matching
CURATED_MATCH_MIN_HITS = 1       # Minimum keyword hits to consider a match
CURATED_MATCH_MIN_RATIO = 0.10   # Minimum hit_ratio (hits / total_keywords)
CURATED_MATCH_STRONG_RATIO = 0.20  # Ratio above which we consider it a strong match
CURATED_MATCH_MIN_GAP = 1       # Minimum gap in keyword hits between best and second-best

def _get_merged_curated_entry(word: str) -> dict:
    """
    Returns the curated entry for a word, preferring the JSON dataset
    over the inline HOMONYM_DICT. The JSON dataset has richer keyword lists.
    Falls back to HOMONYM_DICT if the word is not in JSON.
    """
    curated = _load_curated_dataset()
    if word in curated:
        return curated[word]
    if word in HOMONYM_DICT:
        return HOMONYM_DICT[word]
    return None

def _extract_context_window(tokens: list, target_index: int, window: int = CONTEXT_WINDOW_SIZE) -> str:
    """
    Extracts a local context window of ±window tokens around the target_index.
    Returns the window as a plain string for pipeline consumption.
    """
    start = max(0, target_index - window)
    end = min(len(tokens), target_index + window + 1)
    return ' '.join(t['text'] for t in tokens[start:end])


def _context_has_enough_content(context: str, word: str) -> bool:
    """
    Checks if a context window has at least MIN_CONTEXT_WORDS meaningful words
    (excluding the target word itself and stopwords).
    """
    context_tokens = re.findall(r'\b\w+\b', context.lower())
    meaningful = [t for t in context_tokens if t != word and t not in STOPWORDS and len(t) >= 3]
    return len(meaningful) >= MIN_CONTEXT_WORDS


def _try_curated_match(sentence: str, word: str, local_context: bool = False) -> dict:
    """
    Layer 1: Fast keyword-scoring engine using the curated dataset.
    
    Logic:
      1. Get curated meanings for the word
      2. Tokenize the sentence into lowercase words
      3. For each meaning, count how many of its keywords appear in the sentence
      4. If the best meaning has enough hits AND a clear gap over second-best,
         return a confident curated result immediately.
    
    When local_context=True (multi-occurrence context window), matching is
    relaxed: any single keyword hit is sufficient to accept a meaning.
    This ensures strong context words like "loan", "river" etc. immediately
    resolve the correct meaning from a short context window.
    
    Returns:
      dict with word/meaning/confidence/score/score_gap if confident match found.
      None if no confident match (falls through to Layer 2).
    """
    entry = _get_merged_curated_entry(word)
    if not entry:
        return None

    # Tokenize sentence to lowercase word set for fast lookup
    sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
    
    scores = []
    for meaning_key, meaning_data in entry.items():
        keywords = meaning_data.get("keywords", [])
        if not keywords:
            scores.append((0, 0.0, meaning_key, meaning_data))
            continue
        
        # Count how many curated keywords appear in the sentence
        hits = sum(1 for kw in keywords if kw in sentence_words)
        hit_ratio = hits / len(keywords)
        scores.append((hits, hit_ratio, meaning_key, meaning_data))
    
    if not scores:
        return None
    
    # Sort by hits descending, then by ratio
    scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_hits, best_ratio, best_key, best_data = scores[0]
    
    # Calculate gap with second-best
    if len(scores) > 1:
        second_hits = scores[1][0]
        hit_gap = best_hits - second_hits
    else:
        hit_gap = best_hits  # Only one meaning, gap is the score itself
    
    # ---- Matching decision ----
    matched = False
    
    if local_context:
        # RELAXED RULE for multi-occurrence context windows:
        # Any single keyword hit is sufficient — strong context words like
        # "loan", "river", "money", "water" immediately resolve meaning
        if best_hits >= 1 and hit_gap >= 0:
            matched = True
    else:
        # STRICT RULE for single-occurrence (full sentence):
        # Original thresholds preserved exactly
        if (best_hits >= CURATED_MATCH_MIN_HITS and
                best_ratio >= CURATED_MATCH_MIN_RATIO and
                hit_gap >= CURATED_MATCH_MIN_GAP):
            matched = True
    
    if matched:
        # Determine confidence level
        if best_ratio >= CURATED_MATCH_STRONG_RATIO and hit_gap >= 2:
            confidence_label = "high"
        elif best_hits >= 2 or (local_context and hit_gap >= 1):
            confidence_label = "high"
        elif best_ratio >= CURATED_MATCH_MIN_RATIO:
            confidence_label = "medium"
        else:
            confidence_label = "low"
        
        return {
            "word": word,
            "meaning": best_data["definition"],
            "confidence": confidence_label,
            "score": best_ratio,
            "score_gap": hit_gap / max(len(entry), 1)
        }
    
    # Not confident enough — fall through to Layer 2
    return None


# ============================================================
# LAYER 2 — SBERT + WordNet AI Fallback (PRESERVED)
# ============================================================

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

    # 1. Check curated dictionary first (merged: JSON + inline)
    entry = _get_merged_curated_entry(word)
    if entry:
        for data in entry.values():
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
    Enhanced: Uses natural sentence encoding instead of raw keyword lists.
    """
    clean_meanings = []
    enriched_meanings = []
    source = "sbert_fallback"
    
    # Check merged curated data first (JSON + inline dict)
    entry = _get_merged_curated_entry(word)
    if entry:
        for meaning_key, data in entry.items():
            clean_meanings.append(data["definition"])
            # IMPROVED: Encode as natural sentence for better SBERT cosine similarity
            # Instead of raw keywords, use: "word meaning: definition. Example: ..."
            example = data.get("example", "")
            if example:
                enriched_meanings.append(f"{word} ({meaning_key}): {data['definition']} Example: {example}")
            else:
                enriched_meanings.append(f"{word} ({meaning_key}): {data['definition']}")
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


# ============================================================
# MAIN PIPELINE — Hybrid 3-Layer Architecture
# ============================================================

def _assign_confidence_label(result: dict, is_curated: bool) -> dict:
    """
    Assigns a human-readable confidence label to a result from Layer 2/3.
    Replaces raw engine names with 'high', 'medium', or 'low'.
    """
    score = result.get("score", 0)
    gap = result.get("score_gap", 0)
    
    if is_curated:
        # Curated words: more lenient thresholds (demo reliability)
        if score >= 0.30 and gap >= 0.03:
            result["confidence"] = "high"
        elif score >= CURATED_THRESHOLD and gap >= CURATED_GAP:
            result["confidence"] = "medium"
        else:
            result["confidence"] = "low"
    else:
        # General words: stricter thresholds (precision)
        if score >= 0.50 and gap >= 0.06:
            result["confidence"] = "high"
        elif score >= GENERAL_THRESHOLD and gap >= GENERAL_GAP:
            result["confidence"] = "medium"
        else:
            result["confidence"] = "low"
    
    return result


def _run_single_occurrence_pipeline(context: str, word: str, is_curated: bool) -> dict:
    """
    Runs the full 3-layer pipeline for a single occurrence of a word
    using the given context string. This is the core unit of analysis.
    
    Returns a result dict or None if no confident match.
    """
    # ===== LAYER 1: Try curated keyword match (relaxed for local context) =====
    curated_result = _try_curated_match(context, word, local_context=True)
    if curated_result:
        return curated_result
    
    # ===== LAYER 2: SBERT + WordNet AI Fallback =====
    result = detect_homonym_meaning_wic(context, word)
    if result:
        score = result.get("score", 0)
        gap = result.get("score_gap", 0)
        
        th = CURATED_THRESHOLD if is_curated else GENERAL_THRESHOLD
        gp = CURATED_GAP if is_curated else GENERAL_GAP
        
        if score >= th and gap >= gp:
            result = _assign_confidence_label(result, is_curated)
            return result
    
    # ===== LAYER 3: Silently skip low confidence =====
    return None


def _deduplicate_by_meaning(results: list) -> list:
    """
    Smart deduplication for multi-occurrence results of the SAME word.
    Selects the best-confidence result per unique meaning.
    Demo-safety: if all occurrences resolved to the same meaning,
    return all original results instead of collapsing.
    """
    if len(results) <= 1:
        return results

    # Confidence ranking for comparison
    conf_rank = {'high': 3, 'medium': 2, 'low': 1}

    # Build best result per unique meaning
    seen_meanings = {}
    for r in results:
        meaning = r.get('meaning', '')
        rank = conf_rank.get(r.get('confidence', 'low'), 0)

        if meaning not in seen_meanings or rank > conf_rank.get(seen_meanings[meaning].get('confidence', 'low'), 0):
            seen_meanings[meaning] = r

    # Demo-safety: do NOT collapse when all occurrences share the same meaning
    if len(results) > 1 and len(seen_meanings) == 1:
        return results

    return list(seen_meanings.values())


def analyze_homonyms_sbert_pipeline(text: str) -> dict:
    """
    Entry point for high-precision homonym detection.
    
    HYBRID 3-LAYER ARCHITECTURE:
      Layer 1: Curated JSON keyword-scoring (fast, no ML needed)
      Layer 2: SBERT + WordNet AI fallback (existing pipeline, unchanged)
      Layer 3: Safe low-confidence handling (explicit response instead of silence)
    
    MULTI-OCCURRENCE EXTENSION:
      When a word appears more than once, each occurrence is analyzed
      independently using a local context window (±5 tokens). Results
      with identical meanings are deduplicated, keeping the best confidence.
    
    Uses spaCy for POS filtering, falling back to NLTK if needed.
    """
    # Ensure curated dataset is loaded (cached after first call)
    _load_curated_dataset()
    
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
    
    # ====================================================================
    # STEP 1: Build word occurrence map (token index -> word)
    # ====================================================================
    word_positions = {}  # word -> list of token indices
    for idx, item in enumerate(tokens_to_process):
        w = item["word"]
        if w not in word_positions:
            word_positions[w] = []
        word_positions[w].append(idx)
    
    # ====================================================================
    # STEP 2: Identify which words are multi-occurrence candidates
    # ====================================================================
    multi_occ_words = set()
    for w, positions in word_positions.items():
        if len(positions) > 1 and w not in STOPWORDS and w not in VERB_BLACKLIST and len(w) >= 3:
            multi_occ_words.add(w)
    
    results = []
    seen_words = set()  # For single-occurrence dedup (preserves old behavior)

    # ====================================================================
    # STEP 3: Process multi-occurrence words FIRST (per-occurrence pipeline)
    # ====================================================================
    for word in multi_occ_words:
        positions = word_positions[word]
        is_curated = _get_merged_curated_entry(word) is not None
        
        # POS check: at least one occurrence must pass POS filter
        any_valid_pos = False
        for pos_idx in positions:
            pos = tokens_to_process[pos_idx]["pos"]
            if pos in ("NOUN", "PROPN", "ADJ", "VERB") or is_curated:
                any_valid_pos = True
                break
        
        if not any_valid_pos:
            continue
        
        # Check for lexical ambiguity
        synsets = wn.synsets(word)
        if not is_curated and len(synsets) <= 1:
            continue
        
        # Run pipeline per occurrence with local context
        occurrence_results = []
        for occ_num, pos_idx in enumerate(positions):
            pos = tokens_to_process[pos_idx]["pos"]
            
            # Each occurrence must independently pass POS filter
            if pos not in ("NOUN", "PROPN", "ADJ", "VERB") and not is_curated:
                continue
            
            # Extract local context window
            context = _extract_context_window(tokens_to_process, pos_idx)
            
            # Skip if context is too weak
            if not _context_has_enough_content(context, word):
                continue
            
            result = _run_single_occurrence_pipeline(context, word, is_curated)
            if result:
                # Add occurrence metadata
                result["context"] = context
                result["position"] = occ_num + 1  # 1-indexed occurrence number
                occurrence_results.append(result)
        
        # Deduplicate: only keep entries with DIFFERENT meanings
        if occurrence_results:
            deduped = _deduplicate_by_meaning(occurrence_results)
            results.extend(deduped)
        
        # Mark word as handled so single-occurrence loop skips it
        seen_words.add(word)

    # ====================================================================
    # STEP 4: Process single-occurrence words (original logic, unchanged)
    # ====================================================================
    for item in tokens_to_process:
        word = item["word"]
        pos = item["pos"]
        is_curated = _get_merged_curated_entry(word) is not None
        
        # 1. POS Filtering: Allow NOUN, PROPN, ADJ, VERB. Always allow curated words to bypass POS filter.
        if pos not in ["NOUN", "PROPN", "ADJ", "VERB"] and not is_curated:
            continue
            
        # 2. Verb Blacklist (and common short words)
        if word in VERB_BLACKLIST or len(word) < 3 or word in seen_words:
            continue
            
        # 3. Check for lexical ambiguity
        synsets = wn.synsets(word)
        
        if is_curated or len(synsets) > 1:
            
            # ===== LAYER 1: Try curated keyword match first =====
            curated_result = _try_curated_match(text, word)
            if curated_result:
                results.append(curated_result)
                seen_words.add(word)
                continue  # Skip Layer 2 — curated match is confident
            
            # ===== LAYER 2: SBERT + WordNet AI Fallback =====
            result = detect_homonym_meaning_wic(text, word)
            
            if result:
                score = result.get("score", 0)
                gap = result.get("score_gap", 0)
                
                # Apply Dual-Threshold Strategy
                th = CURATED_THRESHOLD if is_curated else GENERAL_THRESHOLD
                gp = CURATED_GAP if is_curated else GENERAL_GAP
                
                if score >= th and gap >= gp:
                    # Assign human-readable confidence label
                    result = _assign_confidence_label(result, is_curated)
                    results.append(result)
                    seen_words.add(word)
                # ===== LAYER 3: If score too low, word is silently skipped =====
                # (No false positive is better than a wrong answer for a demo)
            
    return {"homonyms": results}


# ============================================================
# LEGACY — Old rule-based method preserved for reference
# ============================================================

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
