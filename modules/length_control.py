"""
length_control.py — AI-Powered Length Control with Gemini API

Architecture:
  Primary: Gemini API (gemini-2.5-flash) for intelligent rewriting
  Fallback: Deterministic rule-based engine (original logic preserved)

Modes:
  A) Compression — target < original
  B) Expansion   — target > original
  C) Refinement  — target ≈ original
"""

import nltk
import os
import re
import time
import logging
from utils.preprocess import get_words, get_sentences, get_spacy_doc

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger("length_control")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[LengthCtrl] %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# NLTK PATH
# ---------------------------------------------------------------------------
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

WORD_TOLERANCE = 2

# ---------------------------------------------------------------------------
# GEMINI CLIENT (lazy singleton)
# ---------------------------------------------------------------------------
_gemini_client = None
_GEMINI_MODEL = "gemini-2.5-flash"

def _get_gemini_client():
    """Lazy-load Gemini client. Returns None if unavailable."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — will use fallback engine")
        return None
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialised")
        return _gemini_client
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return None


# ============================================================================
# GEMINI REWRITING ENGINE
# ============================================================================

def _build_prompt(text: str, target: int, original_count: int, critical_entities: list) -> str:
    if target < original_count:
        mode_instruction = (
            "COMPRESS the text. Rewrite it shorter while preserving the core meaning. "
            "Remove unnecessary details but keep the sentence grammatically complete and natural."
        )
    elif target > original_count:
        mode_instruction = (
            "EXPAND the text. Add relevant context, clarification, or detail to make it longer. "
            "Do NOT add random filler words. Every added word must contribute meaningfully."
        )
    else:
        mode_instruction = (
            "REFINE the text. Improve grammar, clarity, and style while keeping the same length."
        )

    entity_instruction = ""
    if critical_entities:
        entities_str = ", ".join(critical_entities)
        entity_instruction = f"\nCRITICAL ENTITIES TO PRESERVE:\n{entities_str}\n\nYou MUST preserve these details in your rewritten text. Preserving these details is more important than hitting the exact word count if forced to choose.\n"

    return f"""You are a professional English rewriting assistant.

TASK: {mode_instruction}
{entity_instruction}
STRICT RULES:
- Output MUST be close to {target} words. Count carefully.
- Output ONLY the rewritten text — no explanations, no labels, no quotes.
- Preserve the original meaning strictly.
- Use natural, professional English.
- No bullet points or lists.
- Proper grammar, punctuation, and capitalization.
- The output must be a complete sentence or paragraph.

ORIGINAL TEXT ({original_count} words):
{text}

TARGET: approximately {target} words.

Rewritten text:"""


def _gemini_rewrite(text: str, target: int, original_count: int, critical_entities: list, tolerance: int) -> str | None:
    """Call Gemini API with retries. Returns best valid output or None."""
    client = _get_gemini_client()
    if client is None:
        return None

    prompt = _build_prompt(text, target, original_count, critical_entities)
    best_result = None
    best_deviation = float('inf')
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Gemini request attempt {attempt}/{max_attempts}")
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip() if response.text else ""

            # Validate output
            if not raw:
                logger.warning(f"Attempt {attempt}: empty response")
                continue
            if raw.lower() == text.strip().lower():
                logger.warning(f"Attempt {attempt}: output identical to input")
                continue
            if _has_excessive_repetition(raw):
                logger.warning(f"Attempt {attempt}: excessive repetition")
                continue

            wc = _count_words(raw)
            dev = abs(wc - target)

            if dev <= tolerance:
                logger.info(f"Attempt {attempt}: exact match ({wc}w, target {target})")
                return raw

            if dev < best_deviation:
                best_deviation = dev
                best_result = raw
                logger.info(f"Attempt {attempt}: kept as best ({wc}w, dev {dev})")

            # Retry with tighter prompt
            retry_entities = f"Make sure to preserve these entities: {', '.join(critical_entities)}\n" if critical_entities else ""
            prompt = f"""Rewrite the following text to EXACTLY {target} words.
Your previous attempt was {wc} words which is wrong. Adjust to exactly {target} words.
Output ONLY the rewritten text.
{retry_entities}
Original: {text}

Rewritten ({target} words):"""

        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            continue

    return best_result


# ============================================================================
# RULE-BASED LENGTH ANALYSIS (for report_generator — PRESERVED)
# ============================================================================

def analyze_length(text: str, sentences: list, words: list) -> list:
    """
    Rule-based length analysis that flags sentences which are too long or too short.
    Returns a list of issue dictionaries for the report generator.
    """
    issues = []

    # Flag overly long sentences (> 30 words)
    for sent in sentences:
        sent_words = get_words(sent)
        if len(sent_words) > 30:
            issues.append({
                "type": "length",
                "severity": "warning",
                "message": f"Sentence is too long ({len(sent_words)} words). Consider splitting it.",
                "sentence": sent
            })

    # Flag very short paragraphs that may lack detail (< 5 words)
    if len(words) < 5:
        issues.append({
            "type": "length",
            "severity": "info",
            "message": f"Text is very short ({len(words)} words). It may lack sufficient detail.",
            "sentence": text
        })

    return issues


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _count_words(text: str) -> int:
    return len(text.split())


def _get_doc(text: str):
    try:
        return get_spacy_doc(text)
    except Exception:
        return None


def _has_excessive_repetition(text: str) -> bool:
    stop = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'is',
            'are', 'was', 'were', 'with', 'that', 'this', 'it', 'on', 'by', 'as'}
    counts = {}
    for w in text.lower().split():
        c = w.strip('.,!?;:')
        if c and c not in stop:
            counts[c] = counts.get(c, 0) + 1
    return any(v > 3 for v in counts.values())


# ============================================================================
# POST-PROCESSING PIPELINE
# ============================================================================

DANGLING_ENDINGS = frozenset({
    'and', 'but', 'or', 'nor', 'yet', 'so', 'for', 'with', 'without',
    'in', 'on', 'at', 'to', 'of', 'by', 'from', 'as', 'if', 'the',
    'a', 'an', 'is', 'are', 'was', 'were', 'be', 'that', 'which',
    'who', 'whose', 'where', 'when', 'while', 'because', 'since',
    'although', 'its', 'their', 'our', 'your', 'my', 'his', 'her',
    'this', 'these', 'those', 'such', 'very', 'also', 'not', 'no',
})


def _postprocess(text: str) -> str:
    if not text or not text.strip():
        return text

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove consecutive duplicate words
    words = text.split()
    if len(words) > 1:
        cleaned = [words[0]]
        for w in words[1:]:
            if w.lower().strip('.,!?;:') != cleaned[-1].lower().strip('.,!?;:'):
                cleaned.append(w)
        text = ' '.join(cleaned)

    # Remove comma spam
    text = re.sub(r'(,\s*){2,}', ', ', text)

    # Clean trailing dangling words
    words = text.split()
    removed = 0
    while words and removed < 3:
        last = words[-1].lower().strip('.,!?;:"\'-')
        if last in DANGLING_ENDINGS:
            words.pop()
            removed += 1
        else:
            break
    text = ' '.join(words) if words else text

    # Ensure final punctuation
    text = text.rstrip().rstrip(',;:-').rstrip()
    if text and text[-1] not in '.!?':
        text += '.'

    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text


# ============================================================================
# DETERMINISTIC FALLBACK ENGINES (original logic preserved)
# ============================================================================

FILLER_WORDS = frozenset({
    'very', 'really', 'actually', 'basically', 'essentially', 'literally',
    'honestly', 'totally', 'definitely', 'certainly', 'obviously', 'simply',
    'merely', 'quite', 'rather', 'somewhat', 'fairly', 'just', 'highly',
    'extremely', 'particularly', 'especially', 'specifically', 'generally',
    'practically', 'virtually', 'absolutely', 'truly', 'surely',
})

PHRASE_SHORTCUTS = [
    ('due to the fact that', 'because'),
    ('in order to', 'to'),
    ('has the ability to', 'can'),
    ('have the ability to', 'can'),
    ('at this point in time', 'now'),
    ('in the event that', 'if'),
    ('a large number of', 'many'),
    ('a significant number of', 'many'),
    ('for the purpose of', 'for'),
    ('in spite of the fact that', 'despite'),
    ('with regard to', 'regarding'),
    ('with respect to', 'regarding'),
    ('on the grounds that', 'because'),
    ('as a result of', 'from'),
    ('in the process of', 'while'),
    ('for the reason that', 'because'),
    ('in the near future', 'soon'),
    ('at the present time', 'now'),
    ('prior to', 'before'),
    ('subsequent to', 'after'),
    ('is able to', 'can'),
    ('are able to', 'can'),
    ('was able to', 'could'),
    ('it is important to note that', ''),
    ('it should be noted that', ''),
    ('the fact that', 'that'),
    ('whether or not', 'whether'),
    ('each and every', 'every'),
    ('first and foremost', 'first'),
    ('various different', 'various'),
    ('as well as', 'and'),
]
PHRASE_SHORTCUTS.sort(key=lambda x: -len(x[0]))

ABBREVIATION_MAP = {
    'AI': 'Artificial intelligence',
    'ML': 'Machine learning',
    'NLP': 'Natural language processing',
    'IT': 'Information technology',
    'HR': 'Human resources',
    'IoT': 'Internet of Things',
    'UI': 'User interface',
    'UX': 'User experience',
    'QA': 'Quality assurance',
}

TEMPORAL_WORDS = frozenset({
    'tomorrow', 'today', 'yesterday', 'tonight', 'morning', 'afternoon',
    'evening', 'night', 'monday', 'tuesday', 'wednesday', 'thursday',
    'friday', 'saturday', 'sunday', 'week', 'month', 'year',
})

TEMPORAL_PREFIXES = frozenset({'next', 'last', 'this', 'every'})


def _shorten_phrases(text):
    result = text
    for long_phrase, short_phrase in PHRASE_SHORTCUTS:
        pattern = re.compile(re.escape(long_phrase), re.IGNORECASE)
        result = pattern.sub(short_phrase, result)
    return re.sub(r'\s+', ' ', result).strip()


def _remove_fillers(text):
    words = text.split()
    if len(words) <= 3:
        return text
    result = []
    for i, word in enumerate(words):
        clean = word.lower().strip('.,!?;:"\'-')
        if clean in FILLER_WORDS and i > 0:
            continue
        result.append(word)
    return ' '.join(result)


def _score_sentence(sent, position, total, critical_entities=None):
    doc = _get_doc(sent)
    words = sent.split()
    if not words:
        return 0.0
        
    critical_words = set()
    if critical_entities:
        for ent in critical_entities:
            for w in ent.split():
                critical_words.add(w.lower().strip('.,!?;:'))
                
    content = 0
    if doc:
        for token in doc:
            word_clean = token.text.lower().strip('.,!?;:')
            if word_clean in critical_words or token.ent_type_ in ('TIME', 'DATE', 'GPE', 'LOC', 'PERSON', 'ORG', 'MONEY', 'CARDINAL', 'QUANTITY'):
                content += 5 # High boost for critical entities
            elif token.pos_ in ('NOUN', 'PROPN', 'VERB', 'NUM'):
                content += 1
    else:
        content = sum(1 for w in words if len(w) > 3)
        
    density = content / len(words)
    bonus = 1.3 if position == 0 else 1.0
    return density * bonus


def _compress_single_sentence(text, target, critical_entities=None):
    doc = _get_doc(text)
    words = text.split()
    if not doc or len(words) <= target:
        return ' '.join(words[:target])
        
    critical_words = set()
    if critical_entities:
        for ent in critical_entities:
            for w in ent.split():
                critical_words.add(w.lower().strip('.,!?;:'))
                
    scored = []
    for i, token in enumerate(doc):
        if i >= len(words):
            break
            
        word_clean = words[i].lower().strip('.,!?;:')
        if word_clean in critical_words or token.ent_type_ in ('TIME', 'DATE', 'GPE', 'LOC', 'PERSON', 'ORG', 'MONEY', 'CARDINAL', 'QUANTITY'):
            score = 100.0
        elif token.pos_ in ('NOUN', 'PROPN', 'NUM'):
            score = 5.0
        elif token.pos_ == 'VERB' and token.dep_ != 'aux':
            score = 4.0
        elif token.pos_ in ('ADP', 'CCONJ', 'SCONJ'):
            score = 2.5
        elif token.pos_ == 'DET':
            score = 1.5
        elif token.pos_ == 'ADJ':
            score = 2.0
        elif token.pos_ == 'ADV':
            score = 0.5
        elif token.pos_ == 'PUNCT':
            score = 0.1
        else:
            score = 1.0
            
        if i == 0:
            score *= 1.5
        scored.append((score, i))
        
    scored.sort(key=lambda x: -x[0])
    keep = set(s[1] for s in scored[:target])
    result = [words[i] for i in sorted(keep) if i < len(words)]
    return ' '.join(result)


def _smart_trim(text, target, tolerance, critical_entities=None):
    words = text.split()
    if len(words) <= target:
        return text
    candidate_words = words[:target + tolerance]
    for i in range(min(len(candidate_words) - 1, target + tolerance - 1),
                   max(0, target - tolerance - 1), -1):
        if candidate_words[i].rstrip().endswith(('.', '!', '?')):
            candidate = ' '.join(candidate_words[:i + 1])
            if abs(_count_words(candidate) - target) <= tolerance:
                return candidate
    result = ' '.join(words[:target])
    rwords = result.split()
    removed = 0
    while rwords and removed < 2:
        if rwords[-1].lower().strip('.,!?;:') in DANGLING_ENDINGS:
            rwords.pop()
            removed += 1
        else:
            break
    result = ' '.join(rwords)
    if result and result[-1] not in '.!?':
        result = result.rstrip(',;:-') + '.'
    return result


def _fallback_compress(text, target, info):
    tolerance = info['tolerance']
    critical_entities = info['critical_entities']
    
    result = _shorten_phrases(text)
    result = _remove_fillers(result)
    if abs(_count_words(result) - target) <= tolerance:
        return result
    current = _count_words(result)
    if current > target + tolerance:
        sentences = get_sentences(result)
        if len(sentences) > 1:
            scored = []
            for i, s in enumerate(sentences):
                sc = _score_sentence(s, i, len(sentences), critical_entities)
                scored.append((sc, i, s))
            scored.sort(key=lambda x: -x[0])
            selected = set()
            running = 0
            for sc, idx, s in scored:
                wc = _count_words(s)
                if running + wc <= target + tolerance:
                    selected.add(idx)
                    running += wc
            if not selected:
                selected.add(scored[0][1])
            result = ' '.join(sentences[i] for i in sorted(selected))
            if _count_words(result) > target + tolerance:
                result = _compress_single_sentence(result, target, critical_entities)
        else:
            result = _compress_single_sentence(result, target, critical_entities)
    if _count_words(result) > target + tolerance:
        result = _smart_trim(result, target, tolerance, critical_entities)
    return result


def _fallback_expand(text, target, info):
    tolerance = info['tolerance']
    result = text
    # Expand abbreviations
    words = result.split()
    expanded = []
    for word in words:
        core = word.rstrip('.,!?;:"\'-')
        suffix = word[len(core):]
        if core in ABBREVIATION_MAP:
            expanded.append(ABBREVIATION_MAP[core] + suffix)
        else:
            expanded.append(word)
    result = ' '.join(expanded)
    if _count_words(result) >= target - tolerance:
        return result
    # Structurize fragments
    doc = _get_doc(result)
    if doc and not any(t.pos_ == 'VERB' for t in doc):
        wds = result.split()
        wds_lower = [w.lower().strip('.,!?;:') for w in wds]
        subject_parts, time_parts = [], []
        in_time = False
        for i, wl in enumerate(wds_lower):
            if wl in TEMPORAL_PREFIXES and i + 1 < len(wds_lower) and wds_lower[i + 1] in TEMPORAL_WORDS:
                in_time = True
            if wl in TEMPORAL_WORDS or in_time:
                time_parts.append(wds[i])
                in_time = False
            else:
                subject_parts.append(wds[i])
        if subject_parts and time_parts:
            subj = ' '.join(subject_parts).strip('.,!?')
            tm = ' '.join(time_parts).strip('.,!?')
            if doc[0].pos_ not in ('DET', 'PROPN'):
                subj = 'The ' + subj[0].lower() + subj[1:]
            result = f"{subj} is scheduled for {tm}"
        elif subject_parts:
            subj = ' '.join(subject_parts).strip('.,!?')
            if doc[0].pos_ not in ('DET', 'PROPN'):
                subj = 'The ' + subj[0].lower() + subj[1:]
            result = f"{subj} is being planned"
    if _count_words(result) >= target - tolerance:
        return result
    # Add modifiers
    doc = _get_doc(result)
    if doc:
        noun_mods = ['important', 'key', 'relevant', 'comprehensive', 'effective', 'significant']
        verb_mods = ['effectively', 'efficiently', 'significantly', 'systematically']
        tokens = list(doc)
        res = []
        n_idx = v_idx = added = 0
        words_needed = target - _count_words(result)
        for token in tokens:
            if added >= words_needed:
                res.append(token.text)
                continue
            if token.pos_ == 'NOUN' and token.dep_ != 'compound':
                if not any(c.pos_ == 'ADJ' for c in token.children) and n_idx < len(noun_mods):
                    res.append(noun_mods[n_idx])
                    n_idx += 1
                    added += 1
            if token.pos_ == 'VERB' and token.dep_ != 'aux':
                if not any(c.pos_ == 'ADV' for c in token.children) and v_idx < len(verb_mods):
                    res.append(token.text)
                    res.append(verb_mods[v_idx])
                    v_idx += 1
                    added += 1
                    continue
            res.append(token.text)
        result = ' '.join(res)
    return result


def _fallback_refine(text):
    result = _shorten_phrases(text)
    result = _remove_fillers(result)
    return result


def _enforce_target(text, original, target, tolerance):
    count = _count_words(text)
    if abs(count - target) <= tolerance:
        return text
    if count > target + tolerance:
        return _smart_trim(text, target, tolerance, [])
    if count < target - tolerance:
        words_needed = target - count
        orig_words = original.split()
        current_lower = set(w.lower().strip('.,!?;:') for w in text.split())
        extras = []
        for w in orig_words:
            c = w.lower().strip('.,!?;:')
            if c and c not in current_lower and len(c) > 2:
                extras.append(w.strip('.,!?;:'))
                current_lower.add(c)
            if len(extras) >= words_needed:
                break
        if extras:
            base = text.rstrip('.!?')
            result = base + ' ' + ' '.join(extras[:words_needed]) + '.'
            if _count_words(result) > target + tolerance:
                result = _smart_trim(result, target, tolerance, [])
            return result
    return text


# ============================================================================
# INPUT ANALYSIS
# ============================================================================

def _analyze_input(text, target):
    original_count = _count_words(text)
    sentences = get_sentences(text)
    diff = target - original_count
    
    # Entity Extraction using spaCy
    doc = _get_doc(text)
    critical_entities = []
    if doc:
        for ent in doc.ents:
            if ent.label_ in ('TIME', 'DATE', 'GPE', 'LOC', 'PERSON', 'ORG', 'MONEY', 'CARDINAL', 'QUANTITY', 'FAC'):
                critical_entities.append(ent.text)
                
    # Regex fallback for time and money if spaCy misses them
    time_regex = re.findall(r'\b\d{1,2}:\d{2}(?:\s?[aApP][mM])?\b', text)
    money_regex = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
    critical_entities.extend(time_regex)
    critical_entities.extend(money_regex)
    
    # Deduplicate and sort to maintain stable output
    critical_entities = sorted(list(set(critical_entities)))
    
    # Dynamic tolerance logic
    tolerance = 3 if critical_entities else 2

    if abs(diff) <= tolerance:
        mode = 'refine'
    elif diff < 0:
        mode = 'compress'
    else:
        mode = 'expand'
        
    return {
        'original_count': original_count,
        'sentences': sentences,
        'sentence_count': len(sentences),
        'target': target,
        'diff': diff,
        'mode': mode,
        'critical_entities': critical_entities,
        'tolerance': tolerance
    }


# ============================================================================
# MAIN API
# ============================================================================

def analyze_length_and_rewrite(text: str, target_word_count: int) -> dict:
    """
    Rewrites text to target word count using Gemini AI with deterministic fallback.

    Pipeline:
      1. Analyze input -> determine mode and critical entities
      2. Try Gemini AI rewrite (up to 3 attempts) preserving entities
      3. If Gemini fails/unavailable, use deterministic fallback engine
      4. Post-process and enforce target ± tolerance
      5. Return compatible JSON response

    Returns dict compatible with existing frontend.
    """
    # Validate inputs
    if not text or not text.strip():
        return {
            "original_word_count": 0,
            "target_word_count": target_word_count,
            "rewritten_text": text or "",
            "final_word_count": 0,
            "deviation": 0,
            "status": "error",
            "message": "Empty input text"
        }

    target_word_count = max(3, target_word_count)
    info = _analyze_input(text, target_word_count)
    original_count = info['original_count']
    tolerance = info['tolerance']
    critical_entities = info['critical_entities']

    # Near-equal mode: quick refine
    if info['mode'] == 'refine':
        # Try Gemini even for refinement
        gemini_out = _gemini_rewrite(text, target_word_count, original_count, critical_entities, tolerance)
        if gemini_out:
            result = _postprocess(gemini_out)
        else:
            result = _postprocess(_fallback_refine(text))
        final_count = _count_words(result)
        return {
            "original_word_count": original_count,
            "target_word_count": target_word_count,
            "rewritten_text": result,
            "final_word_count": final_count,
            "deviation": final_count - target_word_count,
            "status": "success",
            "message": "Text refined — already near target length"
        }

    # --- Primary: Gemini AI rewrite ---
    used_gemini = False
    gemini_out = _gemini_rewrite(text, target_word_count, original_count, critical_entities, tolerance)

    if gemini_out:
        result = _postprocess(gemini_out)
        wc = _count_words(result)
        if abs(wc - target_word_count) <= tolerance:
            used_gemini = True
            logger.info(f"Gemini output accepted ({wc}w)")
        else:
            # Try enforcement on Gemini output
            enforced = _enforce_target(result, text, target_word_count, tolerance)
            enforced = _postprocess(enforced)
            ewc = _count_words(enforced)
            if abs(ewc - target_word_count) <= tolerance:
                result = enforced
                used_gemini = True
                logger.info(f"Gemini output enforced to {ewc}w")
            else:
                # Gemini was close but not perfect — still may be better than fallback
                result = enforced
                used_gemini = True
                logger.info(f"Gemini best effort: {ewc}w (target {target_word_count})")

    if not used_gemini:
        # --- Fallback: deterministic engine ---
        logger.info("Using fallback deterministic engine")
        if info['mode'] == 'compress':
            result = _fallback_compress(text, target_word_count, info)
        else:
            result = _fallback_expand(text, target_word_count, info)

        result = _postprocess(result)
        result = _enforce_target(result, text, target_word_count, tolerance)
        result = _postprocess(result)

    # Final response
    final_count = _count_words(result)
    deviation = final_count - target_word_count

    if abs(deviation) <= tolerance:
        status = "success"
        message = f"Rewritten to {final_count} words (target: {target_word_count}, ±{tolerance})"
    else:
        status = "partial"
        message = f"Best effort: {final_count} words (target: {target_word_count})"

    engine = "gemini" if used_gemini else "fallback"
    logger.info(f"Final: {final_count}w | target: {target_word_count} | engine: {engine} | status: {status} | entities: {len(critical_entities)}")

    return {
        "original_word_count": original_count,
        "target_word_count": target_word_count,
        "rewritten_text": result,
        "final_word_count": final_count,
        "deviation": deviation,
        "status": status,
        "message": message
    }
