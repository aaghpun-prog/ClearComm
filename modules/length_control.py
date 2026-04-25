"""
length_control.py — Practical Hybrid Length Control System

Architecture:
  Mode A: COMPRESSION (target < original) — deterministic extractive summarization
  Mode B: EXPANSION (target > original) — pattern-based elaboration
  Mode C: REFINEMENT (target ≈ original) — grammar/polish pass

T5 is used ONLY as optional secondary polish. All core logic is rule-based.
"""

import nltk
import os
import re
from utils.preprocess import get_words, get_sentences, get_spacy_doc

# Ensure NLTK data path is set
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

# ±2 word tolerance
WORD_TOLERANCE = 2

# ---------------------------------------------------------------------------
# LINGUISTIC RESOURCES
# ---------------------------------------------------------------------------

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
# Sort longest first so longer phrases match before shorter substrings
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

DANGLING_ENDINGS = frozenset({
    'and', 'but', 'or', 'nor', 'yet', 'so', 'for', 'with', 'without',
    'in', 'on', 'at', 'to', 'of', 'by', 'from', 'as', 'if', 'the',
    'a', 'an', 'is', 'are', 'was', 'were', 'be', 'that', 'which',
    'who', 'whose', 'where', 'when', 'while', 'because', 'since',
    'although', 'its', 'their', 'our', 'your', 'my', 'his', 'her',
    'this', 'these', 'those', 'such', 'very', 'also', 'not', 'no',
})


# =============================================================================
# RULE-BASED LENGTH ANALYSIS (for report_generator — PRESERVED)
# =============================================================================

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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _count_words(text: str) -> int:
    """Count words using whitespace split."""
    return len(text.split())


def _get_doc(text: str):
    """Get spaCy doc with graceful fallback."""
    try:
        return get_spacy_doc(text)
    except Exception:
        return None


# =============================================================================
# POST-PROCESSING PIPELINE (always applied)
# =============================================================================

def _postprocess(text: str) -> str:
    """
    Mandatory cleanup:
      1. Fix whitespace
      2. Remove consecutive duplicate words
      3. Remove comma spam
      4. Clean trailing fragments
      5. Ensure punctuation
      6. Capitalize first letter
    """
    if not text or not text.strip():
        return text

    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Remove consecutive duplicate words
    words = text.split()
    if len(words) > 1:
        cleaned = [words[0]]
        for w in words[1:]:
            if w.lower().strip('.,!?;:') != cleaned[-1].lower().strip('.,!?;:'):
                cleaned.append(w)
        text = ' '.join(cleaned)

    # 3. Remove comma spam (3+ commas in a row after splitting)
    text = re.sub(r'(,\s*){2,}', ', ', text)

    # 4. Clean trailing dangling words
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

    # 5. Ensure final punctuation
    text = text.rstrip()
    text = text.rstrip(',;:-')
    text = text.rstrip()
    if text and text[-1] not in '.!?':
        text += '.'

    # 6. Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


# =============================================================================
# INPUT ANALYSIS
# =============================================================================

def _analyze_input(text: str, target: int) -> dict:
    """Detect mode and gather input statistics."""
    original_count = _count_words(text)
    sentences = get_sentences(text)
    diff = target - original_count

    if abs(diff) <= WORD_TOLERANCE:
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
    }


# =============================================================================
# MODE A: COMPRESSION ENGINE
# =============================================================================

def _shorten_phrases(text: str) -> str:
    """Replace verbose phrases with shorter equivalents."""
    result = text
    for long_phrase, short_phrase in PHRASE_SHORTCUTS:
        pattern = re.compile(re.escape(long_phrase), re.IGNORECASE)
        result = pattern.sub(short_phrase, result)
    # Clean up double spaces from removals
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def _remove_fillers(text: str) -> str:
    """Remove filler/hedge words that add no meaning."""
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


def _score_sentence(sent: str, position: int, total: int) -> float:
    """Score sentence importance using POS tags + position."""
    doc = _get_doc(sent)
    words = sent.split()
    if not words:
        return 0.0

    # Content word density
    if doc:
        content = sum(1 for t in doc if t.pos_ in ('NOUN', 'PROPN', 'VERB', 'NUM'))
    else:
        content = sum(1 for w in words if len(w) > 3)

    density = content / len(words)

    # Position bonus: first sentence usually most important
    bonus = 1.3 if position == 0 else 1.0
    return density * bonus


def _compress_single_sentence(text: str, target: int) -> str:
    """Compress one sentence by removing least important words."""
    doc = _get_doc(text)
    words = text.split()

    if not doc or len(words) <= target:
        return ' '.join(words[:target])

    # Score each token by POS importance
    scored = []
    for i, token in enumerate(doc):
        if i >= len(words):
            break
        if token.pos_ in ('NOUN', 'PROPN', 'NUM'):
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
        # Boost first/last tokens for sentence structure
        if i == 0:
            score *= 1.5
        scored.append((score, i))

    # Keep the top-N most important words, preserving original order
    scored.sort(key=lambda x: -x[0])
    keep = set(s[1] for s in scored[:target])
    result = [words[i] for i in sorted(keep) if i < len(words)]
    return ' '.join(result)


def _compress(text: str, target: int, info: dict) -> str:
    """
    Deterministic compression:
      1. Shorten verbose phrases
      2. Remove filler words
      3. Select important sentences (if multi-sentence)
      4. Compress at word level if single sentence
      5. Smart trim to target
    """
    result = _shorten_phrases(text)
    result = _remove_fillers(result)

    if abs(_count_words(result) - target) <= WORD_TOLERANCE:
        return result

    current = _count_words(result)

    if current > target + WORD_TOLERANCE:
        sentences = get_sentences(result)

        if len(sentences) > 1:
            # Score sentences and greedily select the best ones
            scored = []
            for i, s in enumerate(sentences):
                sc = _score_sentence(s, i, len(sentences))
                scored.append((sc, i, s))
            scored.sort(key=lambda x: -x[0])

            selected = set()
            running = 0
            for sc, idx, s in scored:
                wc = _count_words(s)
                if running + wc <= target + WORD_TOLERANCE:
                    selected.add(idx)
                    running += wc
            if not selected:
                selected.add(scored[0][1])

            result = ' '.join(sentences[i] for i in sorted(selected))

            # If still over, compress the combined text at word level
            if _count_words(result) > target + WORD_TOLERANCE:
                result = _compress_single_sentence(result, target)
        else:
            result = _compress_single_sentence(result, target)

    # Final smart trim
    if _count_words(result) > target + WORD_TOLERANCE:
        result = _smart_trim(result, target)

    return result


# =============================================================================
# MODE B: EXPANSION ENGINE
# =============================================================================

def _expand_abbreviations(text: str) -> str:
    """Expand known abbreviations to full forms."""
    words = text.split()
    result = []
    for word in words:
        # Separate trailing punctuation
        core = word.rstrip('.,!?;:"\'-')
        suffix = word[len(core):]
        if core in ABBREVIATION_MAP:
            result.append(ABBREVIATION_MAP[core] + suffix)
        else:
            result.append(word)
    return ' '.join(result)


def _is_fragment(text: str) -> bool:
    """Check if text is a sentence fragment (no main verb)."""
    doc = _get_doc(text)
    if doc is None:
        return _count_words(text) <= 4
    return not any(t.pos_ == 'VERB' for t in doc)


def _has_temporal(text: str) -> bool:
    """Check if text contains temporal references."""
    words_lower = set(w.lower().strip('.,!?;:') for w in text.split())
    return bool(words_lower & TEMPORAL_WORDS)


def _structurize_fragment(text: str) -> str:
    """Convert a fragment into a complete sentence."""
    doc = _get_doc(text)
    if doc is None:
        return text

    words = text.split()
    words_lower = [w.lower().strip('.,!?;:') for w in words]

    # Separate subject part from temporal part
    subject_parts = []
    time_parts = []
    in_time = False

    for i, wl in enumerate(words_lower):
        if wl in TEMPORAL_PREFIXES and i + 1 < len(words_lower) and words_lower[i + 1] in TEMPORAL_WORDS:
            in_time = True
        if wl in TEMPORAL_WORDS or in_time:
            time_parts.append(words[i])
            in_time = False
        else:
            subject_parts.append(words[i])

    if subject_parts and time_parts:
        subj = ' '.join(subject_parts).strip('.,!?')
        time = ' '.join(time_parts).strip('.,!?')
        # Add article if subject doesn't start with one
        if doc[0].pos_ != 'DET' and doc[0].pos_ != 'PROPN':
            subj = 'The ' + subj[0].lower() + subj[1:]
        return f"{subj} is scheduled for {time}"
    elif subject_parts:
        subj = ' '.join(subject_parts).strip('.,!?')
        if doc[0].pos_ != 'DET' and doc[0].pos_ != 'PROPN':
            subj = 'The ' + subj[0].lower() + subj[1:]
        return f"{subj} is being planned"

    return text


def _add_modifiers(text: str, words_needed: int) -> str:
    """Add adjectives/adverbs before key nouns/verbs to increase word count."""
    doc = _get_doc(text)
    if doc is None or words_needed <= 0:
        return text

    noun_mods = ['important', 'key', 'relevant', 'comprehensive',
                 'effective', 'significant', 'essential', 'valuable']
    verb_mods = ['effectively', 'efficiently', 'significantly',
                 'systematically', 'proactively', 'successfully']

    tokens = list(doc)
    result = []
    n_idx = 0
    v_idx = 0
    added = 0

    for token in tokens:
        if added >= words_needed:
            result.append(token.text)
            continue

        # Add adjective before unmodified nouns
        if token.pos_ in ('NOUN',) and token.dep_ not in ('compound',):
            has_adj = any(c.pos_ == 'ADJ' for c in token.children)
            if not has_adj and n_idx < len(noun_mods):
                result.append(noun_mods[n_idx])
                n_idx += 1
                added += 1

        # Add adverb after verbs
        if token.pos_ == 'VERB' and token.dep_ != 'aux':
            has_adv = any(c.pos_ == 'ADV' for c in token.children)
            if not has_adv and v_idx < len(verb_mods):
                result.append(token.text)
                result.append(verb_mods[v_idx])
                v_idx += 1
                added += 1
                continue

        result.append(token.text)

    return ' '.join(result)


def _expand(text: str, target: int, info: dict) -> str:
    """
    Deterministic expansion:
      1. Expand abbreviations
      2. Structurize fragments (add verb/article)
      3. Add contextual modifiers
    """
    result = _expand_abbreviations(text)

    if _count_words(result) >= target - WORD_TOLERANCE:
        return result

    # If it's a fragment, build a full sentence
    if _is_fragment(result):
        result = _structurize_fragment(result)

    if _count_words(result) >= target - WORD_TOLERANCE:
        return result

    # Add modifiers to reach target
    words_needed = target - _count_words(result)
    if words_needed > 0:
        result = _add_modifiers(result, words_needed)

    return result


# =============================================================================
# MODE C: REFINEMENT ENGINE
# =============================================================================

def _refine(text: str) -> str:
    """Polish grammar and readability without changing length much."""
    result = _shorten_phrases(text)
    result = _remove_fillers(result)
    return result


# =============================================================================
# SMART TRIM (sentence-boundary aware)
# =============================================================================

def _smart_trim(text: str, target: int) -> str:
    """Trim to target words, preferring sentence boundaries."""
    words = text.split()
    if len(words) <= target:
        return text

    # Look for sentence boundary near target
    candidate_words = words[:target + WORD_TOLERANCE]
    for i in range(min(len(candidate_words) - 1, target + WORD_TOLERANCE - 1),
                   max(0, target - WORD_TOLERANCE - 1), -1):
        if candidate_words[i].rstrip().endswith(('.', '!', '?')):
            candidate = ' '.join(candidate_words[:i + 1])
            if abs(_count_words(candidate) - target) <= WORD_TOLERANCE:
                return candidate

    # No good boundary — hard cut + cleanup
    result = ' '.join(words[:target])
    # Remove trailing dangling words
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


# =============================================================================
# OPTIONAL T5 POLISH (secondary only)
# =============================================================================

def _try_t5_polish(text: str, original: str, target: int) -> str | None:
    """
    Try Flan-T5 as optional polish. Returns None if output is bad.
    Used ONLY when deterministic output needs smoothing.
    """
    try:
        from models.transformer_loader import get_models
        models = get_models()
        raw = models.rewrite_text(original, target)

        if not raw or not raw.strip():
            return None

        raw_wc = _count_words(raw)

        # Reject if T5 output is terrible
        if abs(raw_wc - target) > max(target * 0.4, 8):
            return None  # Way off target
        if raw.strip().lower() == original.strip().lower():
            return None  # Just copied input
        if _has_excessive_repetition(raw):
            return None  # Nonsense repetition

        return raw
    except Exception:
        return None


def _has_excessive_repetition(text: str) -> bool:
    """Check if any content word repeats more than 3 times."""
    stop = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'is',
            'are', 'was', 'were', 'with', 'that', 'this', 'it', 'on', 'by', 'as'}
    counts = {}
    for w in text.lower().split():
        c = w.strip('.,!?;:')
        if c and c not in stop:
            counts[c] = counts.get(c, 0) + 1
    return any(v > 3 for v in counts.values())


# =============================================================================
# TARGET ENFORCEMENT
# =============================================================================

def _enforce_target(text: str, original: str, target: int) -> str:
    """Ensure output is within ±WORD_TOLERANCE of target."""
    count = _count_words(text)

    if abs(count - target) <= WORD_TOLERANCE:
        return text

    if count > target + WORD_TOLERANCE:
        return _smart_trim(text, target)

    if count < target - WORD_TOLERANCE:
        # Try to extend by borrowing words from original
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
            if _count_words(result) > target + WORD_TOLERANCE:
                result = _smart_trim(result, target)
            return result

    return text


# =============================================================================
# MAIN API
# =============================================================================

def analyze_length_and_rewrite(text: str, target_word_count: int) -> dict:
    """
    Rewrites text to target word count using a practical hybrid system.

    Pipeline:
      1. Analyze input → determine mode (compress/expand/refine)
      2. Run deterministic mode-specific engine
      3. Optionally try T5 polish (secondary, only if result needs it)
      4. Apply post-processing
      5. Enforce target ±2

    Returns dict compatible with existing frontend.
    """
    # Validate inputs
    if not text or not text.strip():
        return {
            "original_word_count": 0,
            "target_word_count": target_word_count,
            "rewritten_text": text or "",
            "final_word_count": 0,
            "status": "error",
            "message": "Empty input text"
        }

    target_word_count = max(3, target_word_count)

    # 1. Analyze input
    info = _analyze_input(text, target_word_count)
    original_count = info['original_count']

    # Edge case: already within tolerance
    if info['mode'] == 'refine':
        refined = _postprocess(_refine(text))
        final_count = _count_words(refined)
        return {
            "original_word_count": original_count,
            "target_word_count": target_word_count,
            "rewritten_text": refined,
            "final_word_count": final_count,
            "deviation": final_count - target_word_count,
            "status": "success",
            "message": "Text refined — already near target length"
        }

    # 2. Run deterministic engine based on mode
    if info['mode'] == 'compress':
        result = _compress(text, target_word_count, info)
    else:
        result = _expand(text, target_word_count, info)

    # 3. Post-process
    result = _postprocess(result)

    # 4. Try T5 polish if deterministic result is far from target
    det_diff = abs(_count_words(result) - target_word_count)
    if det_diff > WORD_TOLERANCE:
        t5_output = _try_t5_polish(result, text, target_word_count)
        if t5_output:
            t5_processed = _postprocess(t5_output)
            t5_diff = abs(_count_words(t5_processed) - target_word_count)
            # Use T5 only if it's actually better
            if t5_diff < det_diff:
                result = t5_processed

    # 5. Enforce strict target
    result = _enforce_target(result, text, target_word_count)
    result = _postprocess(result)

    # 6. Build response
    final_count = _count_words(result)
    deviation = final_count - target_word_count

    if abs(deviation) <= WORD_TOLERANCE:
        status = "success"
        message = f"Rewritten to {final_count} words (target: {target_word_count}, ±{WORD_TOLERANCE})"
    else:
        status = "partial"
        message = f"Best effort: {final_count} words (target: {target_word_count})"

    return {
        "original_word_count": original_count,
        "target_word_count": target_word_count,
        "rewritten_text": result,
        "final_word_count": final_count,
        "deviation": deviation,
        "status": status,
        "message": message
    }
