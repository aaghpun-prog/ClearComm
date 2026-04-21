import nltk
import os
import re
from utils.preprocess import get_words, get_sentences

# Ensure NLTK data path is set
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

# ±2 word tolerance for "exact" length control
WORD_TOLERANCE = 2

# Maximum retry attempts for AI regeneration
MAX_RETRIES = 3


# =============================================================================
# RULE-BASED LENGTH ANALYSIS (for report_generator)
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
# POST-PROCESSING: STRICT WORD-COUNT CONTROLLER
# =============================================================================

def _count_words(text: str) -> int:
    """Count words using simple whitespace split (matches user-visible word count)."""
    return len(text.split())


def _trim_to_target(text: str, target: int) -> str:
    """
    Trim text to exactly `target` words.
    Cuts at the last full sentence boundary that fits, otherwise hard-cuts
    at the word limit and appends a period for grammatical closure.
    """
    words = text.split()
    if len(words) <= target:
        return text

    # Try to find a sentence boundary within the target window
    trimmed_words = words[:target]
    trimmed_text = ' '.join(trimmed_words)

    # Walk backwards from the cut point to find a sentence-ending punctuation
    for i in range(len(trimmed_words) - 1, max(0, len(trimmed_words) - 6), -1):
        if trimmed_words[i].endswith(('.', '!', '?')):
            candidate = ' '.join(trimmed_words[:i + 1])
            candidate_count = _count_words(candidate)
            # Only use sentence boundary if it's within tolerance
            if abs(candidate_count - target) <= WORD_TOLERANCE:
                return candidate

    # No good sentence boundary found — hard cut at target words
    result = ' '.join(words[:target])
    # Ensure it ends with a period for clean output
    if not result.endswith(('.', '!', '?')):
        # Remove trailing comma/semicolon if present
        result = result.rstrip(',;:')
        result += '.'
    return result


def _extend_to_target(text: str, original_text: str, target: int) -> str:
    """
    Extend text to reach `target` words by borrowing from the original text.

    Strategy:
      1. Clean up model output (remove repetitive junk tokens)
      2. Borrow meaningful words/phrases from the original
      3. If still short (extreme expansion), use elaboration padding
    """
    # Step 1: Clean repetitive junk from model output
    # e.g. "fix, fix, fix, fix" -> "fix"
    cleaned_words = []
    seen_runs = {}
    for w in text.split():
        clean = w.lower().strip('.,!?;:')
        seen_runs[clean] = seen_runs.get(clean, 0) + 1
        if seen_runs[clean] <= 2:  # allow at most 2 occurrences
            cleaned_words.append(w)

    text = ' '.join(cleaned_words)
    current_words = text.split()
    current_count = len(current_words)

    if current_count >= target:
        return text

    words_needed = target - current_count

    # Step 2: Collect unique words from original not in rewrite
    original_words = original_text.split()
    rewrite_lower = set(w.lower().strip('.,!?;:') for w in current_words)

    filler_candidates = []
    for word in original_words:
        clean = word.lower().strip('.,!?;:')
        if clean not in rewrite_lower and len(clean) > 1:
            filler_candidates.append(word.strip('.,!?;:'))
            rewrite_lower.add(clean)

    # Step 3: If still not enough words, cycle through ALL original words
    if len(filler_candidates) < words_needed:
        for word in original_words:
            clean = word.lower().strip('.,!?;:')
            if clean not in rewrite_lower and len(clean) > 1:
                filler_candidates.append(word.strip('.,!?;:'))
                rewrite_lower.add(clean)

    # Step 4: If original is extremely short, use elaboration padding
    PADDING_PHRASES = [
        "as soon as possible",
        "with high priority",
        "to ensure quality",
        "for the team",
        "in the system",
        "before the deadline",
        "and verify the results",
        "according to requirements",
        "with proper documentation",
        "for further review",
    ]

    padding_idx = 0
    while len(filler_candidates) < words_needed and padding_idx < len(PADDING_PHRASES):
        phrase_words = PADDING_PHRASES[padding_idx].split()
        filler_candidates.extend(phrase_words)
        padding_idx += 1

    # Build extension
    extension_words = filler_candidates[:words_needed]

    if not extension_words:
        return text

    # Ensure base text doesn't end with period before appending
    base = text.rstrip('.!?')
    extension = ' '.join(extension_words)

    result = f"{base} {extension}."
    return result


def _enforce_word_count(text: str, original_text: str, target: int) -> str:
    """
    Post-processing controller that guarantees output is within ±WORD_TOLERANCE
    of the target word count.
    """
    count = _count_words(text)

    # Already within tolerance
    if abs(count - target) <= WORD_TOLERANCE:
        return text

    # Too many words — trim
    if count > target + WORD_TOLERANCE:
        return _trim_to_target(text, target)

    # Too few words — extend
    if count < target - WORD_TOLERANCE:
        extended = _extend_to_target(text, original_text, target)
        ext_count = _count_words(extended)

        # If extension overshot, trim back
        if ext_count > target + WORD_TOLERANCE:
            return _trim_to_target(extended, target)
        return extended

    return text


# =============================================================================
# MAIN API: AI REWRITE WITH STRICT LENGTH ENFORCEMENT
# =============================================================================

def analyze_length_and_rewrite(text: str, target_word_count: int) -> dict:
    """
    Rewrites text to a strict word count using Flan-T5 + post-processing.

    Pipeline:
      1. Flan-T5 generates an initial rewrite aimed at the target
      2. Post-processor trims or extends to hit ±2 words of target
      3. If first attempt misses badly, retries with adjusted prompts

    Returns dict with rewritten_text, word counts, and metadata.
    """
    from models.transformer_loader import get_models

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

    target_word_count = max(3, target_word_count)  # Floor at 3 words

    # 1. Get original stats
    original_words = text.split()
    original_count = len(original_words)

    # Edge case: original is already within tolerance
    if abs(original_count - target_word_count) <= WORD_TOLERANCE:
        return {
            "original_word_count": original_count,
            "target_word_count": target_word_count,
            "rewritten_text": text,
            "final_word_count": original_count,
            "status": "success",
            "message": "Original text already meets target length"
        }

    # 2. Generate rewrite(s) with Flan-T5
    models = get_models()
    best_text = None
    best_diff = float('inf')

    for attempt in range(1, MAX_RETRIES + 1):
        # Adjust target slightly on retries to compensate for model drift
        adjusted_target = target_word_count
        if attempt == 2:
            # If first attempt was too long, ask for shorter; vice versa
            if best_text and _count_words(best_text) > target_word_count:
                adjusted_target = max(3, target_word_count - 5)
            else:
                adjusted_target = target_word_count + 5
        elif attempt == 3:
            # More aggressive adjustment
            if best_text and _count_words(best_text) > target_word_count:
                adjusted_target = max(3, target_word_count - 10)
            else:
                adjusted_target = target_word_count + 8

        raw_rewrite = models.rewrite_text(text, adjusted_target)
        raw_count = _count_words(raw_rewrite)
        diff = abs(raw_count - target_word_count)

        if diff < best_diff:
            best_diff = diff
            best_text = raw_rewrite

        # If already within tolerance, stop early
        if diff <= WORD_TOLERANCE:
            break

    # 3. Post-process to enforce strict word count
    final_text = _enforce_word_count(best_text, text, target_word_count)
    final_count = _count_words(final_text)
    diff = abs(final_count - target_word_count)

    # Determine status
    if diff <= WORD_TOLERANCE:
        status = "success"
        message = f"Rewritten to {final_count} words (target: {target_word_count}, tolerance: +/-{WORD_TOLERANCE})"
    else:
        status = "partial"
        message = f"Best effort: {final_count} words (target: {target_word_count}). Post-processing could not fully converge."

    return {
        "original_word_count": original_count,
        "target_word_count": target_word_count,
        "rewritten_text": final_text,
        "final_word_count": final_count,
        "deviation": final_count - target_word_count,
        "status": status,
        "message": message
    }
