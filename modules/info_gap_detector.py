"""
info_gap_detector.py — Smart Hybrid Information Gap Detection System

Architecture:
  Layer 1: Intent classification (keyword + spaCy-based)
  Layer 2: Multi-entity extraction (spaCy NER + regex + keyword fallback)
  Layer 3: Per-intent gap rules (what's required vs what's present)
  Layer 4: Multi-sentence support (analyze each sentence independently)
  Layer 5: Confidence filtering (skip casual/general statements)

Supports 8 message types:
  Meeting, Event, Task, Interview, Workshop, Travel, Payment, Emergency
"""

import re
from utils.preprocess import get_spacy_doc, get_sentences

# ==============================================================================
# ENTITY DETECTION PATTERNS
# ==============================================================================

RE_TIME = re.compile(
    r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b'
    r'|\b(?:noon|midnight|morning|afternoon|evening)\b',
    re.IGNORECASE
)

RE_DATE = re.compile(
    r'\b(?:today|tomorrow|yesterday|tonight)\b'
    r'|\b(?:next|last|this|every)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
    r'|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
    r'|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    r'|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b',
    re.IGNORECASE
)

RE_EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
RE_PHONE = re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')
RE_URL = re.compile(r'https?://\S+|www\.\S+')

RE_MONEY = re.compile(
    r'[\$£€₹]\s*\d+(?:,\d{3})*(?:\.\d{1,2})?'
    r'|\b\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars|rupees|rs|inr|usd|eur|gbp)\b'
    r'|\brs\.?\s*\d+'
    r'|\b₹\d+',
    re.IGNORECASE
)

LOCATION_KEYWORDS = frozenset({
    'room', 'office', 'venue', 'hall', 'building', 'campus', 'auditorium',
    'lab', 'library', 'center', 'centre', 'floor', 'block', 'wing',
    'stadium', 'theater', 'theatre', 'cafeteria', 'canteen', 'park',
    'ground', 'field', 'court', 'gym', 'headquarters', 'branch',
})

ONLINE_MODE_KEYWORDS = [
    'google meet', 'video call', 'in-person', 'on-site',
    'online', 'zoom', 'teams', 'webex', 'skype',
    'virtual', 'remote', 'offline', 'hybrid',
]


# ==============================================================================
# INTENT CLASSIFICATION
# ==============================================================================

# Words that signal a meeting when they appear before meeting/sync/call keywords
_MEETING_TRIGGER_WORDS = frozenset({
    'meeting', 'sync', 'call', 'meet', 'standup', 'huddle', 'meetup',
    'discussion', 'catch up',
})

INTENT_KEYWORDS = {
    "Meeting": {
        'keywords': ['meeting', 'sync', 'standup', 'stand-up', 'catch up',
                     'discuss', 'discussion', 'huddle', 'meetup',
                     'meet', 'call'],
        'weight': 2,
    },
    "Event": {
        'keywords': ['event', 'fest', 'festival', 'conference', 'ceremony',
                     'celebration', 'function', 'gathering', 'inaugural',
                     'annual', 'competition', 'contest', 'hackathon'],
        'weight': 2,
    },
    "Task": {
        'keywords': ['submit', 'update', 'complete', 'finish', 'prepare',
                     'review', 'send', 'fix', 'resolve', 'implement',
                     'upload', 'deliver', 'deploy', 'report', 'assignment',
                     'task', 'need to', 'must', 'required', 'please',
                     'ensure', 'verify', 'check'],
        'weight': 1,
    },
    "Interview": {
        'keywords': ['interview', 'hiring', 'recruitment', 'job', 'position',
                     'role', 'vacancy', 'candidate', 'applicant', 'resume',
                     'placement', 'selection'],
        'weight': 2,
    },
    "Workshop": {
        'keywords': ['workshop', 'seminar', 'webinar', 'tutorial', 'training',
                     'session', 'lecture', 'course', 'masterclass', 'bootcamp',
                     'demonstration', 'orientation'],
        'weight': 2,
    },
    "Travel": {
        'keywords': ['visit', 'travel', 'trip', 'tour', 'go to', 'fly',
                     'drive', 'commute', 'journey', 'outing', 'field trip',
                     'site visit', 'client visit'],
        'weight': 2,
    },
    "Payment": {
        'keywords': ['pay', 'payment', 'fees', 'fee', 'purchase', 'buy',
                     'cost', 'charge', 'invoice', 'bill', 'dues', 'rent',
                     'subscription', 'deposit', 'transfer', 'reimburse',
                     'sale', 'selling', 'sell', 'discount', 'offer', 'price'],
        'weight': 1,
    },
    "Emergency": {
        'keywords': ['urgent', 'emergency', 'critical', 'asap', 'immediately',
                     'right now', 'priority', 'alert', 'warning', 'danger',
                     'outage', 'downtime', 'incident', 'escalate'],
        'weight': 2,
    },
}

# What fields each intent requires
INTENT_REQUIRED_FIELDS = {
    "Meeting": {
        'fields': ['time', 'location'],
        'labels': {
            'time': 'Time',
            'location': 'Venue/Location',
        },
    },
    "Event": {
        'fields': ['date', 'time', 'location'],
        'labels': {
            'date': 'Date',
            'time': 'Time',
            'location': 'Venue',
        },
    },
    "Task": {
        'fields': ['deadline', 'assignee'],
        'labels': {
            'deadline': 'Deadline',
            'assignee': 'Assignee',
        },
    },
    "Interview": {
        'fields': ['date', 'time', 'location'],
        'labels': {
            'date': 'Date',
            'time': 'Time',
            'location': 'Venue/Mode',
        },
    },
    "Workshop": {
        'fields': ['date', 'time', 'location'],
        'labels': {
            'date': 'Date',
            'time': 'Time',
            'location': 'Venue',
        },
    },
    "Travel": {
        'fields': ['date', 'time', 'destination'],
        'labels': {
            'date': 'Date',
            'time': 'Time',
            'destination': 'Destination',
        },
    },
    "Payment": {
        'fields': ['amount', 'deadline', 'recipient'],
        'labels': {
            'amount': 'Amount',
            'deadline': 'Deadline',
            'recipient': 'Recipient',
        },
    },
    "Emergency": {
        'fields': ['action', 'contact'],
        'labels': {
            'action': 'Required Action',
            'contact': 'Contact Person',
        },
    },
}

# Casual phrases — skip entirely
CASUAL_KEYWORDS = [
    'hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon',
    'good night', 'thanks', 'thank you', 'bye', 'goodbye', 'see you',
    'nice weather', 'how are you', 'what\'s up', 'take care', 'well done',
    'congratulations', 'happy birthday', 'cheers',
]

# General statement indicators (not actionable)
GENERAL_INDICATORS = frozenset({
    'is', 'are', 'was', 'were', 'has been', 'have been', 'will be',
})


# ==============================================================================
# ENTITY EXTRACTION
# ==============================================================================

def _extract_entities(text: str) -> dict:
    """
    Extract all information entities using spaCy NER + regex + keyword fallback.
    Returns a dict of booleans indicating presence of each entity type.
    """
    doc = get_spacy_doc(text)
    text_lower = text.lower()

    entities = {
        'time': bool(RE_TIME.search(text)),
        'date': bool(RE_DATE.search(text)),
        'location': False,
        'person': False,
        'money': bool(RE_MONEY.search(text)),
        'contact': bool(RE_EMAIL.search(text) or RE_PHONE.search(text)),
        'link': bool(RE_URL.search(text)),
        'mode': any(re.search(rf'\b{re.escape(m)}\b', text_lower) for m in ONLINE_MODE_KEYWORDS),
        'organization': False,
    }

    # spaCy NER extraction
    if doc:
        for ent in doc.ents:
            if ent.label_ == 'TIME':
                entities['time'] = True
            elif ent.label_ == 'DATE':
                entities['date'] = True
            elif ent.label_ in ('GPE', 'LOC', 'FAC'):
                entities['location'] = True
            elif ent.label_ == 'MONEY':
                entities['money'] = True
            elif ent.label_ == 'PERSON':
                entities['person'] = True
            elif ent.label_ == 'ORG':
                entities['organization'] = True

    # Keyword fallback for locations
    if not entities['location']:
        for kw in LOCATION_KEYWORDS:
            if re.search(rf'\b{kw}\b', text_lower):
                entities['location'] = True
                break

    # Money keyword fallback
    if not entities['money']:
        if any(w in text_lower for w in ['$', '£', '€', '₹', 'dollars', 'rupees', 'cost', 'price', 'amount']):
            entities['money'] = True

    # Contact keyword fallback
    if not entities['contact'] and 'contact' in text_lower:
        entities['contact'] = True

    return entities


# ==============================================================================
# INTENT CLASSIFICATION
# ==============================================================================

def _is_casual(text: str) -> bool:
    """Filter out non-actionable pleasantries and greetings."""
    text_lower = text.lower().strip()
    words = text_lower.split()
    if len(words) <= 6:
        for cw in CASUAL_KEYWORDS:
            if cw in text_lower:
                return True
    return False


def _is_general_statement(text: str) -> bool:
    """Detect general factual statements that don't need gap analysis."""
    doc = get_spacy_doc(text)
    if not doc:
        return False

    text_lower = text.lower()

    # Check for imperative (starts with verb) — NOT general
    if doc[0].pos_ == 'VERB':
        return False

    # Check for any intent keywords — NOT general
    for intent_data in INTENT_KEYWORDS.values():
        for kw in intent_data['keywords']:
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                return False

    # If no action keywords and has typical general structure, it's general
    has_action_verb = any(
        t.pos_ == 'VERB' and t.dep_ in ('ROOT',) and t.lemma_ not in (
            'be', 'have', 'do', 'seem', 'appear', 'look', 'feel', 'become'
        )
        for t in doc
    )

    # General facts typically have "is/are" as main verb
    has_copula = any(t.lemma_ == 'be' and t.dep_ == 'ROOT' for t in doc)
    if has_copula and not has_action_verb:
        return True

    return False


def _classify_intent(text: str) -> tuple:
    """
    Classify message intent using keyword matching.
    Returns (intent_name: str | None, confidence: str).

    Tie-breaking: If both Meeting and Task score, prefer Meeting when the
    sentence also contains time/date/mode indicators (Rule #2).
    """
    text_lower = text.lower()
    scores = {}

    for intent, data in INTENT_KEYWORDS.items():
        score = 0
        for kw in data['keywords']:
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                score += data['weight']
        if score > 0:
            scores[intent] = score

    if not scores:
        return None, 'Low'

    # Rule #2: Meeting wins over Task when meet/call/zoom + time/date present
    if 'Meeting' in scores and 'Task' in scores:
        has_time_or_date = bool(RE_TIME.search(text) or RE_DATE.search(text))
        has_mode = any(re.search(rf'\b{re.escape(m)}\b', text_lower) for m in ONLINE_MODE_KEYWORDS)
        if has_time_or_date or has_mode:
            scores['Meeting'] += 3  # Strong boost

    # Pick highest scoring intent
    best = max(scores, key=scores.get)
    best_score = scores[best]

    # Confidence based on score
    if best_score >= 4:
        confidence = 'High'
    elif best_score >= 2:
        confidence = 'High'
    else:
        confidence = 'Medium'

    return best, confidence


def _classify_intent_fallback(text: str) -> tuple:
    """
    Fallback intent classification using spaCy dependency parsing.
    Detects imperative sentences (task commands).
    """
    doc = get_spacy_doc(text)
    if not doc or len(doc) == 0:
        return None, 'Low'

    # Imperative sentence: starts with verb
    if doc[0].pos_ == 'VERB':
        return 'Task', 'Medium'

    # Check for "need to", "have to" patterns
    text_lower = text.lower()
    if any(p in text_lower for p in ['need to', 'have to', 'must', 'should']):
        return 'Task', 'Medium'

    return None, 'Low'


# ==============================================================================
# GAP DETECTION LOGIC
# ==============================================================================

def _check_field_present(field: str, entities: dict, text_lower: str) -> bool:
    """Check if a required field is present in the extracted entities or text."""

    if field == 'date':
        return entities['date']

    elif field == 'time':
        return entities['time']

    elif field in ('location', 'destination'):
        return entities['location'] or entities['mode']

    elif field == 'assignee':
        return entities['person'] or entities['organization']

    elif field == 'deadline':
        return entities['date'] or entities['time']

    elif field == 'amount':
        return entities['money']

    elif field == 'recipient':
        return entities['person'] or entities['organization'] or entities['contact']

    elif field == 'contact':
        return entities['contact'] or entities['person'] or entities['link']

    elif field == 'action':
        # For emergency: check if a specific action is mentioned
        doc = get_spacy_doc(text_lower)
        if doc:
            return any(t.pos_ == 'VERB' and t.dep_ == 'ROOT'
                      and t.lemma_ not in ('be', 'have') for t in doc)
        return False

    elif field == 'agenda':
        # Check if purpose/topic is mentioned via explicit keywords
        purpose_keywords = ['about', 'regarding', 'for', 'to discuss',
                          'to review', 'to plan', 'on topic', 'agenda']
        if any(kw in text_lower for kw in purpose_keywords):
            return True

        # Rule #1: Topic words BEFORE a meeting trigger word count as agenda.
        # e.g. "project sync meeting" → "project" is the topic/agenda.
        for trigger in _MEETING_TRIGGER_WORDS:
            pattern = re.search(rf'(\S+(?:\s+\S+){{0,3}})\s+{re.escape(trigger)}\b', text_lower)
            if pattern:
                prefix = pattern.group(1).strip()
                # Filter out non-topic prefixes (articles, pronouns, "let's", etc.)
                noise = {'the', 'a', 'an', 'our', 'my', 'your', 'their', 'its',
                         'this', 'that', 'team', 'group', 'let', "let's", 'lets',
                         'we', 'i', 'he', 'she', 'they', 'weekly', 'daily',
                         'monthly', 'annual', 'quick', 'brief', 'short',
                         'scheduled', 'upcoming', 'next', 'catch'}
                prefix_words = [w for w in prefix.split() if w.lower() not in noise]
                if prefix_words:
                    return True  # Found topic words before meeting keyword

        return False

    return False


def _detect_gaps_for_sentence(sentence: str) -> dict | None:
    """
    Detect information gaps for a single sentence.
    Returns a gap dict or None if no gaps found.
    """
    text = sentence.strip()
    if not text:
        return None

    # Skip casual messages
    if _is_casual(text):
        return None

    # Skip general factual statements
    if _is_general_statement(text):
        return None

    text_lower = text.lower()

    # Extract entities
    entities = _extract_entities(text)

    # Classify intent
    intent, confidence = _classify_intent(text)

    # Fallback classification
    if not intent:
        intent, confidence = _classify_intent_fallback(text)

    # If still no intent, skip
    if not intent:
        return None

    # Get required fields for this intent
    req = INTENT_REQUIRED_FIELDS.get(intent)
    if not req:
        return None

    # Check which required fields are missing
    missing_labels = []
    for field in req['fields']:
        if not _check_field_present(field, entities, text_lower):
            missing_labels.append(req['labels'][field])

    if not missing_labels:
        return None

    missing_str = ', '.join(missing_labels)

    # Build suggestion
    suggestion_parts = [m.lower() for m in missing_labels]
    if len(suggestion_parts) > 1:
        suggestion = f"Please include {', '.join(suggestion_parts[:-1])} and {suggestion_parts[-1]}."
    else:
        suggestion = f"Please include {suggestion_parts[0]}."

    return {
        'sentence': text,
        'missing': missing_str,
        'confidence': confidence,
        'intent': intent,
        'suggestion': suggestion,
    }


# ==============================================================================
# MAIN API
# ==============================================================================

def check_info_gaps(text: str) -> dict:
    """
    Hybrid Info Gap Detection pipeline.

    Multi-sentence support: analyzes each sentence independently and
    returns all detected gaps across the entire input.

    Returns: {"gaps": [{"sentence": ..., "missing": ..., ...}, ...]}
    Compatible with existing frontend (uses g.missing and g.sentence).
    """
    if not text or not text.strip():
        return {"gaps": []}

    # Split into sentences for multi-sentence analysis
    sentences = get_sentences(text)

    # If only one sentence (or short text), analyze as whole
    if len(sentences) <= 1:
        sentences = [text.strip()]

    all_gaps = []
    seen = set()  # Deduplicate by (sentence, missing) pair

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        gap = _detect_gaps_for_sentence(sent)
        if gap:
            # Rule #4: preserve all gaps, dedup by content not just sentence
            key = (sent, gap['missing'])
            if key not in seen:
                seen.add(key)
                all_gaps.append(gap)

    return {"gaps": all_gaps}
