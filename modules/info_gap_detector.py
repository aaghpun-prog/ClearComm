"""
info_gap_detector.py — Smart Hybrid Information Gap Detection System

Architecture:
  Layer 1: Multi-clause Sentence Splitting (spaCy conjunction logic)
  Layer 2: General Action Meaning Extraction (ROOT verb & dependency logic)
  Layer 3: WordNet Generalization (synonym mapping for unseen verbs)
  Layer 4: Deep Object/Participant Extraction (spaCy dobj, nsubj, NER, Regex)
  Layer 5: Generalized Semantic Rules (natural slot requirements per action type)

Supports true generalized reasoning.
"""

import re
import nltk
from nltk.corpus import wordnet as wn
from utils.preprocess import get_spacy_doc, get_sentences

# Ensure wordnet is available
try:
    wn.synsets('run')
except LookupError:
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

# ==============================================================================
# ENTITY DETECTION PATTERNS
# ==============================================================================

RE_TIME = re.compile(
    r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b'
    r'|\b(?:noon|midnight|morning|afternoon|evening|now)\b',
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
    'there', 'here', 'inside', 'outside', 'home'
})

ONLINE_MODE_KEYWORDS = [
    'google meet', 'video call', 'in-person', 'on-site',
    'online', 'zoom', 'teams', 'webex', 'skype',
    'virtual', 'remote', 'offline', 'hybrid',
]

CASUAL_KEYWORDS = frozenset({
    'hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon',
    'good night', 'thanks', 'thank you', 'bye', 'goodbye', 'see you',
    'nice weather', 'how are you', 'what\'s up', 'take care', 'well done',
    'congratulations', 'happy birthday', 'cheers'
})

# ==============================================================================
# GENERAL ACTION SEMANTICS
# ==============================================================================

FAMILY_VERBS = {
    'Travel': frozenset({'go', 'come', 'arrive', 'reach', 'visit', 'travel', 'head', 'proceed', 'enter', 'return', 'leave', 'move', 'depart'}),
    'Meeting': frozenset({'meet', 'join', 'attend', 'gather', 'assemble', 'report', 'connect', 'see', 'convene'}),
    'Task': frozenset({'submit', 'upload', 'send', 'finish', 'complete', 'prepare', 'review', 'fix', 'deploy', 'update', 'deliver', 'dispatch', 'assign', 'book', 'reserve'}),
    'Payment': frozenset({'pay', 'transfer', 'buy', 'purchase', 'reimburse', 'deposit', 'remit', 'spend', 'charge'}),
    'Communication': frozenset({'call', 'inform', 'notify', 'email', 'message', 'contact', 'tell', 'ask', 'reply', 'ping'})
}

# Pre-fetch synsets for WordNet generalization mapping
FAMILY_REPS = {}
try:
    FAMILY_REPS = {
        'Travel': [wn.synset('travel.v.01'), wn.synset('move.v.02')],
        'Meeting': [wn.synset('meet.v.01'), wn.synset('assemble.v.01')],
        'Task': [wn.synset('submit.v.01'), wn.synset('deliver.v.01'), wn.synset('perform.v.01')],
        'Payment': [wn.synset('pay.v.01'), wn.synset('buy.v.01')],
        'Communication': [wn.synset('communicate.v.01'), wn.synset('inform.v.01')]
    }
except Exception:
    pass

def _split_clauses(text: str) -> list:
    """Intelligently split multi-action sentences using spaCy conjunction logic."""
    doc = get_spacy_doc(text)
    if not doc:
        return [text]
        
    clauses = []
    current_clause = []
    
    for token in doc:
        if token.lower_ in ('and', 'but', 'then'):
            # If there's another verb closely following, it's likely a distinct action clause
            has_verb_after = any(t.pos_ == 'VERB' for t in doc[token.i + 1 : token.i + 5])
            if has_verb_after:
                if current_clause:
                    clauses.append(''.join(current_clause).strip())
                    current_clause = []
                continue
        current_clause.append(token.text_with_ws)
        
    if current_clause:
        clauses.append(''.join(current_clause).strip())
        
    # Filter out empty or trivially small non-action parts
    valid_clauses = [c for c in clauses if len(c.split()) > 1]
    return valid_clauses if valid_clauses else [text]

def _extract_main_action(doc) -> str:
    """Extract the main action verb of the sentence using dependency parsing."""
    if not doc:
        return None
        
    # 1. Imperative (starts with verb)
    if len(doc) > 0 and doc[0].pos_ == 'VERB':
        return doc[0].lemma_.lower()
        
    # 2. Root verb
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            return token.lemma_.lower()
            
    # 3. Any verb fallback
    for token in doc:
        if token.pos_ == 'VERB':
            return token.lemma_.lower()
            
    return None

def _get_action_family(verb: str) -> tuple:
    """Determine the semantic family of a given verb, returning (family, matching_method)."""
    if not verb:
        return None, None
        
    # Direct fast mapping
    for family, verbs in FAMILY_VERBS.items():
        if verb in verbs:
            return family, 'direct'
            
    # Generalization using WordNet similarity for unseen verbs
    if FAMILY_REPS:
        try:
            verb_synsets = wn.synsets(verb, pos=wn.VERB)
            if verb_synsets:
                best_family = None
                max_sim = 0.45  # Semantic similarity threshold
                
                for v_syn in verb_synsets:
                    for family, reps in FAMILY_REPS.items():
                        for rep_syn in reps:
                            sim = v_syn.path_similarity(rep_syn)
                            if sim and sim > max_sim:
                                max_sim = sim
                                best_family = family
                                
                if best_family:
                    return best_family, 'wordnet'
        except Exception:
            pass
            
    return None, None


# ==============================================================================
# INTENT KEYWORDS & REQUIRED FIELDS
# ==============================================================================

INTENT_KEYWORDS = {
    "Event": {'keywords': ['event', 'fest', 'festival', 'conference', 'ceremony', 'gathering'], 'weight': 2},
    "Workshop": {'keywords': ['workshop', 'seminar', 'webinar', 'tutorial', 'training', 'masterclass'], 'weight': 2},
    "Interview": {'keywords': ['interview', 'hiring', 'recruitment', 'job', 'applicant'], 'weight': 2},
    "Emergency": {'keywords': ['urgent', 'emergency', 'critical', 'asap', 'immediately', 'priority'], 'weight': 2},
    "Task": {'keywords': ['need to', 'have to', 'must', 'should', 'please', 'ensure', 'verify'], 'weight': 1},
}

INTENT_REQUIRED_FIELDS = {
    # Core Semantic Action Families
    "Travel": {
        'fields': ['location', 'time'],
        'labels': {'location': 'Destination/Location', 'time': 'Time'},
    },
    "Meeting": {
        'fields': ['participant', 'location', 'time'],
        'labels': {'participant': 'Person/Group', 'location': 'Location/Mode', 'time': 'Time'},
    },
    "Task": {
        'fields': ['object', 'assignee', 'deadline'],
        'labels': {'object': 'Task Object', 'assignee': 'Assignee', 'deadline': 'Deadline'},
    },
    "Payment": {
        'fields': ['amount', 'recipient', 'deadline'],
        'labels': {'amount': 'Amount', 'recipient': 'Recipient', 'deadline': 'Date/Time'},
    },
    "Communication": {
        'fields': ['recipient', 'object'],
        'labels': {'recipient': 'Recipient', 'object': 'Content/Topic'},
    },
    
    # Specific Domain Fallbacks
    "Event": {
        'fields': ['date', 'time', 'location'],
        'labels': {'date': 'Date', 'time': 'Time', 'location': 'Venue'},
    },
    "Interview": {
        'fields': ['date', 'time', 'location'],
        'labels': {'date': 'Date', 'time': 'Time', 'location': 'Venue/Mode'},
    },
    "Workshop": {
        'fields': ['date', 'time', 'location'],
        'labels': {'date': 'Date', 'time': 'Time', 'location': 'Venue'},
    },
    "Emergency": {
        'fields': ['action', 'contact'],
        'labels': {'action': 'Required Action', 'contact': 'Contact Person'},
    },
}

# ==============================================================================
# ENTITY EXTRACTION
# ==============================================================================

def _extract_entities(text: str) -> dict:
    """
    Extract all information entities using spaCy NER, dependencies, regex, and keywords.
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
        'object': False,
    }

    has_object_dep = False

    if doc:
        for token in doc:
            # NER Mapping
            if token.ent_type_ == 'TIME': entities['time'] = True
            elif token.ent_type_ == 'DATE': entities['date'] = True
            elif token.ent_type_ in ('GPE', 'LOC', 'FAC'): entities['location'] = True
            elif token.ent_type_ == 'MONEY': entities['money'] = True
            elif token.ent_type_ == 'PERSON': entities['person'] = True
            elif token.ent_type_ == 'ORG': entities['organization'] = True
            
            # Semantic object mapping (dobj, pobj)
            if token.dep_ in ('dobj', 'pobj', 'attr', 'nsubjpass'):
                if token.lemma_.lower() not in ('it', 'this', 'that', 'there', 'here', 'them'):
                    has_object_dep = True
                    
            # Pronoun subject mapping (satisfies assignee implicitly)
            if token.dep_ == 'nsubj' and token.pos_ == 'PRON':
                entities['person'] = True

    # Finalize Object status
    entities['object'] = has_object_dep or entities['link']

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
# CLASSIFICATION LOGIC
# ==============================================================================

def _is_casual_or_general(text: str) -> bool:
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    if len(words) <= 6:
        for cw in CASUAL_KEYWORDS:
            if cw in text_lower:
                return True
                
    doc = get_spacy_doc(text)
    if not doc:
        return False
        
    # Imperative is usually actionable, not general fact
    if doc[0].pos_ == 'VERB':
        return False
        
    has_action_verb = any(
        t.pos_ == 'VERB' and t.dep_ in ('ROOT',) and t.lemma_ not in (
            'be', 'have', 'do', 'seem', 'appear', 'look', 'feel', 'become'
        )
        for t in doc
    )
    has_copula = any(t.lemma_ == 'be' and t.dep_ == 'ROOT' for t in doc)
    
    if has_copula and not has_action_verb:
        return True
        
    return False

def _classify_intent(text: str, doc) -> tuple:
    """
    Classify message intent using General Action Meaning + Keyword fallback.
    Returns (intent, confidence, main_verb)
    """
    text_lower = text.lower()
    scores = {}

    # 1. General Action Meaning Engine
    main_verb = _extract_main_action(doc)
    family, method = _get_action_family(main_verb)
    
    if family:
        # Action semantics take massive priority
        scores[family] = 10
        confidence = 'High' if method == 'direct' else 'Medium'
        return family, confidence, main_verb

    # 2. Domain Keyword Fallbacks
    for intent, data in INTENT_KEYWORDS.items():
        score = 0
        for kw in data['keywords']:
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                score += data['weight']
        if score > 0:
            scores[intent] = scores.get(intent, 0) + score

    if not scores:
        return None, 'Low', main_verb

    best = max(scores, key=scores.get)
    best_score = scores[best]

    confidence = 'Medium' if best_score >= 2 else 'Low'
    return best, confidence, main_verb


# ==============================================================================
# GAP DETECTION LOGIC
# ==============================================================================

def _check_field_present(field: str, entities: dict, text_lower: str) -> bool:
    if field == 'date': return entities['date']
    if field == 'time': return entities['time']
    if field == 'location': return entities['location'] or entities['mode']
    if field == 'assignee': return entities['person'] or entities['organization']
    if field == 'deadline': return entities['date'] or entities['time']
    if field == 'amount': return entities['money']
    if field == 'recipient': return entities['person'] or entities['organization'] or entities['contact']
    if field == 'participant': return entities['person'] or entities['organization'] or entities['contact']
    if field == 'object': return entities['object']
    if field == 'action':
        doc = get_spacy_doc(text_lower)
        if doc:
            return any(t.pos_ == 'VERB' and t.dep_ == 'ROOT'
                      and t.lemma_ not in ('be', 'have') for t in doc)
    return False

def _detect_gaps_for_sentence(sentence: str) -> dict | None:
    text = sentence.strip()
    if not text:
        return None

    if _is_casual_or_general(text):
        return None

    text_lower = text.lower()
    doc = get_spacy_doc(text)

    entities = _extract_entities(text)
    intent_result = _classify_intent(text, doc)
    
    if not intent_result or not intent_result[0]:
        return None
        
    intent, confidence, main_verb = intent_result

    req = INTENT_REQUIRED_FIELDS.get(intent)
    if not req:
        return None

    missing_labels = []
    for field in req['fields']:
        if not _check_field_present(field, entities, text_lower):
            label = req['labels'][field]
            
            # Contextual label adjustment: exact time vs general time
            if intent in ('Meeting', 'Travel') and field == 'time' and entities['date']:
                label = 'exact Time'
                
            # Contextual label adjustment: delivery verbs target recipient instead of assignee
            if intent == 'Task' and label == 'Assignee' and main_verb in ('send', 'dispatch', 'deliver', 'forward', 'mail', 'email'):
                label = 'Recipient'
                
            missing_labels.append(label)

    if not missing_labels:
        return None

    missing_str = ', '.join(missing_labels)

    suggestion_parts = [m.lower() for m in missing_labels]
    if len(suggestion_parts) > 1:
        suggestion = f"Please include {', '.join(suggestion_parts[:-1])} and {suggestion_parts[-1]}."
    else:
        suggestion = f"Please include {suggestion_parts[0]}."

    reason = None
    if confidence == 'High':
        act_name = intent
        if intent == 'Travel': act_name = 'Movement'
        
        pres_mapped = []
        for f in req['fields']:
            if _check_field_present(f, entities, text_lower):
                lbl = req['labels'][f].lower().split('/')[0]
                if lbl == 'person': lbl = 'participant'
                elif lbl == 'task object': lbl = 'object'
                elif lbl == 'date time': lbl = 'date'
                pres_mapped.append(lbl)
                
        # Explicit check for Date if it's not a formal requirement but present
        if 'time' in req['fields'] and entities['date'] and not entities['time']:
            pres_mapped.append('date')
            
        miss_mapped = []
        for m in missing_labels:
            m_lower = m.lower().split('/')[0]
            if m_lower == 'recipient' and intent == 'Payment':
                miss_mapped.append('payee')
            elif m_lower == 'assignee':
                miss_mapped.append('responsible person')
            elif m_lower == 'deadline':
                miss_mapped.append('due time')
            elif m_lower == 'location':
                if intent == 'Travel':
                    miss_mapped.append('destination')
                else:
                    miss_mapped.append('place')
            else:
                miss_mapped.append(m_lower)
                
        if pres_mapped:
            if len(pres_mapped) == 1:
                pres_str = f"with {pres_mapped[0]} present"
            else:
                pres_str = f"with {pres_mapped[0]} and {pres_mapped[1]} present"
        else:
            pres_str = "without key details"
            
        if len(miss_mapped) == 1:
            miss_str = f"no {miss_mapped[0]} found"
        else:
            if intent == 'Payment':
                miss_str = f"{miss_mapped[0]} and {miss_mapped[1]} missing"
            else:
                miss_str = f"no {miss_mapped[0]} or {miss_mapped[1]} found"
            
        reason = f"{act_name} action detected {pres_str}, but {miss_str}."

    res = {
        'sentence': text,
        'missing': missing_str,
        'confidence': confidence,
        'intent': intent,
        'suggestion': suggestion,
    }
    
    if reason:
        res['reason'] = reason
        
    return res


# ==============================================================================
# MAIN API
# ==============================================================================

def check_info_gaps(text: str) -> dict:
    """
    Hybrid Info Gap Detection pipeline with Deep Semantic Action Reasoning.
    """
    if not text or not text.strip():
        return {"gaps": []}

    sentences = get_sentences(text)
    if len(sentences) <= 1:
        sentences = [text.strip()]

    all_gaps = []
    seen = set()

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Split multi-action sentences into individual logical clauses
        clauses = _split_clauses(sent)
        
        for clause in clauses:
            gap = _detect_gaps_for_sentence(clause)
            if gap:
                key = (clause, gap['missing'])
                if key not in seen:
                    seen.add(key)
                    all_gaps.append(gap)

    return {"gaps": all_gaps}
