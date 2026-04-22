import re
from utils.preprocess import get_spacy_doc
from models.transformer_loader import get_models

# ==============================================================================
# LAYER 1: Rule-Based Logic (Fast Execution)
# ==============================================================================

# Fast RegEx Patterns
RE_TIME = re.compile(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b', re.IGNORECASE)
RE_EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
RE_PHONE = re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')
RE_URL = re.compile(r'https?://\S+|www\.\S+')

# Intent Categories and Keyword Sets
INTENT_KEYWORDS = {
    "Meeting": ["meeting", "sync", "catch up", "discuss", "call", "meetup", "meet"],
    "Event": ["event", "party", "workshop", "webinar", "conference", "session", "gathering"],
    "Sale": ["sale", "selling", "buy", "discount", "price", "offer", "available", "product"],
    "Job": ["interview", "hiring", "job", "position", "role", "vacancy"],
    "Task": ["please", "need to", "must", "submit", "update", "task", "required"]
}

# Casual phrases to completely ignore
CASUAL_KEYWORDS = [
    "hello", "hi", "good morning", "good evening", "good afternoon", 
    "thanks", "thank you", "bye", "goodbye", "nice weather", "how are you"
]

def _is_casual(text: str) -> bool:
    """Filter out non-actionable pleasantries."""
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) <= 5:
        for cw in CASUAL_KEYWORDS:
            if cw in text_lower:
                return True
    return False

def _extract_entities(doc, text: str) -> dict:
    """Extract required information entities using spaCy + RegEx."""
    text_lower = text.lower()
    
    entities = {
        "time": bool(RE_TIME.search(text)),
        "date": False,
        "location": False,
        "contact": bool(RE_EMAIL.search(text) or RE_PHONE.search(text)),
        "price": False,
        "person": False,
        "link": bool(RE_URL.search(text)),
        "mode": any(m in text_lower for m in ["online", "zoom", "teams", "meet", "offline", "in-person"])
    }

    if doc:
        for ent in doc.ents:
            if ent.label_ in ['TIME']:
                entities["time"] = True
            elif ent.label_ in ['DATE']:
                entities["date"] = True
            elif ent.label_ in ['GPE', 'LOC', 'FAC', 'ORG']:
                entities["location"] = True
            elif ent.label_ in ['MONEY']:
                entities["price"] = True
            elif ent.label_ in ['PERSON']:
                entities["person"] = True
                
    # Fallback keyword checks for misses
    if not entities["location"] and any(w in text_lower for w in ["room", "office", "venue", "hall"]):
        entities["location"] = True
    if not entities["price"] and any(w in text_lower for w in ["$", "£", "€", "dollars", "rupees", "cost", "price"]):
        entities["price"] = True
    if not entities["contact"] and "contact" in text_lower:
        entities["contact"] = True
    if not entities["date"] and any(w in text_lower for w in ["today", "tomorrow", "tonight", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
        entities["date"] = True
        
    return entities

def _detect_intent_layer1(text_lower: str) -> str:
    """Fast keyword-based intent classification."""
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf'\b{kw}\b', text_lower):
                scores[intent] += 1
                
    best_intent = max(scores, key=scores.get)
    if scores[best_intent] > 0:
        return best_intent
    return None

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def check_info_gaps(text: str) -> dict:
    """
    Hybrid Info Gap Detection pipeline.
    Layer 1: Rule-based fast execution.
    Layer 2: Zero-shot fallback for ambiguous sentences.
    Layer 3: Confidence filtering and response formatting.
    """
    if not text or not text.strip():
        return {"gaps": []}
        
    # Ignore casual chatter
    if _is_casual(text):
        return {"gaps": []}
        
    doc = get_spacy_doc(text)
    text_lower = text.lower()
    
    # Extract existing entities
    entities = _extract_entities(doc, text)
    
    # Layer 1 Classification
    intent = _detect_intent_layer1(text_lower)
    confidence = "High"
    
    # Layer 2 Fallback
    if not intent:
        has_action_verb = False
        if doc and len(doc) > 0:
            # Check if sentence starts with a verb (imperative/command)
            has_action_verb = doc[0].pos_ == 'VERB'
            
        if has_action_verb:
            intent = "Task"
            confidence = "Medium"
        else:
            # Zero-shot classification fallback
            models = get_models()
            categories = ["meeting announcement", "event announcement", "product sale", "job interview", "task request", "general statement"]
            try:
                result = models.classify_zero_shot(text, categories)
                top_label = result['labels'][0]
                if top_label == "meeting announcement": intent = "Meeting"
                elif top_label == "event announcement": intent = "Event"
                elif top_label == "product sale": intent = "Sale"
                elif top_label == "job interview": intent = "Job"
                elif top_label == "task request": intent = "Task"
                else: intent = None
                confidence = "Low"
            except Exception:
                intent = None
                
    # If it's a general statement or unclassifiable, return no gaps
    if not intent:
        return {"gaps": []}
        
    # Layer 3: Calculate Gaps & Formatting
    missing = []
    
    if intent == "Meeting":
        if not entities["time"]: missing.append("Time")
        if not entities["location"] and not entities["mode"]: missing.append("Location")
    elif intent == "Event":
        if not entities["date"]: missing.append("Date")
        if not entities["time"]: missing.append("Time")
        if not entities["location"] and not entities["mode"]: missing.append("Venue")
    elif intent == "Sale":
        if not entities["price"]: missing.append("Price")
        if not entities["contact"] and not entities["link"]: missing.append("Contact details")
    elif intent == "Job":
        if not entities["date"]: missing.append("Date")
        if not entities["time"]: missing.append("Time")
        if not entities["location"] and not entities["mode"]: missing.append("Venue")
    elif intent == "Task":
        if not entities["date"] and not entities["time"]: missing.append("Deadline")
        if not entities["person"]: missing.append("Assignee")
        
    if missing:
        missing_str = ", ".join(missing)
        
        # Build human-readable suggestion
        suggestion_parts = [m.lower().replace('details', 'information') for m in missing]
        if len(suggestion_parts) > 1:
            suggestion = f"Please include {', '.join(suggestion_parts[:-1])} and {suggestion_parts[-1]}."
        else:
            suggestion = f"Please include {suggestion_parts[0]}."
            
        return {
            "gaps": [
                {
                    "sentence": text,
                    "missing": missing_str,
                    "confidence": confidence,
                    "suggestion": suggestion
                }
            ]
        }
        
    return {"gaps": []}
