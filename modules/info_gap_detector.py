from models.transformer_loader import get_models
from utils.preprocess import get_spacy_doc

def check_info_gaps(text: str) -> dict:
    """Hybrid approach to detect missing information in statements."""
    gaps = []
    
    # 1. Rule-Based Checks (using spaCy)
    doc = get_spacy_doc(text)
    
    has_person = False
    has_time = False
    has_action = False
    
    if doc:
        has_person = any(ent.label_ in ['PERSON', 'ORG'] for ent in doc.ents)
        has_time = any(ent.label_ in ['DATE', 'TIME'] for ent in doc.ents)
        has_action = any(token.pos_ == 'VERB' for token in doc)
        
        # Rule check: Subject mapping
        has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
        if not has_subject and not has_person:
            gaps.append({
                "sentence": text,
                "missing": "responsible person"
            })
            
    # 2. Transformer (Zero-Shot) Classification
    models = get_models()
    
    # If the sentence looks like an actionable request (contains a verb),
    # use the zero-shot classifier to see if it implies a deadline or condition.
    if has_action:
        categories = ["contains a specific deadline or time", "has a condition or dependency", "is a general statement"]
        result = models.classify_zero_shot(text, categories)
        
        # If the model thinks it's a general actionable statement but doesn't have a deadline:
        top_label = result['labels'][0]
        
        if top_label == "is a general statement" and not has_time:
            # We can flag it as potentially missing a deadline if it sounds like an action
            gaps.append({
                 "sentence": text,
                 "missing": "deadline"
             })
             
        # Optional: Condition logic
        if top_label == "is a general statement" and "if" not in text.lower() and "when" not in text.lower():
            pass # Skipping condition check as it's highly context dependent
            
    # Deduplicate
    unique_gaps = []
    seen_missing = set()
    for gap in gaps:
        if gap['missing'] not in seen_missing:
            unique_gaps.append(gap)
            seen_missing.add(gap['missing'])
            
    return {"gaps": unique_gaps}
