import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class WiCModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WiCModelLoader, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = None
        return cls._instance

    def load_model(self, model_dir="saved_models/wic_model"):
        """
        Loads the trained DistilBERT WiC model from the specified directory.
        Uses a Singleton pattern to avoid re-loading the model in memory.
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to base model if fine-tuned doesn't exist yet
            if not os.path.exists(model_dir):
                print(f"Warning: Fine-tuned model not found at {model_dir}. Loading base model to prevent crash. Please run the training script.")
                model_name_or_path = "distilbert-base-uncased"
            else:
                model_name_or_path = model_dir
                
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
            self.model.to(self.device)
            self.model.eval()
            
        return self.tokenizer, self.model, self.device

def predict_meaning_wic(user_sentence, word, candidate_meanings_with_examples):
    """
    Predicts the best meaning for an ambiguous word using the fine-tuned WiC model.
    Now returns top_score and score_gap for precision filtering.
    """
    loader = WiCModelLoader()
    tokenizer, model, device = loader.load_model()
    
    scores = []
    
    # We will compute the softmax probabilities for all combinations
    for candidate in candidate_meanings_with_examples:
        context2 = candidate.get('example', '')
        if not context2:
            context2 = candidate.get('meaning', '')
            
        inputs = tokenizer(
            user_sentence,
            context2,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_same_meaning = probs[0][1].item()
            
        scores.append((prob_same_meaning, candidate))
            
    if not scores:
        return None

    # Sort scores descending
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_candidate = scores[0]
    
    # Calculate gap
    if len(scores) > 1:
        score_gap = best_score - scores[1][0]
    else:
        score_gap = 1.0 # Only one candidate, high confidence in the choice itself relative to others
        
    return {
        "word": word,
        "meaning": best_candidate["meaning"],
        "confidence": "wic_model",
        "score": best_score,
        "score_gap": score_gap
    }
