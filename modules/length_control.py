from models.transformer_loader import get_models
from utils.preprocess import get_words

def analyze_length_and_rewrite(text: str, target_word_count: int) -> dict:
    """Uses a local summarization model to rewrite the text to the target word count."""
    
    # 1. Load the models
    models = get_models()
    
    # 2. Generate rewritten text
    rewritten_text = models.rewrite_text(text, target_word_count)
    
    # 3. Calculate final word count
    final_words = get_words(rewritten_text)
    final_count = len(final_words)
    
    return {
        "rewritten_text": rewritten_text,
        "final_word_count": final_count,
        "status": "success"
    }
