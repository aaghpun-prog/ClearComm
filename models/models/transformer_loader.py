import torch
from transformers import pipeline
from typing import List
import spacy


class TransformerModelsLoader:
    """
    Singleton-style loader for transformer models.
    Loads models only once to save memory and improve performance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._summarizer = None
            cls._instance._zero_shot = None
            cls._instance._nlp = spacy.load("en_core_web_sm")
        return cls._instance

    # -----------------------------------------------------
    # DEVICE HANDLING (Apple GPU Support)
    # -----------------------------------------------------
    @property
    def device(self):
        if torch.backends.mps.is_available():
            print("Using Apple GPU (MPS)")
            return "mps"
        print("Using CPU")
        return -1

    # -----------------------------------------------------
    # BART SUMMARIZER
    # -----------------------------------------------------
    @property
    def summarizer(self):
        if self._summarizer is None:
            print("Loading Flan-T5 (Instruction-Tuned) for Rewriting...")
            self._summarizer = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=self.device
            )
            print("Flan-T5 loaded.")
        return self._summarizer

    # -----------------------------------------------------
    # ZERO-SHOT CLASSIFIER (INFO GAP)
    # -----------------------------------------------------
    @property
    def zero_shot(self):
        if self._zero_shot is None:
            print("Loading Zero-Shot Classifier...")
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model="valhalla/distilbart-mnli-12-1",
                device=self.device
            )
            print("Zero-shot model loaded.")
        return self._zero_shot

    # -----------------------------------------------------
    # KEYWORD EXTRACTION (LOCK IMPORTANT WORDS)
    # -----------------------------------------------------
    def extract_keywords(self, text: str) -> List[str]:
        doc = self._nlp(text)
        keywords = []

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "NUM"]:
                keywords.append(token.text)

        return list(set(keywords))

    # -----------------------------------------------------
    # REPILOT STYLE LENGTH-CONTROLLED REWRITING
    def rewrite_text(self, text: str, target_length: int) -> str:
        """
        Rewrite text using Flan-T5 while controlling length
        and preserving important keywords.
        """
        keywords = self.extract_keywords(text)

        # Flan-T5 token math: ~1.3 tokens per word.
        target_tokens = int(target_length * 1.3)
        
        max_len = target_tokens + 20
        min_len = max(5, target_tokens - 10)

        # Instruction-tuned prompt
        prompt = (
            f"Rewrite the following text to be exactly {target_length} words long. "
            f"Preserve the following important keywords: {', '.join(keywords)}.\n\n"
            f"Text: {text}"
        )

        try:
            result = self.summarizer(
                prompt,
                max_length=max_len,
                min_length=min_len,
                do_sample=True,
                temperature=0.7,
                num_beams=4,
                early_stopping=True
            )

            output = result[0]["generated_text"]
            return output

        except Exception as e:
            print(f"Rewrite Error: {e}")
            return text

    # -----------------------------------------------------
    # ZERO SHOT CLASSIFICATION
    # -----------------------------------------------------
    def classify_zero_shot(self, text: str, candidate_labels: List[str]):
        return self.zero_shot(text, candidate_labels)


# ---------------------------------------------------------
# GLOBAL INSTANCE ACCESS
# ---------------------------------------------------------
def get_models():
    return TransformerModelsLoader()