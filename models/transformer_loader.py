import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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
            cls._instance._summarizer_model = None
            cls._instance._summarizer_tokenizer = None
            cls._instance._zero_shot = None
            cls._instance._sbert = None
            cls._instance._torch_device = None
            try:
                cls._instance._nlp = spacy.load("en_core_web_sm")
            except Exception:
                print("Warning: spaCy en_core_web_sm model not found. Proceeding with fallback mode for keyword extraction.")
                cls._instance._nlp = None
        return cls._instance

    # -----------------------------------------------------
    # DEVICE HANDLING (Apple GPU Support)
    # -----------------------------------------------------
    @property
    def device(self):
        if torch.backends.mps.is_available():
            print("Using Apple GPU (MPS)")
            return torch.device("mps")
        if torch.cuda.is_available():
            print("Using CUDA GPU")
            return torch.device("cuda")
        print("Using CPU")
        return torch.device("cpu")

    @property
    def torch_device(self):
        if self._torch_device is None:
            self._torch_device = self.device
        return self._torch_device

    # Pipeline device helper: transformers pipeline expects int or str
    @property
    def pipeline_device(self):
        d = self.torch_device
        if d.type == "cpu":
            return -1
        return d.type

    # -----------------------------------------------------
    # FLAN-T5 SEQ2SEQ MODEL (direct loading, no pipeline)
    # -----------------------------------------------------
    def _load_summarizer(self):
        if self._summarizer_model is None:
            print("Loading Flan-T5 (Instruction-Tuned) for Rewriting...")
            self._summarizer_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self._summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            self._summarizer_model.to(self.torch_device)
            print("Flan-T5 loaded.")

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
                device=self.pipeline_device
            )
            print("Zero-shot model loaded.")
        return self._zero_shot

    # -----------------------------------------------------
    # SBERT HOMONYM MODEL
    # -----------------------------------------------------
    @property
    def sbert(self):
        if self._sbert is None:
            print("Loading SBERT...")
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer('all-MiniLM-L6-v2')
            print("SBERT loaded.")
        return self._sbert

    # -----------------------------------------------------
    # KEYWORD EXTRACTION (LOCK IMPORTANT WORDS)
    # -----------------------------------------------------
    def extract_keywords(self, text: str) -> List[str]:
        if not self._nlp:
            return [] # Fail gracefully if spaCy is missing
            
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
        if not text or not text.strip():
            return text

        keywords = self.extract_keywords(text)

        # Flan-T5 token math: ~1.3 tokens per word.
        target_tokens = int(target_length * 1.3)
        
        max_len = max(10, target_tokens + 20)
        min_len = max(5, target_tokens - 10)

        # Ensure min_len < max_len
        if min_len >= max_len:
            min_len = max(5, max_len - 5)

        # Instruction-tuned prompt
        keyword_str = ', '.join(keywords) if keywords else 'key concepts'
        prompt = (
            f"Rewrite the following text to be exactly {target_length} words long. "
            f"Preserve the following important keywords: {keyword_str}.\n\n"
            f"Text: {text}"
        )

        try:
            self._load_summarizer()
            tokenizer = self._summarizer_tokenizer
            model = self._summarizer_model

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.torch_device)

            outputs = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                early_stopping=True
            )

            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
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