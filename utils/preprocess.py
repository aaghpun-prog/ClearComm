import spacy
import nltk
import os
from nltk.tokenize import sent_tokenize, word_tokenize

# Set local NLTK data path
NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

# Setup spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy en_core_web_sm model not found. Please ensure it is installed.")
    nlp = None

def get_sentences(text: str) -> list[str]:
    """Splits text into sentences using NLTK."""
    return sent_tokenize(text)

def get_words(text: str) -> list[str]:
    """Splits text into words using NLTK."""
    return word_tokenize(text)

def get_spacy_doc(text: str):
    """Returns a spaCy Doc object for the text."""
    if nlp:
        return nlp(text)
    return None
