# ClearComm 🎙️📝

**ClearComm** is an advanced AI-powered text clarity analysis and enrichment system. Designed to bridge the gap between complex information and clear communication, the system leverages state-of-the-art NLP models to detect ambiguity, control verbosity, and identify missing context in written content.

---

## 🌟 Key Features

ClearComm is built around three core NLP modules:

### 1. 🔍 Homonym Disambiguation (Context-Aware)
It dynamically detects ambiguous words (homonyms) and determines their exact meaning based on sentence context using **SBERT** and robust fallback linguistic filtering (NLTK/spaCy).
* **Dual-Threshold Architecture:** Balances high precision to prevent false positives with guaranteed recall for core vocabulary.
* **WiC Integration:** Includes an extensible pipeline designed for the *Word-in-Context (WiC)* DistilBERT model.

### 2. 📏 Length Control & Generative Rewriting
Allows users to specify an exact target length for a sentence without losing vital contextual information.
* Extracts critical keywords using **spaCy** Named Entity Recognition (NER) & POS tagging.
* Leverages instruction-tuned **Flan-T5** to intelligently rewrite text to the target word count while strictly preserving the extracted keywords.

### 3. 🧩 Information Gap Detector (Zero-Shot)
Analyzes text against user-defined communication parameters to ensure completeness in messaging.
* Uses **DistilBART (Zero-Shot Classification)** to calculate confidence scores on whether required topics (e.g., "Time", "Location", "Price") are adequately covered in the passage.

---

## 🛠️ Architecture & Tech Stack

ClearComm employs a modular ML architecture, caching weights locally and utilizing Singleton design patterns to minimize RAM overhead during inference.

* **Backend Framework:** Flask (Python)
* **NLP & Tokenization:** spaCy (`en_core_web_sm`), NLTK (WordNet API, Averaged Perceptron Tagger)
* **Machine Learning & Transformers:** 
  * `sentence-transformers` (all-MiniLM-L6-v2)
  * Hugging Face `transformers` (Flan-T5-base, DistilBART-MNLI-12-1)
  * PyTorch

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.9+
* Git

### Step-by-Step Guide
1. **Clone the repository**
   ```bash
   git clone https://github.com/aaghpun-prog/ClearComm.git
   cd ClearComm
   ```

2. **Set up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy English Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```
   *Note: NLTK datasets (WordNet, Punkt, etc.) are automatically mapped to a local `nltk_data` directory upon initialization to cleanly quarantine ML logic inside the workspace.*

5. **Run the Application**
   ```bash
   python app.py
   ```
   The Flask application will initialize. Open your web browser and navigate to `http://127.0.0.1:5001`.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to document bugs or suggest new features.

---
*Developed as a Software Engineering College Project.*
