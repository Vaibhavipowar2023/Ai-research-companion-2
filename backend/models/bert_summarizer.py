# models/bert_summarizer.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# ensure punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def extractive_summary(text: str, sentences: int = 2) -> str:
    if not text:
        return ""
    sents = nltk.tokenize.sent_tokenize(text)
    if len(sents) <= sentences:
        return " ".join(sents)
    model = _load_model()
    doc_emb = model.encode([text])[0]
    sent_embs = model.encode(sents)
    scores = cosine_similarity([doc_emb], sent_embs)[0]
    idx = np.argsort(-scores)[:sentences]
    selected = [sents[i] for i in sorted(idx)]
    return " ".join(selected)
