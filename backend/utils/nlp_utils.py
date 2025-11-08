# utils/nlp_utils.py (ranking)
from sentence_transformers import SentenceTransformer, util
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def _embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")


def rank_papers_by_query(query: str, papers: list, top_k: int = 4) -> list:
    texts = [(p.get("abstract","") or "") for p in papers]
    if not any(texts):
        return papers[:top_k]
    model = _embedder()
    q_emb = model.encode(query, convert_to_tensor=True)
    abs_embs = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, abs_embs)[0].cpu().numpy()
    idx = np.argsort(-scores)[:top_k]
    ranked = []
    for i in idx:
        p = papers[i].copy()
        p["score"] = float(scores[i])
        ranked.append(p)
    return ranked
