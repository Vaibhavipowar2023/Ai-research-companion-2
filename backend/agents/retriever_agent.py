import os
import json
import time
from utils.api_utils import fetch_arxiv  # keep only arXiv
from utils.nlp_utils import rank_papers_by_query
from config import ARXIV_MAX_RESULTS

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_get(key):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _cache_set(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def retrieve_papers(query: str, top_k: int = 4):
    """
    Lightweight paper retriever (arXiv only, cached).
    Optimized for low memory environments like Render free tier.
    """
    start = time.time()
    key = query.replace(" ", "_").lower()

    cached = _cache_get(key)
    if cached:
        print(f"[Retriever] Cache hit for '{query}'")
        return cached[:top_k]

    try:
        arx = fetch_arxiv(query, max_results=min(ARXIV_MAX_RESULTS, 5))
    except Exception as e:
        print(f"[Retriever] arXiv fetch failed: {e}")
        arx = []

    if not arx:
        print(f"[Retriever] No results for '{query}'")
        return []

    ranked = rank_papers_by_query(query, arx, top_k=top_k)

    simple = []
    for p in ranked:
        simple.append({
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "link": p.get("link", "") or p.get("url", ""),
            "url": p.get("link", "") or p.get("url", ""),
            "authors": p.get("authors", []),
            "source": p.get("source", ""),
            "score": round(p.get("score", 0.0), 4)
        })

    _cache_set(key, simple)
    print(f"[Retriever] Done '{query}' in {time.time() - start:.2f}s, {len(simple)} papers")
    return simple
