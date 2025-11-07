import os
import json
import time
import concurrent.futures
from utils.api_utils import fetch_arxiv, fetch_pubmed
from utils.nlp_utils import rank_papers_by_query
from config import ARXIV_MAX_RESULTS, PUBMED_MAX_RESULTS

# Local cache to avoid re-fetching same topic
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


def _fetch_arxiv_safe(query):
    try:
        return fetch_arxiv(query, max_results=min(ARXIV_MAX_RESULTS, 10))
    except Exception as e:
        print(f"[Retriever] arXiv fetch failed: {e}")
        return []


def _fetch_pubmed_safe(query):
    try:
        return fetch_pubmed(query, retmax=min(PUBMED_MAX_RESULTS, 10))
    except Exception as e:
        print(f"[Retriever] PubMed fetch failed: {e}")
        return []


def retrieve_papers(query: str, top_k: int = 4):
    """
    Fast paper retriever with parallel fetching and caching.
    Fetches from arXiv + PubMed concurrently, then ranks results.
    """
    start = time.time()
    key = query.replace(" ", "_").lower()

    cached = _cache_get(key)
    if cached:
        print(f"[Retriever] Cache hit for '{query}'")
        return cached[:top_k]

    # Fetch arXiv + PubMed concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_fetch_arxiv_safe, query)
        f2 = pool.submit(_fetch_pubmed_safe, query)
        arx = f1.result(timeout=30)
        pub = f2.result(timeout=30)

    combined = (arx or []) + (pub or [])
    if not combined:
        print(f"[Retriever] No results for '{query}'")
        return []

    ranked = rank_papers_by_query(query, combined, top_k=top_k)

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
