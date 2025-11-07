# agents/retriever_agent.py
from utils.api_utils import fetch_arxiv, fetch_pubmed
from utils.nlp_utils import rank_papers_by_query
from config import ARXIV_MAX_RESULTS, PUBMED_MAX_RESULTS

def retrieve_papers(query: str, top_k: int = 4):
    try:
        arx = fetch_arxiv(query, max_results=ARXIV_MAX_RESULTS)
    except Exception:
        arx = []
    try:
        pub = fetch_pubmed(query, retmax=PUBMED_MAX_RESULTS)
    except Exception:
        pub = []
    combined = arx + pub
    if not combined:
        return []
    ranked = rank_papers_by_query(query, combined, top_k=top_k)
    # minimal fields only
    simple = []
    for p in ranked:
        simple.append({
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "link": p.get("link", "") or p.get("url", ""),
            "url": p.get("link", "") or p.get("url", ""),
            "authors": p.get("authors", []),
            "source": p.get("source", ""),
            "score": p.get("score", 0.0)
        })
    return simple
