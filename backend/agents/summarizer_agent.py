import asyncio
import concurrent.futures
from models.bert_summarizer import extractive_summary
from models.get_summarizer import generate_abstractive


def summarize_one(paper):
    """Summarize a single paper (blocking)"""
    abstract = paper.get("abstract", "") or ""
    extractive = extractive_summary(abstract, sentences=2) if abstract else ""
    abstractive = extractive
    if abstract:
        try:
            prompt = (
                f"Title: {paper.get('title','')}\n\n"
                f"Extractive summary:\n{extractive}\n\n"
                "Provide a concise abstractive summary (2-3 sentences)."
            )
            abstractive = generate_abstractive(prompt, max_tokens=180)
        except Exception:
            abstractive = extractive

    return {
        "title": paper.get("title", ""),
        "extractive": extractive,
        "abstractive": abstractive,
        "link": paper.get("link", ""),
        "source": paper.get("source", ""),
        "authors": paper.get("authors", []),
    }


def summarize_papers(papers: list):
    """
    Parallelized summarization for multiple papers.
    Limits to top 2–3 papers for speed. Uses a ThreadPoolExecutor
    since generate_abstractive() is blocking.
    """
    if not papers:
        return []

    # Limit for performance on free Render
    papers = papers[:3]

    out = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(summarize_one, papers))
            out.extend(results)
    except Exception as e:
        print(f"[SUMMARIZER ERROR] {e}")
        # Fallback to sequential loop if threadpool fails
        for p in papers:
            out.append(summarize_one(p))
    return out
