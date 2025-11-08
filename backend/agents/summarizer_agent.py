from models.bert_summarizer import extractive_summary

def summarize_one(paper):
    """Fast, lightweight summarization (extractive only)."""
    abstract = paper.get("abstract", "") or ""
    extractive = extractive_summary(abstract, sentences=2) if abstract else ""

    return {
        "title": paper.get("title", ""),
        "extractive": extractive,
        "abstractive": extractive,  # skip OpenAI for free-tier stability
        "link": paper.get("link", ""),
        "source": paper.get("source", ""),
        "authors": paper.get("authors", []),
    }


def summarize_papers(papers: list):
    """
    Summarize top 3 papers only.
    Purely extractive to avoid OpenAI latency & memory spikes.
    """
    if not papers:
        return []
    papers = papers[:3]  # limit
    return [summarize_one(p) for p in papers]
