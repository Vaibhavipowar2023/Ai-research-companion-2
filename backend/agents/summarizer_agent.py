# agents/summarizer_agent.py
from models.bert_summarizer import extractive_summary
from models.get_summarizer import generate_abstractive

def summarize_papers(papers: list):
    out = []
    for p in papers:
        abstract = p.get("abstract", "") or ""
        extractive = extractive_summary(abstract, sentences=2) if abstract else ""
        abstractive = extractive
        if abstract:
            try:
                prompt = f"Title: {p.get('title','')}\n\nExtractive summary:\n{extractive}\n\nProvide a concise abstractive summary (2-3 sentences)."
                abstractive = generate_abstractive(prompt, max_tokens=180)
            except Exception:
                abstractive = extractive
        out.append({
            "title": p.get("title",""),
            "extractive": extractive,
            "abstractive": abstractive,
            "link": p.get("link",""),
            "source": p.get("source",""),
            "authors": p.get("authors",[])
        })
    return out
