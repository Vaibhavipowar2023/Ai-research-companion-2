# utils/api_utils.py (arXiv + PubMed fetchers)
import requests
import xmltodict

ARXIV_BASE = "http://export.arxiv.org/api/query"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def fetch_arxiv(query, max_results=5):
    url = f"{ARXIV_BASE}?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = xmltodict.parse(r.text)
    entries = data.get("feed", {}).get("entry", []) or []
    if isinstance(entries, dict):
        entries = [entries]
    papers = []
    for e in entries:
        authors = e.get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        papers.append({
            "title": (e.get("title","") or "").strip(),
            "abstract": (e.get("summary","") or "").strip(),
            "link": e.get("id",""),
            "source": "arXiv",
            "authors": [a.get("name","") for a in authors if isinstance(a, dict)]
        })
    return papers

def fetch_pubmed(query, retmax=5):
    params = {"db":"pubmed","term":query,"retmax":retmax,"retmode":"json"}
    r = requests.get(PUBMED_ESEARCH, params=params, timeout=20)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", []) or []
    papers = []
    for pid in ids:
        try:
            res = requests.get(PUBMED_EFETCH, params={"db":"pubmed","id":pid,"retmode":"xml"}, timeout=20)
            doc = xmltodict.parse(res.text)
            art = doc.get("PubmedArticleSet", {}).get("PubmedArticle", {})
            article = art.get("MedlineCitation", {}).get("Article", {})
            if not article:
                continue
            abstract = ""
            if "Abstract" in article and article["Abstract"]:
                abs_text = article["Abstract"].get("AbstractText")
                if isinstance(abs_text, list):
                    abstract = " ".join(abs_text)
                else:
                    abstract = abs_text or ""
            papers.append({
                "title": article.get("ArticleTitle",""),
                "abstract": abstract,
                "link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                "source": "PubMed",
                "authors": []
            })
        except Exception:
            continue
    return papers
