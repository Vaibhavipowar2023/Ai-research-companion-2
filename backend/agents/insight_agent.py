import os
import json
import time
import concurrent.futures
from models.get_summarizer import try_llm_completion

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_get(key):
    path = os.path.join(CACHE_DIR, f"{key}.insights.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _cache_set(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.insights.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _generate_insight_chunk(chunk_summaries, idx):
    """
    Generates insights for a small subset of summaries.
    Keeps prompt size manageable for speed and token cost.
    """
    prompt = "You are an AI research assistant.\n"
    prompt += f"Analyze the following {len(chunk_summaries)} paper summaries and extract:\n"
    prompt += "- Key themes\n- Strengths (pros)\n- Weaknesses (cons)\n- Research gaps\n\n"

    for i, s in enumerate(chunk_summaries, 1):
        prompt += f"Summary {i}:\n{s.strip()}\n\n"

    prompt += (
        "Return a JSON with keys: themes (list), pros (list), cons (list), gaps (list)."
    )

    resp = try_llm_completion(prompt, expect_json=False, max_tokens=350)
    if isinstance(resp, dict) and resp.get("error"):
        return {"error": resp["error"], "chunk": idx}
    try:
        data = json.loads(resp)
        return data
    except Exception:
        return {"raw": resp, "chunk": idx}


def _merge_json_chunks(chunks):
    """Merge multiple insight JSONs intelligently."""
    merged = {"themes": [], "pros": [], "cons": [], "gaps": []}
    for c in chunks:
        for key in merged.keys():
            val = c.get(key)
            if isinstance(val, list):
                merged[key].extend(val)
    # Deduplicate entries
    for key in merged.keys():
        merged[key] = list({v.strip(): None for v in merged[key] if v.strip()}.keys())
    return merged


def synthesize_insights(summaries):
    """
    Parallel + cached insight synthesis.
    Splits summaries into small batches for speed and token efficiency.
    """
    start = time.time()
    if not summaries:
        return {"themes": [], "pros": [], "cons": [], "gaps": []}

    key = str(abs(hash(" ".join(summaries[:5]))))
    cached = _cache_get(key)
    if cached:
        print(f"[Insights] Cache hit ({len(cached.get('themes', []))} themes)")
        return cached

    # Split into manageable chunks of 3 summaries each
    chunk_size = 3
    chunks = [summaries[i:i + chunk_size] for i in range(0, len(summaries), chunk_size)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_generate_insight_chunk, c, idx): idx for idx, c in enumerate(chunks)}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    merged = _merge_json_chunks(results)
    _cache_set(key, merged)

    print(f"[Insights] Generated in {time.time() - start:.2f}s from {len(summaries)} summaries")
    return merged
