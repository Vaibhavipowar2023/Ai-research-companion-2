# agents/insight_agent.py
import json
from models.get_summarizer import try_llm_completion

def synthesize_insights(summaries):
    # summaries: list of strings
    prompt = "You are a research assistant. Given the following paper summaries, extract themes, pros, cons, and gaps as JSON:\n\n"
    for i, s in enumerate(summaries, 1):
        prompt += f"Summary {i}:\n{s}\n\n"
    prompt += "Return a JSON object with keys: themes (list), pros(list), cons(list), gaps(list)."

    resp = try_llm_completion(prompt, expect_json=False, max_tokens=400)
    if isinstance(resp, dict) and resp.get("error"):
        return {"error": resp["error"]}
    # Try to parse JSON anywhere in the response
    try:
        obj = json.loads(resp)
        return obj
    except Exception:
        return {"raw": str(resp)}
