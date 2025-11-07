import os
import json
import time
from hashlib import md5
from models.get_summarizer import generate_abstractive

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_get(key):
    path = os.path.join(CACHE_DIR, f"{key}.plan.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _cache_set(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.plan.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def plan_research(insights_text: str, topic: str):
    """
    Generate an actionable research plan using insights.
    Cached, token-efficient, and fault-tolerant.
    """
    start = time.time()

    # Empty input safety
    if not insights_text.strip():
        return "No insights available to generate a research plan."

    # Create cache key based on topic + insight hash
    cache_key = md5((topic + insights_text[:500]).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        print(f"[Planner] Cache hit for topic: '{topic}'")
        return cached.get("plan", "")

    # Dynamically scale token limit based on input length
    token_limit = min(250 + len(insights_text) // 4, 600)

    prompt = (
        f"You are an expert research assistant.\n\n"
        f"Topic: {topic}\n\n"
        "Given these insights, design a clear, concise, and actionable research plan.\n"
        "Each step should be specific, realistic, and relevant to academic or applied research.\n"
        "Include 4–6 structured steps with a short explanation for each.\n\n"
        "Insights:\n"
        f"{insights_text.strip()}\n\n"
        "Return the plan as plain text with numbered steps (Step 1, Step 2, ...)."
    )

    # Retry logic for robustness
    attempt = 0
    while attempt < 3:
        try:
            plan = generate_abstractive(prompt, max_tokens=token_limit)
            if plan and len(plan.strip()) > 30:
                _cache_set(cache_key, {"plan": plan})
                print(f"[Planner] Generated in {time.time() - start:.2f}s")
                return plan
        except Exception as e:
            print(f"[Planner ERROR] Attempt {attempt+1}: {e}")
        attempt += 1
        time.sleep(1.5)

    return "Planner failed after multiple attempts."
