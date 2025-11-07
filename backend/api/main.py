import sys, os, json, time, asyncio, concurrent.futures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.retriever_agent import retrieve_papers
from agents.summarizer_agent import summarize_papers
from agents.insight_agent import synthesize_insights
from agents.planner_agent import plan_research

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="AI Research Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Simple timer utility
def log_time(stage, start):
    print(f"[TIMER] {stage}: {time.time() - start:.2f}s")

# Cache folder
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_get(key):
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def cache_set(key, obj):
    path = os.path.join(CACHE_DIR, key + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

@app.on_event("startup")
def preload():
    import nltk
    try:
        nltk.download("punkt", quiet=True)
        print("[INIT] NLTK ready.")
    except Exception:
        pass

@app.get("/")
def home():
    return {"message": "AI Research Companion backend is running"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/analyze", response_class=JSONResponse)
async def analyze(topic: str, top_k: int = 3):
    if not OPENAI_API_KEY:
        return JSONResponse({"error": "Missing OPENAI_API_KEY"}, status_code=400)

    cache_key = topic.replace(" ", "_") + f"_{top_k}"
    cached = cache_get(cache_key)
    if cached:
        print("[CACHE] Hit for", topic)
        return cached

    try:
        t0 = time.time()
        papers = retrieve_papers(topic, top_k=top_k)
        log_time("Retrieve papers", t0)

        # Parallelize summarization in threadpool
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            t0 = time.time()
            summaries = await loop.run_in_executor(pool, summarize_papers, papers)
            log_time("Summarize papers", t0)

        # Generate insights + plan
        t0 = time.time()
        summary_texts = [s.get("abstractive") or s.get("extractive") or "" for s in summaries]
        insights = synthesize_insights(summary_texts)
        log_time("Synthesize insights", t0)

        t0 = time.time()
        raw_text = insights.get("raw", "") if isinstance(insights, dict) else str(insights)
        plan = plan_research(raw_text, topic)
        log_time("Generate plan", t0)

        response = {
            "topic": topic,
            "papers": papers,
            "summaries": summaries,
            "insights": insights,
            "plan": plan
        }
        cache_set(cache_key, response)
        log_time("TOTAL", t0)
        return response

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
