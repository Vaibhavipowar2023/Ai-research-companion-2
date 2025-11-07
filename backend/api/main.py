import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.retriever_agent import retrieve_papers
from agents.summarizer_agent import summarize_papers
from agents.insight_agent import synthesize_insights
from agents.planner_agent import plan_research

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="AI Research Companion")

# CORS so your Vercel frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # You can restrict this later to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Root route — simple health check
@app.get("/")
def home():
    return {"message": "AI Research Companion backend is running"}

# Analyze endpoint
@app.get("/analyze", response_class=JSONResponse)
def analyze(topic: str = Query(...), top_k: int = Query(3, ge=1, le=8)):
    if not OPENAI_API_KEY:
        return JSONResponse({"error": "Missing OPENAI_API_KEY"}, status_code=400)

    try:
        papers = retrieve_papers(topic, top_k=top_k)
        summaries = summarize_papers(papers)
        summary_texts = [s.get("abstractive") or s.get("extractive") or "" for s in summaries]
        insights = synthesize_insights(summary_texts)

        if isinstance(insights, dict):
            parsed_insights = insights
        elif isinstance(insights, str):
            try:
                parsed_insights = json.loads(insights)
                if not isinstance(parsed_insights, dict):
                    parsed_insights = {"raw": str(parsed_insights)}
            except json.JSONDecodeError:
                parsed_insights = {"raw": insights}
        else:
            parsed_insights = {"raw": str(insights)}

        plan = plan_research(parsed_insights.get("raw", ""), topic)
        return {
            "topic": topic,
            "papers": papers,
            "summaries": summaries,
            "insights": parsed_insights,
            "plan": plan
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
