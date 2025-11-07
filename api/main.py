# api/main.py
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import OPENAI_API_KEY
from agents.retriever_agent import retrieve_papers
from agents.summarizer_agent import summarize_papers
from agents.insight_agent import synthesize_insights
from agents.planner_agent import plan_research


# ----------------------------------------------------------------------
# FastAPI Configuration
# ----------------------------------------------------------------------
app = FastAPI(title="AI Research Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ----------------------------------------------------------------------
# Home Route
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve frontend HTML."""
    return templates.TemplateResponse("index.html", {"request": request})


# ----------------------------------------------------------------------
# Analyze Endpoint
# ----------------------------------------------------------------------
@app.get("/analyze", response_class=JSONResponse)
def analyze(topic: str = Query(..., description="Research topic"),
            top_k: int = Query(3, ge=1, le=8)):
    """Main route: retrieve papers, summarize, extract insights, and create plan."""
    if not OPENAI_API_KEY:
        return JSONResponse({"error": "Missing OPENAI_API_KEY"}, status_code=400)

    try:
        print(f"[ANALYZE] Topic: {topic}, TopK: {top_k}")

        # Step 1: Retrieve papers
        papers = retrieve_papers(topic, top_k=top_k)
        if not papers:
            return {"topic": topic, "papers": [], "summaries": [], "insights": {}, "plan": ""}

        # Step 2: Summarize
        summaries = summarize_papers(papers)
        summary_texts = [s.get("abstractive") or s.get("extractive") or "" for s in summaries]

        # Step 3: Generate insights
        insights = synthesize_insights(summary_texts)

        # Handle all possible formats from insights agent
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

        # Step 4: Generate research plan
        raw_text = parsed_insights.get("raw", "")
        plan = plan_research(raw_text, topic)

        # Final structured response
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


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[STARTUP] Launching FastAPI server...")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8080, reload=False)
