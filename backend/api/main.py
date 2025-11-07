import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.retriever_agent import retrieve_papers
from agents.summarizer_agent import summarize_papers
from agents.insight_agent import synthesize_insights
from agents.planner_agent import plan_research

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")

app = FastAPI(title="AI Research Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        return {"topic": topic, "papers": papers, "summaries": summaries, "insights": parsed_insights, "plan": plan}

    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8080)
