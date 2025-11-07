# agents/planner_agent.py
from models.get_summarizer import generate_abstractive

def plan_research(insights_text: str, topic: str):
    prompt = f"Generate a short actionable research plan (4-6 steps) for '{topic}' based on these insights:\n\n{insights_text}"
    try:
        plan = generate_abstractive(prompt, max_tokens=400)
        return plan
    except Exception as e:
        return f"Planner failed: {e}"
