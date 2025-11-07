# models/get_summarizer.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from config import OPENAI_API_KEY, OPENAI_MODEL

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

def try_llm_completion(prompt: str, expect_json: bool = False, max_tokens: int = 512):
    if not client:
        return {"error": "OpenAI client not configured. Set OPENAI_API_KEY env var."}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are a helpful research assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        content = ""
        try:
            content = resp.choices[0].message.content.strip()
        except Exception:
            content = str(resp)
        return content
    except Exception as e:
        return {"error": str(e)}

def generate_abstractive(prompt: str, max_tokens: int = 256) -> str:
    resp = try_llm_completion(prompt, expect_json=False, max_tokens=max_tokens)
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(resp["error"])
    return resp or ""
