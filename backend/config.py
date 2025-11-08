import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Retrieval & Database settings
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "25"))
PUBMED_MAX_RESULTS = int(os.getenv("PUBMED_MAX_RESULTS", "25"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")

SQLITE_PATH = os.getenv("SQLITE_PATH", "data/research_memory.db")
dir_path = os.path.dirname(SQLITE_PATH)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

# Optional quick sanity log
print(f"[CONFIG] Using model={OPENAI_MODEL}, DB={SQLITE_PATH}, Embedding={EMBEDDING_MODEL}")
