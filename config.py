# =============================================================
#  config.py — Central Configuration
#  All tuneable parameters live here. Change once, affects all.
# =============================================================

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

# ── OpenAI ────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-ada-002"
CHAT_MODEL: str = "gpt-4o"
MAX_TOKENS: int = 1024
TEMPERATURE: float = 0.0          # 0 = deterministic; good for factual Q&A

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
CHROMA_DIR: str = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
ESCALATION_LOG: str = os.path.join(LOGS_DIR, "escalation_log.jsonl")

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# ── Retrieval ─────────────────────────────────────────────────
TOP_K: int = 4                     # Number of chunks to retrieve per query

# ── Routing ───────────────────────────────────────────────────
MIN_ANSWER_LENGTH: int = 30        # Answers shorter than this → escalate

UNCERTAINTY_PHRASES: list[str] = [
    "i don't have enough information",
    "i do not have enough information",
    "i'm not sure",
    "i am not sure",
    "i cannot find",
    "i can't find",
    "not mentioned in the context",
    "not mentioned in the provided context",
    "the context does not",
    "the provided context does not",
    "no information",
    "i don't know",
    "i do not know",
]

SENSITIVE_KEYWORDS: list[str] = [
    "refund",
    "lawsuit",
    "legal",
    "lawyer",
    "complaint",
    "compensation",
    "sue",
    "court",
    "attorney",
    "dispute",
    "fraud",
    "chargeback",
]

# ── LangGraph ─────────────────────────────────────────────────
THREAD_ID: str = "rag-support-session"    # Default single-user thread ID

# ── LangSmith Tracing ─────────────────────────────────────────
# Get your key at: https://smith.langchain.com
# Set LANGSMITH_TRACING=true in .env to enable
LANGSMITH_API_KEY: str        = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING: bool       = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_PROJECT: str        = os.getenv("LANGSMITH_PROJECT", "rag-customer-support")
LANGSMITH_ENDPOINT: str       = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Propagate to LangChain env vars (LangChain reads these automatically)
if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]    = LANGSMITH_ENDPOINT

# ── FastAPI ────────────────────────────────────────────────────
API_HOST: str  = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int  = int(os.getenv("API_PORT", "8000"))
API_TITLE: str = "RAG Customer Support API"
API_VERSION: str = "1.0.0"

# ── Streamlit ─────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE: str = "RAG Customer Support Assistant"
STREAMLIT_PAGE_ICON: str  = "🤖"

# ── Display ───────────────────────────────────────────────────
SEPARATOR: str = "=" * 60
