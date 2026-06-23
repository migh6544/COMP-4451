from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "agentic_rag_course_docs"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 512
LLM_MODEL = "gpt-4o-mini"

CHUNK_TARGET_TOKENS = 650
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
TOP_K = 6
MIN_RELEVANT_RESULTS = 1
MAX_HISTORY_TURNS = 4
ANSWER_MAX_TOKENS = 700
