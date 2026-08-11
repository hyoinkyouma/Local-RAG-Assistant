"""Central configuration, constants, and paths for the DocuStore backend."""
import os
import logging

from path_utils import RES_DIR, DATA_ROOT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Data directories ──
DATA_PATH = os.path.join(DATA_ROOT, "data")
UPLOAD_PATH = os.path.join(DATA_ROOT, "uploads")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
PERSIST_DIRECTORY = os.path.join(DATA_ROOT, "chroma_db")
CHATS_DIR = os.path.join(DATA_ROOT, "chats")
SETTINGS_FILE = os.path.join(DATA_ROOT, "settings.json")
# Legacy files migrated into settings.json on first load
LEGACY_DOMAINS_FILE = os.path.join(DATA_ROOT, "domains.json")
LEGACY_CURRENT_MODEL_FILE = os.path.join(MODELS_DIR, "current_model.txt")
LEGACY_EMBEDDING_MODEL_FILE = os.path.join(PERSIST_DIRECTORY, "embedding_model.txt")

# ── Embedding model ──
EMBEDDING_MODEL_FILENAME = "granite-embedding-english-r2.Q8_0.gguf"
EMBEDDING_MODEL_PATH = os.path.join(RES_DIR, "models", EMBEDDING_MODEL_FILENAME)
EMBEDDING_MODEL_INFO = {
    "repo_id": "mradermacher/granite-embedding-english-r2-GGUF",
    "filename": "granite-embedding-english-r2.Q8_0.gguf",
}

# ── Chat models ──
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

AVAILABLE_MODELS = {
    "granite-4.1-3b-instruct": {
        "id": "granite-4.1-3b-instruct",
        "name": "Granite 4.1 3B Instruct",
        "repo_id": "unsloth/granite-4.1-3b-GGUF",
        "filename": "granite-4.1-3b-UD-Q4_K_XL.gguf",
        "size_human": "~2.1 GB",
        "param_size_b": 3.0,
        "allow_search": True,
        "description": "IBM Granite 4.1 3B - strong tool calling & RAG, Apache-2.0"
    },
    "phi-3-mini-4k-instruct": {
        "id": "phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini 4K Instruct",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "size_human": "~2.2 GB",
        "param_size_b": 3.8,
        "allow_search": True,
        "description": "Microsoft's efficient 3.8B model"
    },
    "llama-3.2-1b-instruct": {
        "id": "llama-3.2-1b-instruct",
        "name": "Llama 3.2 1B Instruct",
        "repo_id": "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "size_human": "~0.8 GB",
        "param_size_b": 1.0,
        "allow_search": False,
        "description": "Fast and lightweight for basic Q&A"
    }
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    }
}

# Models with param_size_b >= this use native function calling instead of prompt injection.
FC_THRESHOLD_B = 4.0

# ── RAG tuning ──
CHUNK_SIZE = 700          # chars per chunk (was 300 — too small, broke sentence context)
CHUNK_OVERLAP = 100       # overlap between chunks (was 30)
RETRIEVAL_K = 6           # candidates fetched (was 2 — far too few)
RELEVANCE_THRESHOLD = 0.35  # min cosine sim for a chunk to reach the LLM
INDEX_VERSION = 2         # bump to force a clean index rebuild

SEARCH_INTENT_PATTERNS = [
    # Time-sensitive queries (only these should pull the model away from local docs)
    r"\b(latest|current|recent|up[- ]to[- ]date|breaking|newest)\b",
    r"\b(as of|as at)\b",
    r"\b(today|yesterday|tonight|this (year|month|week|quarter))\b",
    r"\b(20\d{2})\b",
    r"\b(news)\b",
    r"\b(weather|forecast|temperature|rain|storm)\b",
    r"\b(stock|share price|market|index|nasdaq|dow|s&p)\b",
    r"\b(score|result|winner|standing|fixture|match)\b",
    r"\b(population|GDP|inflation|unemployment|election)\b",
    r"\b(CEO|president|prime minister|chancellor|secretary)\b",
    r"\b(who (is|are|was|were)|what (is|are|was|were) the (latest|current|newest))\b",
]

STATIC_DIR = os.path.join(RES_DIR, "static")
