import os
import sys
import time
import logging
import threading
import gc
import shutil
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import json
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ddgs import DDGS
import requests

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from llama_cpp import Llama, llama_supports_gpu_offload

from path_utils import RES_DIR, DATA_ROOT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = os.path.join(DATA_ROOT, "data")
UPLOAD_PATH = os.path.join(DATA_ROOT, "uploads")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
PERSIST_DIRECTORY = os.path.join(DATA_ROOT, "chroma_db")
CHATS_DIR = os.path.join(DATA_ROOT, "chats")
DOMAINS_CONFIG = os.path.join(DATA_ROOT, "domains.json")
EMBEDDING_MODEL_FILENAME = "granite-embedding-english-r2.Q8_0.gguf"
EMBEDDING_MODEL_PATH = os.path.join(RES_DIR, "models", EMBEDDING_MODEL_FILENAME)
EMBEDDING_MODEL_INFO = {
    "repo_id": "mradermacher/granite-embedding-english-r2-GGUF",
    "filename": "granite-embedding-english-r2.Q8_0.gguf",
}
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
CURRENT_MODEL_FILE = os.path.join(MODELS_DIR, "current_model.txt")

# ── Domain helpers ──

def load_domains() -> list[str]:
    if os.path.exists(DOMAINS_CONFIG):
        with open(DOMAINS_CONFIG) as f:
            return json.load(f)
    return ["General"]

def save_domains(domains: list[str]):
    os.makedirs(os.path.dirname(DOMAINS_CONFIG), exist_ok=True)
    with open(DOMAINS_CONFIG, "w") as f:
        json.dump(domains, f, indent=2)

def get_domain_path(domain: str) -> str:
    return os.path.join(DATA_ROOT, "data", domain)

def get_domain_files(domain: str) -> list[dict]:
    dpath = get_domain_path(domain)
    if not os.path.isdir(dpath):
        return []
    files = []
    for fname in sorted(os.listdir(dpath)):
        fpath = os.path.join(dpath, fname)
        if os.path.isfile(fpath):
            files.append({"name": fname, "size": os.path.getsize(fpath)})
    return files

def ensure_domains():
    """Migrate old flat data/ into domain structure on first run."""
    if os.path.exists(DOMAINS_CONFIG):
        return
    domains = ["General"]
    dst = get_domain_path("General")
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(DATA_PATH):
        fpath = os.path.join(DATA_PATH, fname)
        if os.path.isfile(fpath):
            shutil.move(fpath, os.path.join(dst, fname))
    save_domains(domains)

# ── /Domain helpers ──

# ── RAG tuning ──
CHUNK_SIZE = 700          # chars per chunk (was 300 — too small, broke sentence context)
CHUNK_OVERLAP = 100       # overlap between chunks (was 30)
RETRIEVAL_K = 6           # candidates fetched (was 2 — far too few)
RELEVANCE_THRESHOLD = 0.35  # min cosine sim for a chunk to reach the LLM
INDEX_VERSION = 2         # bump to force a clean index rebuild

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

FC_THRESHOLD_B = 4.0

# ── GPU Detection ──
_gpu_info = {"available": False, "name": None}

def detect_gpu():
    if not llama_supports_gpu_offload():
        log.info("GPU offload not available (llama-cpp-python compiled without CUDA)")
        return
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip().split("\n")[0]
            _gpu_info["available"] = True
            _gpu_info["name"] = name
            log.info(f"GPU detected: {name}")
        else:
            log.info("nvidia-smi failed — no NVIDIA GPU or driver issue")
    except Exception as e:
        log.info(f"GPU detection skipped: {e}")

def get_gpu_layers() -> int:
    return -1 if _gpu_info["available"] else 0


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and orphan </think> tags with leading text."""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?\s*think\s*/?>', '', text, flags=re.IGNORECASE)
    return text.strip()


def _parse_qwen_tool_call(xml_text: str) -> dict | None:
    """Parse Qwen tool call (JSON or XML inside <tool_call>) into {name, arguments} dict."""
    content = xml_text.strip()
    if content.startswith("<tool_call>"):
        content = content[len("<tool_call>"):]
    if content.endswith("</tool_call>"):
        content = content[:-len("</tool_call>")]
    content = content.strip()

    log.info(f"[parse_tool] inner content (first 200): {content[:200]}")

    # JSON format: {"name": "web_search", "arguments": {"query": "..."}}
    if content.startswith("{"):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                log.info(f"[parse_tool] parsed JSON: name={obj['name']} args={args}")
                return {"name": obj["name"], "arguments": args}
        except (json.JSONDecodeError, TypeError) as e:
            log.info(f"[parse_tool] JSON parse failed: {e}")

    # Fallback XML format: <function=web_search><parameter=query>...</parameter>
    import re
    m = re.search(r'<function=(\w+)>', content)
    if m:
        name = m.group(1)
        args = {}
        for param_m in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', content, re.DOTALL):
            args[param_m.group(1)] = param_m.group(2).strip()
        log.info(f"[parse_tool] parsed XML: name={name} args={args}")
        return {"name": name, "arguments": args}
    log.info(f"[parse_tool] no format matched")
    return None


def supports_function_calling() -> bool:
    if CURRENT_MODEL and CURRENT_MODEL in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[CURRENT_MODEL].get("allow_search", False)
    return False

class LlamaCppEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.client = Llama(model_path=model_path, embedding=True, verbose=False, n_gpu_layers=get_gpu_layers())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.client.create_embedding(text)["data"][0]["embedding"] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.client.create_embedding(text)["data"][0]["embedding"]


llm_instance = None
retriever = None
vector_store = None
embeddings_instance = None
ingestion_progress = {"status": "idle", "current": 0, "total": 0, "current_file": "", "message": ""}
download_progress = {"status": "idle", "progress": 0, "message": ""}
CURRENT_MODEL = None


def load_current_model_setting():
    global CURRENT_MODEL
    if os.path.exists(CURRENT_MODEL_FILE):
        with open(CURRENT_MODEL_FILE) as f:
            key = f.read().strip()
            if key in AVAILABLE_MODELS:
                CURRENT_MODEL = key
                return
    for key, info in AVAILABLE_MODELS.items():
        if os.path.exists(os.path.join(MODELS_DIR, info["filename"])):
            CURRENT_MODEL = key
            save_current_model_setting(key)
            return
    CURRENT_MODEL = None


def save_current_model_setting(key: str):
    with open(CURRENT_MODEL_FILE, "w") as f:
        f.write(key)

def get_current_model_param_size() -> float | None:
    if CURRENT_MODEL and CURRENT_MODEL in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[CURRENT_MODEL].get("param_size_b")
    return None


def get_current_model_path() -> str | None:
    if CURRENT_MODEL and CURRENT_MODEL in AVAILABLE_MODELS:
        path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[CURRENT_MODEL]["filename"])
        if os.path.exists(path):
            return path
    if os.path.exists(LLM_MODEL_PATH):
        return LLM_MODEL_PATH
    return None


def ensure_embedding_model() -> bool:
    if os.path.exists(EMBEDDING_MODEL_PATH):
        return True
    log.info("Embedding model not found, downloading...")
    os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)
    url = f"https://huggingface.co/{EMBEDDING_MODEL_INFO['repo_id']}/resolve/main/{EMBEDDING_MODEL_INFO['filename']}"
    tmp = EMBEDDING_MODEL_PATH + ".tmp"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=(10, 60))
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        log.info(f"Downloading embedding model... {pct}%")
            os.replace(tmp, EMBEDDING_MODEL_PATH)
            log.info(f"Embedding model downloaded ({downloaded} bytes)")
            return True
        except Exception as e:
            log.warning(f"Embedding download failed (attempt {attempt+1}/{max_retries}): {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    log.error(f"Embedding download failed after {max_retries} attempts")
    return False


def ensure_embedding_index_matches():
    """Wipe the persisted Chroma index if it was built with a different embedding model or index version."""
    marker_path = os.path.join(PERSIST_DIRECTORY, "embedding_model.txt")
    if os.path.exists(PERSIST_DIRECTORY):
        expected = f"{EMBEDDING_MODEL_FILENAME}|{INDEX_VERSION}"
        actual = None
        if os.path.exists(marker_path):
            with open(marker_path) as f:
                actual = f.read().strip()
        if actual != expected:
            log.info(f"Index format changed ({actual or 'unknown'} -> {expected}); rebuilding index")
            for entry in os.listdir(PERSIST_DIRECTORY):
                entry_path = os.path.join(PERSIST_DIRECTORY, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)


def mark_embedding_index_matches():
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    with open(os.path.join(PERSIST_DIRECTORY, "embedding_model.txt"), "w") as f:
        f.write(f"{EMBEDDING_MODEL_FILENAME}|{INDEX_VERSION}")


def load_documents_from_data_dir() -> list:
    """Load all supported documents (.pdf + .txt) from the data dir."""
    raw_documents = []
    if not os.path.isdir(DATA_PATH):
        return raw_documents
    for f in sorted(os.listdir(DATA_PATH)):
        fpath = os.path.join(DATA_PATH, f)
        if not os.path.isfile(fpath):
            continue
        try:
            if f.lower().endswith(".pdf"):
                raw_documents.extend(PyPDFLoader(fpath).load())
            elif f.lower().endswith(".txt"):
                raw_documents.extend(TextLoader(fpath, encoding="utf-8").load())
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    return raw_documents


def indexed_source_files(vector_store) -> set:
    """Return the set of source filenames already present in the index."""
    try:
        data = vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        return {os.path.basename(m.get("source", "")) for m in metadatas if m and m.get("source")}
    except Exception as e:
        log.warning(f"Could not read indexed sources: {e}")
        return set()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def build_resources():
    global llm_instance, retriever, vector_store, embeddings_instance

    if os.path.isdir(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except Exception:
            pass

    ensure_embedding_index_matches()

    if not ensure_embedding_model():
        log.error("Embedding model unavailable. RAG features disabled.")
        embeddings_instance = None
        model_path = get_current_model_path()
        if model_path:
            log.info(f"Loading chat model: {model_path}")
            llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=get_gpu_layers())
        return

    log.info("Initializing embedding model")
    embeddings_instance = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)

    model_path = get_current_model_path()
    if model_path:
        log.info(f"Loading chat model: {model_path}")
        llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=get_gpu_layers())
    else:
        log.warning("No chat model found. Use Settings to download one.")

    log.info("Loading and Chunking Documents from all domains")
    raw_documents = []
    domains = load_domains()
    for domain in domains:
        dpath = get_domain_path(domain)
        if not os.path.isdir(dpath):
            continue
        pdf_files = [f for f in os.listdir(dpath) if f.endswith(".pdf")]
        if pdf_files:
            for pdf_file in pdf_files:
                loader = PyPDFLoader(os.path.join(dpath, pdf_file))
                for doc in loader.load():
                    doc.metadata["domain"] = domain
                    raw_documents.append(doc)
        else:
            loader = DirectoryLoader(dpath, glob="*.txt", loader_cls=TextLoader)
            for doc in loader.load():
                doc.metadata["domain"] = domain
                raw_documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(raw_documents)
    log.info(f"Loaded {len(raw_documents)} source pages -> {len(docs)} chunks across {len(domains)} domain(s): {domains}")

    if docs:
        log.info("Opening Vector Store")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings_instance,
            collection_metadata={"hnsw:space": "cosine"},
        )
        existing = indexed_source_files(vector_store)
        new_docs = [d for d in docs if os.path.basename(d.metadata.get("source", "")) not in existing]
        if new_docs:
            log.info(f"Adding {len(new_docs)} new chunks (skipping {len(docs) - len(new_docs)} already indexed)")
            vector_store.add_documents(new_docs)
        else:
            log.info("All documents already indexed")
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    else:
        log.info("No documents found — vector store will be created on first ingestion")
    mark_embedding_index_matches()


def run_ingestion(file_paths: list[str], domain: str = "General"):
    global ingestion_progress, retriever, vector_store, embeddings_instance

    ensure_embedding_index_matches()
    embeddings = embeddings_instance or LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    domain_dir = get_domain_path(domain)
    os.makedirs(domain_dir, exist_ok=True)

    total = len(file_paths)
    ingestion_progress["status"] = "running"
    ingestion_progress["total"] = total
    ingestion_progress["domain"] = domain

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        ingestion_progress["current"] = i + 1
        ingestion_progress["current_file"] = filename
        ingestion_progress["message"] = f"Loading {filename}..."

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()

            docs = text_splitter.split_documents(raw_docs)

            for doc in docs:
                doc.metadata["domain"] = domain

            try:
                vector_store._collection.delete(where={"source": file_path})
            except Exception:
                pass

            ingestion_progress["message"] = f"Indexing {filename} ({len(docs)} chunks)..."
            vector_store.add_documents(docs)

            dest = os.path.join(domain_dir, filename)
            shutil.move(file_path, dest)

            ingestion_progress["message"] = f"Processed {filename} ({len(docs)} chunks)"
        except Exception as e:
            log.warning(f"Error processing {filename}: {e}")
            ingestion_progress["message"] = f"Error: {filename} - {e}"

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    mark_embedding_index_matches()
    ingestion_progress["status"] = "completed"
    ingestion_progress["message"] = f"Ingested {total} file(s) into '{domain}'"
    


def download_model_background(model_key: str):
    global download_progress, llm_instance

    model_info = AVAILABLE_MODELS[model_key]
    dest_path = os.path.join(MODELS_DIR, model_info["filename"])
    tmp = dest_path + ".tmp"

    download_progress["status"] = "downloading"
    download_progress["progress"] = 0
    download_progress["message"] = f"Starting download of {model_info['name']}..."
    download_progress["model_key"] = model_key

    url = f"https://huggingface.co/{model_info['repo_id']}/resolve/main/{model_info['filename']}"
    log.info(f"Downloading {url}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=(10, 60))
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        download_progress["progress"] = pct
                        download_progress["message"] = f"Downloading {model_info['name']}... {pct}%"

            os.replace(tmp, dest_path)
            download_progress["status"] = "completed"
            download_progress["progress"] = 100
            download_progress["message"] = f"{model_info['name']} downloaded successfully"
            log.info(f"Downloaded {model_info['filename']} ({total} bytes)")
            return
        except Exception as e:
            if downloaded == 0:
                log.warning(f"Download failed (attempt {attempt+1}/{max_retries}): {e}")
                download_progress["message"] = f"Retrying ({attempt+1}/{max_retries})..."
                if os.path.exists(tmp):
                    os.remove(tmp)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            else:
                log.warning(f"Download failed after {downloaded} bytes: {e}")
                break

    download_progress["status"] = "error"
    download_progress["message"] = f"Download failed after {max_retries} attempts"
    if os.path.exists(tmp):
        os.remove(tmp)


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


def requires_web_search(query: str) -> bool:
    import re
    for pat in SEARCH_INTENT_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            log.info(f"Intent classifier matched: {pat!r} -> enabling web search")
            return True
    return False


def web_search(query: str, max_results: int = 3) -> tuple[str, list[dict]]:
    """Return (formatted_text, citations_list) where each citation has source and content."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        snippets = []
        citations = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            link = r.get("link", r.get("href", ""))
            snippets.append(f"Title: {title}\n{body}")
            if link:
                citations.append({"source": link, "content": (title + " — " + body)[:300], "url": link})
        text = "\n\n".join(snippets) if snippets else "No results found."
        log.info(f"Web search got {len(results)} results, {len(text)} chars, {len(citations)} citations")
        return text, citations
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return "Web search failed.", []


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHATS_DIR, exist_ok=True)
    ensure_domains()
    load_current_model_setting()
    detect_gpu()
    build_resources()
    yield


app = FastAPI(title="DocuStore Local Assistant API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    web_search: bool | None = False
    enable_thinking: bool = False
    disable_rag: bool = False
    domains: list[str] | None = None


class Citation(BaseModel):
    source: str
    page: int | None = None
    content: str
    url: str | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict
    citations: list[Citation] = []


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": llm_instance is not None,
        "current_model": CURRENT_MODEL,
        "param_size_b": get_current_model_param_size(),
        "supports_function_calling": supports_function_calling(),
        "gpu_available": _gpu_info["available"],
        "gpu_name": _gpu_info["name"],
    }


def retrieve_docs(query: str) -> list:
    """Vector retrieval plus cosine-similarity filtering so irrelevant chunks never reach the LLM."""
    if retriever is None:
        return []
    doc_chunks = retriever.invoke(query)
    if not doc_chunks or embeddings_instance is None:
        return doc_chunks
    try:
        q_emb = embeddings_instance.embed_query(query)
        kept = []
        for d in doc_chunks:
            sim = _cosine_similarity(q_emb, embeddings_instance.embed_query(d.page_content))
            log.info(f"[retrieve] sim={sim:.3f} source={os.path.basename(d.metadata.get('source',''))} | {d.page_content[:60]!r}")
            if sim >= RELEVANCE_THRESHOLD:
                kept.append(d)
        if kept:
            log.info(f"[retrieve] kept {len(kept)}/{len(doc_chunks)} chunks (threshold={RELEVANCE_THRESHOLD})")
        else:
            log.info("[retrieve] no chunks above threshold — answering without context")
        return kept
    except Exception as e:
        log.warning(f"Relevance filtering failed: {e}")
        return doc_chunks


_STOPWORDS = set((
    "a an and or but if of to in on for with as at by from up down is are was were "
    "be been being this that these those it its not no so than then them they their "
    "he she we you i do does did have has had will would can could should may might "
    "about into over under the your my our what when where which who whom how why "
    "between among during before after because while each some any all both few more "
    "most other such only own same too very just also".split()
))


def _significant_tokens(text: str) -> set:
    import re
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if (len(t) >= 3 or t.isdigit()) and t not in _STOPWORDS}


def _document_url(source: str, page) -> str:
    """Build a URL that opens a local document in the browser, jumping to page."""
    url = f"/v1/documents/{quote(source)}"
    if page is not None:
        url += f"#page={page + 1}"
    return url


def _grounded_citations(candidates: list[dict], answer: str) -> list[Citation]:
    """Keep only citations whose source content actually overlaps the generated answer."""
    if not candidates or not answer:
        return []
    ans_tokens = _significant_tokens(answer)
    if not ans_tokens:
        return []
    seen = set()
    grounded = []
    for c in candidates:
        chunk_tokens = _significant_tokens(c["content"])
        if not chunk_tokens:
            continue
        matched = sum(1 for t in chunk_tokens if t in ans_tokens)
        ratio = matched / len(chunk_tokens)
        key = f"{c['source']}:{c['page']}"
        log.info(f"[citations] source={c['source']} page={c['page']} matched={matched} ratio={ratio:.2f}")
        if (matched >= 3 and ratio >= 0.08) or ratio >= 0.3:
            if key not in seen:
                seen.add(key)
                grounded.append(Citation(
                    source=c["source"],
                    page=c["page"],
                    content=c["content"][:300],
                    url=c.get("url") or _document_url(c["source"], c["page"]),
                ))
    return grounded


def build_messages_and_context(req: ChatRequest):
    query = ""
    for m in reversed(req.messages):
        if m.content and m.role == "user":
            query = m.content
            break

    doc_chunks = []
    if not req.disable_rag:
        if vector_store is not None and req.domains:
            search_kwargs = {"k": RETRIEVAL_K}
            search_kwargs["filter"] = {"domain": {"$in": req.domains}}
            local_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            doc_chunks = retrieve_docs(query) if retriever else local_retriever.invoke(query)
        elif retriever is not None:
            doc_chunks = retrieve_docs(query)
    
    rag_context = "\n\n".join(d.page_content for d in doc_chunks)

    candidate_citations = []
    seen = set()
    for d in doc_chunks:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            candidate_citations.append({
                "source": os.path.basename(src),
                "page": page,
                "content": d.page_content,
                "url": None,
            })

    use_fc = supports_function_calling()
    # Auto-trigger web search by intent only for models that cannot call tools
    # themselves (allow_search=false). Models with allow_search=true decide via
    # tool calling, so the intent classifier stays out of the way.
    do_web_search = req.web_search or (requires_web_search(query) if not use_fc else False)
    log.info(f"[build] query={query[:80]} do_web_search={do_web_search} use_fc={use_fc} model_size={get_current_model_param_size()}B rag_chunks={len(doc_chunks)}")

    # Small models (<4B): prompt injection. Large models (>=4B): tool calling.
    web_citations = []
    if do_web_search:
        if not use_fc:
            log.info(f"Web search triggered (flag={req.web_search}, intent={do_web_search}) — prompt injection (<{FC_THRESHOLD_B}B)")
            web_results, web_citations_tmp = web_search(query)
            web_citations = web_citations_tmp
            for wc in web_citations:
                candidate_citations.append({
                    "source": wc["source"],
                    "page": None,
                    "content": wc["content"],
                    "url": wc.get("url") or wc["source"],
                })
            rag_context = rag_context + "\n\n---\nWeb search results:\n" + web_results if rag_context else f"Web search results:\n{web_results}"
        else:
            log.info(f"Web search triggered (flag={req.web_search}, intent={do_web_search}) — tool calling")

    if rag_context:
        extra_inst = ""
        if use_fc and do_web_search:
            extra_inst = "\n- Use the web_search tool ONLY if the information above is insufficient or the user needs current/real-time information."
        elif use_fc:
            extra_inst = "\n- You have access to the web_search tool. Use it only if the information above is insufficient."
        system_content = (
            "You are a helpful assistant. Answer the user's question using the information provided below.\n"
            "\n"
            "Information:\n"
            f"{rag_context}\n"
            "\n"
            "Instructions:\n"
            "- Answer based on the information above.\n"
            "- If the information does not contain the answer, say so.\n"
            "- If both local information and web search results are present, prefer the local information unless it is outdated or incomplete.\n"
            "- Keep answers concise.\n"
            "- Prefer plain prose. Use bullet points or numbered lists ONLY when the answer genuinely contains multiple distinct items; never use a single bullet for a simple answer.\n"
            f"{extra_inst}\n"
            "- Do NOT mention or discuss the format, source, or limitations of the information provided. Just answer the question.\n"
            "- Do NOT include any thinking, reasoning, or analysis. Only provide the final answer."
        )
    else:
        tool_note = (" You have access to the web_search tool — use it when you need current or up-to-date information."
                     if use_fc else "")
        force_search = (" The user needs current information. You MUST use the web_search tool to find the answer before responding."
                        if use_fc and do_web_search else "")
        system_content = (
            "You are a helpful assistant. Answer the user's question concisely in plain prose; only use bullet points or numbered lists when the answer genuinely contains multiple distinct items."
            f"{tool_note}{force_search}\n"
            "Do NOT include any thinking, reasoning, or analysis. Only propyvide the final answer."
        )



    messages = [{"role": "system", "content": system_content}]
    for m in req.messages:
        msg = {"role": m.role}
        if m.content:
            msg["content"] = m.content
        if m.tool_calls:
            tcs = []
            for tc in m.tool_calls:
                tc = tc.model_dump() if hasattr(tc, 'model_dump') else tc
                if tc.get("function", {}).get("arguments") and isinstance(tc["function"]["arguments"], str):
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                tcs.append(tc)
            msg["tool_calls"] = tcs
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        messages.append(msg)

    return query, candidate_citations, do_web_search, use_fc, messages


def _yield_tool_call_chunks(chat_id, model_name, now, tcd, sse):
    """Yield SSE chunks for a tool call."""
    tool_id = f"call_{now}_web_search"
    yield sse({
        "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": None}, "finish_reason": None}]
    })
    yield sse({
        "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
        "choices": [{"index": 0, "delta": {
            "role": None, "content": None,
            "tool_calls": [{"index": 0, "id": tool_id, "type": "function", "function": {"name": tcd["name"], "arguments": json.dumps(tcd["arguments"])}}]
        }, "finish_reason": None}]
    })
    yield sse({
        "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    })


def _execute_and_feed(kwargs, content_acc, tcd, query, candidate_citations):
    """Execute a tool call, feed the result into kwargs messages, and append citations."""
    name = tcd["name"]
    args = tcd.get("arguments", {})
    search_query = args.get("query", query)
    log.info(f"[exec_tool] name={name} search_query={search_query} content_acc_len={len(content_acc)}")
    result, web_citations = web_search(search_query) if name == "web_search" else (f"Unknown tool: {name}", [])
    log.info(f"[exec_tool] web_search returned {len(result)} chars, {len(web_citations)} citations")

    for wc in web_citations:
        candidate_citations.append({
            "source": wc["source"],
            "page": None,
            "content": wc["content"],
            "url": wc.get("url") or wc["source"],
        })

    assistant_msg = {"role": "assistant", "content": content_acc or None, "tool_calls": [
        {"id": f"call_{int(time.time())}_{name}", "type": "function",
         "function": {"name": name, "arguments": args}}
    ]}
    kwargs["messages"].append(assistant_msg)
    kwargs["messages"].append({"role": "tool", "tool_call_id": assistant_msg["tool_calls"][0]["id"], "content": result})


def stream_chat(req: ChatRequest):
    query, candidate_citations, do_web_search, use_fc, messages = build_messages_and_context(req)

    kwargs = {
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.3,
        "max_tokens": min(req.max_tokens or 1024, 2048),
    }
    if use_fc:
        kwargs["tools"] = [WEB_SEARCH_TOOL]

    now = int(time.time())
    model_name = req.model
    chat_id = f"chatcmpl-{now}"
    resp_parts = []

    def sse(event: dict):
        return f"data: {json.dumps(event, default=str)}\n\n"

    def content_chunk(text: str, finish: str | None = None):
        if text:
            resp_parts.append(text)
        return sse({
            "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
            "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}]
        })

    for iteration in range(4 if use_fc else 1):
        log.info(f"[stream] iteration={iteration} use_fc={use_fc} do_web_search={do_web_search} messages={len(kwargs['messages'])} tools={'tools' in kwargs}")
        stream = llm_instance.create_chat_completion(**kwargs, stream=True)
        content_acc = ""
        finish_reason = None
        think_buf = ""
        confirm_buf = None
        tc_buf = ""
        in_tc = False
        iter_called_tool = False

        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if not delta.get("content"):
                continue
            raw_text = delta["content"]

            if use_fc and not in_tc:
                # Check for tool call start in accumulated content
                check = content_acc + raw_text
                idx = check.find("<tool_call>")
                if idx != -1:
                    log.info(f"[stream] tool_call detected at idx={idx} in accumulated buffer (len={len(check)})")
                    content_acc = ""
                    tc_buf = check[idx + len("<tool_call>"):]
                    in_tc = True
                    if "</tool_call>" in tc_buf:
                        tc_text = "<tool_call>" + tc_buf.split("</tool_call>")[0] + "</tool_call>"
                        tcd = _parse_qwen_tool_call(tc_text)
                        log.info(f"[stream] tool_call complete in first chunk, parsed={tcd is not None}")
                        if tcd:
                            yield from _yield_tool_call_chunks(chat_id, model_name, now, tcd, sse)
                            _execute_and_feed(kwargs, content_acc, tcd, query, candidate_citations)
                            iter_called_tool = True
                            content_acc = ""
                            in_tc = False
                            kwargs.pop("tools", None)
                            break
                    continue

            if use_fc and in_tc:
                tc_buf += raw_text
                log.info(f"[stream] in_tc accumulated tc_buf len={len(tc_buf)}")
                if "</tool_call>" in tc_buf:
                    tc_text = "<tool_call>" + tc_buf.split("</tool_call>")[0] + "</tool_call>"
                    tcd = _parse_qwen_tool_call(tc_text)
                    log.info(f"[stream] tool_call closed, parsed={tcd is not None}")
                    if tcd:
                        yield from _yield_tool_call_chunks(chat_id, model_name, now, tcd, sse)
                        _execute_and_feed(kwargs, content_acc, tcd, query, candidate_citations)
                        kwargs.pop("tools", None)
                        iter_called_tool = True
                        content_acc = ""
                    in_tc = False
                    break
                continue

            # Normal content path
            if not req.enable_thinking:
                if think_buf is not None:
                    think_buf += raw_text
                    if "</think>" in think_buf:
                        if "<think>" in think_buf:
                            resp = think_buf.split("</think>", 1)[1]
                            if resp:
                                log.info(f"[stream] think block closed, yielding {len(resp)} chars after </think>")
                                content_acc += resp
                                yield content_chunk(resp)
                            else:
                                log.info(f"[stream] think block closed, nothing after </think>")
                            think_buf = None
                        else:
                            log.info(f"[stream] orphan </think> in think_buf (no <think>), redirecting to confirm_buf")
                            _, _, after = think_buf.partition("</think>")
                            think_buf = None
                            if after:
                                confirm_buf = after
                    elif len(think_buf) > 4000:
                        log.info(f"[stream] think_buf overflow at 4000, flushing")
                        content_acc += think_buf
                        yield content_chunk(think_buf)
                        think_buf = None
                elif confirm_buf is not None:
                    # Suspicious mode: buffering to check if this is more thinking
                    if "</think>" in raw_text:
                        log.info(f"[stream] orphan </think> - discarded {len(confirm_buf)} chars of thinking, entering direct mode")
                        _, _, after = raw_text.partition("</think>")
                        confirm_buf = None
                        if after:
                            content_acc += after
                            yield content_chunk(after)
                    else:
                        confirm_buf += raw_text
                else:
                    if "</think>" in raw_text:
                        log.info(f"[stream] orphan </think> in direct stream, entering suspicious mode")
                        before, _, after = raw_text.partition("</think>")
                        if before:
                            content_acc += before
                            yield content_chunk(before)
                        confirm_buf = after if after else ""
                    else:
                        content_acc += raw_text
                        yield content_chunk(raw_text)
            else:
                content_acc += raw_text
                yield content_chunk(raw_text)

        if think_buf and not confirm_buf:
            log.info(f"[stream] flushing final think_buf ({len(think_buf)} chars)")
            content_acc += think_buf
            yield content_chunk(think_buf)
            think_buf = None
        if confirm_buf:
            log.info(f"[stream] flushing final confirm_buf ({len(confirm_buf)} chars)")
            content_acc += confirm_buf
            yield content_chunk(confirm_buf)
            confirm_buf = None

        log.info(f"[stream] end of iteration {iteration}: iter_called_tool={iter_called_tool} in_tc={in_tc} finish_reason={finish_reason} content_acc_len={len(content_acc)}")
        if iter_called_tool:
            log.info(f"[stream] tool was called, continuing outer loop")
            continue

        if not in_tc:
            log.info(f"[stream] no tool call this iteration, yielding stop + usage")
            yield sse({"type": "citations", "citations": [
                {"source": c.source, "page": c.page, "content": c.content, "url": c.url} for c in _grounded_citations(candidate_citations, "".join(resp_parts))
            ]})
            yield content_chunk("", finish_reason or "stop")
            yield sse({"type": "usage", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        if not in_tc:
            break
    else:
        log.info(f"[stream] outer loop exhausted (max iterations)")
        yield sse({"type": "citations", "citations": [
            {"source": c.source, "page": c.page, "content": c.content, "url": c.url} for c in _grounded_citations(candidate_citations, "".join(resp_parts))
        ]})
        yield content_chunk("", "stop")

    grounded = _grounded_citations(candidate_citations, "".join(resp_parts))
    yield sse({"type": "citations", "citations": [
        {"source": c.source, "page": c.page, "content": c.content, "url": c.url} for c in grounded
    ]})


def non_stream_chat(req: ChatRequest):
    query, candidate_citations, do_web_search, use_fc, messages = build_messages_and_context(req)

    kwargs = {
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.7,
        "max_tokens": min(req.max_tokens or 1024, 2048),
    }
    if use_fc:
        kwargs["tools"] = [WEB_SEARCH_TOOL]

    for _ in range(4 if use_fc else 1):
        resp = llm_instance.create_chat_completion(**kwargs)
        answer = resp["choices"][0]["message"].get("content", "")

        if use_fc and "<tool_call>" in answer:
            tcd = _parse_qwen_tool_call(answer)
            if tcd:
                _execute_and_feed(kwargs, "", tcd, query, candidate_citations)
                kwargs.pop("tools", None)
                continue

        if not req.enable_thinking:
            answer = _strip_thinking(answer)

        now = int(time.time())
        return ChatResponse(
            id=f"chatcmpl-{now}", created=now, model=req.model,
            choices=[{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            citations=_grounded_citations(candidate_citations, answer),
        )

    now = int(time.time())
    return ChatResponse(
        id=f"chatcmpl-{now}", created=now, model=req.model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        citations=[],
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if llm_instance is None:
        raise HTTPException(503, "No chat model loaded. Download one from Settings.")

    if req.stream:
        return StreamingResponse(stream_chat(req), media_type="text/event-stream")
    return non_stream_chat(req)


# ── File endpoints ──

@app.post("/v1/files/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved.append(file.filename)
    return {"status": "ok", "files": saved}


@app.get("/v1/files")
async def list_uploaded_files():
    files_list = []
    for f in sorted(os.listdir(UPLOAD_PATH)):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            files_list.append({"name": f, "size": os.path.getsize(fpath)})
    return {"files": files_list}


@app.delete("/v1/files/{filename:path}")
async def delete_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    os.remove(file_path)
    return {"status": "deleted"}


@app.post("/v1/files/clear")
async def clear_uploaded_files():
    for f in os.listdir(UPLOAD_PATH):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"status": "cleared"}


class IngestRequest(BaseModel):
    domain: str = "General"

# ── Domain endpoints ──

@app.get("/v1/domains")
async def list_domains():
    domains = load_domains()
    result = []
    for d in domains:
        files = get_domain_files(d)
        result.append({"name": d, "file_count": len(files)})
    return {"domains": result}


@app.post("/v1/domains")
async def create_domain(body: dict):
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Domain name is required")
    domains = load_domains()
    if name in domains:
        raise HTTPException(400, f"Domain '{name}' already exists")
    domains.append(name)
    save_domains(domains)
    os.makedirs(get_domain_path(name), exist_ok=True)
    return {"status": "created", "domain": name}


@app.delete("/v1/domains/{name}")
async def delete_domain(name: str):
    if name == "General":
        raise HTTPException(400, "Cannot delete the default 'General' domain")
    domains = load_domains()
    if name not in domains:
        raise HTTPException(404, f"Domain '{name}' not found")
    domains.remove(name)
    save_domains(domains)
    # Remove domain directory
    dpath = get_domain_path(name)
    if os.path.isdir(dpath):
        shutil.rmtree(dpath)
    # Remove from ChromaDB
    global vector_store
    if vector_store is not None:
        try:
            vector_store._collection.delete(where={"domain": name})
        except Exception as e:
            log.warning(f"Failed to delete ChromaDB entries for domain '{name}': {e}")
    return {"status": "deleted", "domain": name}


@app.get("/v1/domains/{name}/files")
async def list_domain_files(name: str):
    domains = load_domains()
    if name not in domains:
        raise HTTPException(404, f"Domain '{name}' not found")
    files = get_domain_files(name)
    return {"domain": name, "files": files}


@app.delete("/v1/domains/{name}/files/{filename:path}")
async def delete_domain_file(name: str, filename: str):
    domains = load_domains()
    if name not in domains:
        raise HTTPException(404, f"Domain '{name}' not found")
    fpath = os.path.join(get_domain_path(name), filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")
    os.remove(fpath)
    # Remove chunks for this file from ChromaDB
    global vector_store
    if vector_store is not None:
        try:
            result = vector_store._collection.get(where={"domain": name})
            all_ids = result.get("ids", [])
            all_metadatas = result.get("metadatas", [])
            file_ids = []
            for doc_id, meta in zip(all_ids, all_metadatas):
                source = os.path.basename(meta.get("source", "")) if meta else ""
                if source == filename:
                    file_ids.append(doc_id)
            if file_ids:
                vector_store._collection.delete(ids=file_ids)
        except Exception as e:
            log.warning(f"Failed to delete ChromaDB entries for file '{filename}': {e}")
    return {"status": "deleted"}


# ── /Domain endpoints ──

@app.get("/v1/documents/{filename:path}")
async def serve_document(filename: str):
    """Serve an ingested document (PDF/TXT) for in-browser viewing. Browsers
    honor the #page=N fragment to jump to a specific page."""
    base = os.path.basename(filename)
    if base != filename or not base:
        raise HTTPException(400, "Invalid filename")
    for domain in load_domains():
        fpath = os.path.join(get_domain_path(domain), base)
        if os.path.isfile(fpath):
            media = "application/pdf" if base.lower().endswith(".pdf") else "text/plain; charset=utf-8"
            return FileResponse(fpath, media_type=media, headers={"Content-Disposition": "inline"})
    raise HTTPException(404, "Document not found")


@app.post("/v1/ingest")
async def start_ingestion(req: IngestRequest = IngestRequest()):
    global ingestion_progress

    domains = load_domains()
    if req.domain not in domains:
        raise HTTPException(400, f"Domain '{req.domain}' does not exist. Create it first.")

    file_paths = [
        os.path.join(UPLOAD_PATH, f)
        for f in os.listdir(UPLOAD_PATH)
        if os.path.isfile(os.path.join(UPLOAD_PATH, f))
    ]
    if not file_paths:
        raise HTTPException(400, "No files to ingest")

    if ingestion_progress["status"] == "running":
        raise HTTPException(400, "Ingestion already in progress")

    ingestion_progress = {"status": "running", "current": 0, "total": len(file_paths), "current_file": "", "message": "Starting..."}
    thread = threading.Thread(target=run_ingestion, args=(file_paths, req.domain))
    thread.start()
    return {"status": "started", "file_count": len(file_paths), "domain": req.domain}


@app.get("/v1/ingest/progress")
async def get_ingestion_progress():
    return ingestion_progress


# ── Model endpoints ──

@app.get("/v1/models")
async def list_models():
    models = []
    for key, info in AVAILABLE_MODELS.items():
        model_path = os.path.join(MODELS_DIR, info["filename"])
        models.append({
            "id": key,
            "name": info["name"],
            "repo_id": info["repo_id"],
            "filename": info["filename"],
            "size_human": info["size_human"],
            "description": info["description"],
            "downloaded": os.path.exists(model_path),
            "active": key == CURRENT_MODEL,
        })
    return {"models": models, "current_model": CURRENT_MODEL}


@app.post("/v1/models/download/{model_key}")
async def download_model(model_key: str):
    global download_progress

    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, "Model not found")

    model_path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[model_key]["filename"])
    if os.path.exists(model_path):
        return {"status": "already_downloaded"}

    if download_progress["status"] == "downloading":
        raise HTTPException(400, "A download is already in progress")

    download_progress = {"status": "starting", "progress": 0, "message": "Initialising...", "model_key": model_key}
    thread = threading.Thread(target=download_model_background, args=(model_key,))
    thread.start()
    return {"status": "started", "model_key": model_key}


@app.get("/v1/models/download/progress")
async def get_download_progress():
    return download_progress


@app.post("/v1/models/select/{model_key}")
async def select_model(model_key: str):
    global llm_instance, CURRENT_MODEL

    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, "Model not found")

    model_info = AVAILABLE_MODELS[model_key]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        raise HTTPException(400, "Model not downloaded yet. Download it first.")

    if llm_instance is not None:
        log.info(f"Unloading current model: {CURRENT_MODEL}")
        llm_instance = None
        gc.collect()

    try:
        log.info(f"Loading model: {model_path}")
        llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=get_gpu_layers())
        CURRENT_MODEL = model_key
        save_current_model_setting(model_key)
        log.info(f"Switched to model: {model_key}")
        return {"status": "ok", "model": model_key}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")


# ── Chat persistence ───────────────────────────────────────────────────

@app.get("/v1/chats")
async def list_chats():
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CHATS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            chats.append({
                "id": data["id"],
                "title": data.get("title", "New Chat"),
                "updated_at": data.get("updated_at", ""),
                "msg_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


@app.post("/v1/chats")
async def create_chat(body: dict):
    cid = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    data = {
        "id": cid,
        "title": body.get("title", "New Chat"),
        "created_at": now,
        "updated_at": now,
        "messages": body.get("messages", []),
    }
    with open(os.path.join(CHATS_DIR, f"{cid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"id": cid}


@app.get("/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Chat not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.put("/v1/chats/{chat_id}")
async def update_chat(chat_id: str, body: dict):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Chat not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "title" in body:
        data["title"] = body["title"]
    if "messages" in body:
        data["messages"] = body["messages"]
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"ok": True}


@app.patch("/v1/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, body: dict):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Chat not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["title"] = body.get("title", "New Chat")
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"ok": True}


@app.delete("/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}


STATIC_DIR = os.path.join(RES_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
