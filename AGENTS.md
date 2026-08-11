# Local RAG — Agent Guide

## Commands

```powershell
# Run
python -m server                         # FastAPI on http://0.0.0.0:8000
python gui.py                            # pywebview desktop window (starts server + polls /health)
python app.py                            # standalone CLI RAG query (no server)
python -m uvicorn server:app --reload    # dev with auto-reload

# Build distributable
python build.py                              # both CPU + CUDA
python build.py --variant cpu                # CPU-only (~260 MB) → dist/LocalRAG-CPU/
python build.py --variant cuda               # CUDA-enabled (~2 GB) → dist/LocalRAG-CUDA/
```

No tests, no lint/typecheck/CI config exists.

## Architecture

- **Backend**: `server/` package — FastAPI, llama.cpp, ChromaDB (langchain Chroma wrapper), DuckDuckGo web search (`ddgs`). Run with `python -m server`, `uvicorn server:app`, or `python gui.py` (`gui.py` imports `from server import app`).
  - `server/config.py` — constants, paths, `AVAILABLE_MODELS`, RAG tuning
  - `server/state.py` — mutable globals (`llm_instance`, `retriever`, `vector_store`, `CURRENT_MODEL`, progress dicts)
  - `server/domains.py`, `server/gpu.py`, `server/text.py`, `server/schemas.py`
  - `server/embeddings.py`, `server/llm.py`, `server/index.py` — model + Chroma index build/ingestion
  - `server/websearch.py`, `server/citations.py`, `server/chat.py` — retrieval, grounding, streaming/non-streaming completion
  - `server/routes/` — one module per API area (health, chat_completions, files, documents, domains, ingest, models, chats)
  - `server/__main__.py` — enables `python -m server`
- **Frontend**: `static/index.html` + `static/app.js` + `static/styles.css` — TailwindCSS via CDN only (no npm)
- **Desktop**: `gui.py` — pywebview window, runs uvicorn in background daemon thread, polls `/health` (10 min timeout)
- **Paths**: `path_utils.py` — `RES_DIR` (bundled resources) and `DATA_ROOT` (writable user data, `%APPDATA%\LocalRAG\` when frozen, else repo root)

## Key Conventions

### Static files
- Mounted at `/static/` in `server/__init__.py` (`app.mount("/static", ...)`) — links in HTML use `/static/styles.css`, `/static/app.js`
- `index.html` is served at `GET /` via the default route

### Dark mode
- Pure CSS `.dark` class overrides in `styles.css` — no Tailwind `dark:` variants
- Sun/moon icon visibility controlled via `#sun-icon`/`#moon-icon` ID selectors in CSS
- Toggle persisted to `localStorage` (`'theme': 'light'|'dark'`), falls back to `prefers-color-scheme`

### Model management
- Embedding model (`granite-embedding-english-r2.Q8_0.gguf`, IBM Granite, ModernBERT, 768-dim) is auto-downloaded at startup if missing (bundled in PyInstaller build)
- LLMs downloaded at runtime by user via Settings panel — uses `requests` streaming from Hugging Face (NOT `huggingface_hub`)
- Active model stored in `settings.json` (`current_model` key)
- `/v1/models/download/{id}` + `/v1/models/download/progress` for download tracking
- Download progress polling in JS is separate from ingestion polling

### File ingestion flow
1. Upload: `POST /v1/files/upload` (multipart) → staged in `uploads/`
2. List: `GET /v1/files`
3. Delete: `DELETE /v1/files/{name}` or `POST /v1/files/clear`
4. Ingest: `POST /v1/ingest` → background job, poll `GET /v1/ingest/progress`
5. After ingestion, files move from `uploads/` to `data/`

### Chat API
- `POST /v1/chat/completions` — OpenAI-compatible format, returns `citations` array alongside response
- Web search auto-triggers on time-sensitive keywords ("latest", "today", "news") — can force with `web_search: true`
- Web search master switch: `state.web_search_enabled` (default `True`, persisted to `settings.json`). When off, no web search happens at all — intent classifier, tool injection, and forced `web_search: true` are all ignored. Toggle via `GET/PATCH /v1/settings`; UI button is `#web-search-toggle`.

### Data directories
All under `DATA_ROOT`:
- `data/` — source documents (loaded on startup + after ingestion)
- `uploads/` — staging for new files before ingestion
- `models/` — downloaded LLM GGUF files
- `chroma_db/` — ChromaDB persistent index (auto-created)
- `settings.json` — single persisted settings file: `web_search_enabled`, `domains`, `current_model`, `embedding_model` index marker. Legacy `domains.json`, `models/current_model.txt`, and `chroma_db/embedding_model.txt` are migrated in and removed on first load (`server/settings.py`).

### Build notes
- `build.py` runs PyInstaller, bundles `static/` + embedding model; LLMs downloaded at runtime
- `--variant cpu` builds CPU-only (~260 MB); `--variant cuda` builds CUDA-enabled (~2 GB); default `all` builds both
- CPU build from source (compiles on first run); CUDA uses prebuilt wheel from GitHub
- Excludes heavy unused packages: `torch`, `sklearn`, `sentence_transformers`, `transformers`, `langgraph`, `langchain_classic`
- `--collect-all` only for `chromadb` and `llama_cpp` (native binaries); langchain packages use targeted `--hidden-import`
- Hidden imports: `uvicorn.logging`, `uvicorn.loops.auto`, `uvicorn.protocols.http.auto`, `ddgs`
- CPU build ~260 MB (down from ~2 GB by excluding CUDA DLLs and using CPU-only llama-cpp)
- First build takes 15–20 min due to dependency graph; subsequent builds are faster

## .gitignore (important)
Ignores: `models/*`, `data/*`, `chroma_db/*`, `uploads/*`, `env/`, `build/`, `dist/`, `*.spec`, `*.log`
