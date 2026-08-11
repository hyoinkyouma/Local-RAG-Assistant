"""Mutable global state shared across the app.

Kept in one place so modules can read/write the live objects without
circular imports. Prefer mutating dicts in place; for rebinding (e.g.
``state.llm_instance = ...``) always go through this module so every
consumer sees the update.
"""
llm_instance = None
retriever = None
vector_store = None
embeddings_instance = None
ingestion_progress = {"status": "idle", "current": 0, "total": 0, "current_file": "", "message": ""}
download_progress = {"status": "idle", "progress": 0, "message": ""}
CURRENT_MODEL = None
_gpu_info = {"available": False, "name": None}
web_search_enabled = True  # master switch; persisted to settings.json
domains = ["General"]      # domain list; persisted to settings.json
embedding_model_marker = None  # "{embedding_filename}|{INDEX_VERSION}"; persisted to settings.json
