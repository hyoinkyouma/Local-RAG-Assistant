"""DocuStore Local Assistant — FastAPI backend.

The FastAPI ``app`` lives here. ``server.py`` re-exports it so the original
entry points (``python server.py``, ``uvicorn server:app``, ``gui.py``) keep
working unchanged.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .domains import ensure_domains
from .gpu import detect_gpu
from .index import build_resources
from .llm import load_current_model_setting

from .routes import (
    health,
    chat_completions,
    files,
    documents,
    domains,
    ingest,
    models,
    chats,
    settings,
)
from .settings import load_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.UPLOAD_PATH, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHATS_DIR, exist_ok=True)
    load_settings()
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

app.include_router(health.router)
app.include_router(chat_completions.router)
app.include_router(files.router)
app.include_router(documents.router)
app.include_router(domains.router)
app.include_router(ingest.router)
app.include_router(models.router)
app.include_router(chats.router)
app.include_router(settings.router)

app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(config.STATIC_DIR, "index.html"))
