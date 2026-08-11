"""Model management endpoints: list, download (with progress), select."""
import gc
import os
import threading
import logging

from fastapi import APIRouter, HTTPException
from llama_cpp import Llama

from .. import state
from ..config import MODELS_DIR, AVAILABLE_MODELS
from ..gpu import get_gpu_layers
from ..llm import save_current_model_setting, download_model_background

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models")
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
            "active": key == state.CURRENT_MODEL,
        })
    return {"models": models, "current_model": state.CURRENT_MODEL}


@router.post("/v1/models/download/{model_key}")
async def download_model(model_key: str):
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, "Model not found")

    model_path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[model_key]["filename"])
    if os.path.exists(model_path):
        return {"status": "already_downloaded"}

    if state.download_progress["status"] == "downloading":
        raise HTTPException(400, "A download is already in progress")

    state.download_progress = {"status": "starting", "progress": 0, "message": "Initialising...", "model_key": model_key}
    thread = threading.Thread(target=download_model_background, args=(model_key,))
    thread.start()
    return {"status": "started", "model_key": model_key}


@router.get("/v1/models/download/progress")
async def get_download_progress():
    return state.download_progress


@router.post("/v1/models/select/{model_key}")
async def select_model(model_key: str):
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, "Model not found")

    model_info = AVAILABLE_MODELS[model_key]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        raise HTTPException(400, "Model not downloaded yet. Download it first.")

    if state.llm_instance is not None:
        log.info(f"Unloading current model: {state.CURRENT_MODEL}")
        state.llm_instance = None
        gc.collect()

    try:
        log.info(f"Loading model: {model_path}")
        state.llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=get_gpu_layers())
        state.CURRENT_MODEL = model_key
        save_current_model_setting(model_key)
        log.info(f"Switched to model: {model_key}")
        return {"status": "ok", "model": model_key}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")
