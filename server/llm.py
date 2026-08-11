"""Chat model management: current-model settings, paths, download."""
import os
import time
import logging

import requests

from . import state
from .config import MODELS_DIR, LLM_MODEL_PATH, AVAILABLE_MODELS
from .settings import save_settings

log = logging.getLogger(__name__)


def load_current_model_setting():
    if state.CURRENT_MODEL in AVAILABLE_MODELS:
        return
    for key, info in AVAILABLE_MODELS.items():
        if os.path.exists(os.path.join(MODELS_DIR, info["filename"])):
            state.CURRENT_MODEL = key
            save_settings()
            return
    state.CURRENT_MODEL = None


def save_current_model_setting(key: str):
    state.CURRENT_MODEL = key
    save_settings()


def get_current_model_param_size() -> float | None:
    if state.CURRENT_MODEL and state.CURRENT_MODEL in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[state.CURRENT_MODEL].get("param_size_b")
    return None


def get_current_model_path() -> str | None:
    if state.CURRENT_MODEL and state.CURRENT_MODEL in AVAILABLE_MODELS:
        path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[state.CURRENT_MODEL]["filename"])
        if os.path.exists(path):
            return path
    if os.path.exists(LLM_MODEL_PATH):
        return LLM_MODEL_PATH
    return None


def supports_function_calling() -> bool:
    if state.CURRENT_MODEL and state.CURRENT_MODEL in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[state.CURRENT_MODEL].get("allow_search", False)
    return False


def download_model_background(model_key: str):
    model_info = AVAILABLE_MODELS[model_key]
    dest_path = os.path.join(MODELS_DIR, model_info["filename"])
    tmp = dest_path + ".tmp"

    state.download_progress["status"] = "downloading"
    state.download_progress["progress"] = 0
    state.download_progress["message"] = f"Starting download of {model_info['name']}..."
    state.download_progress["model_key"] = model_key

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
                        state.download_progress["progress"] = pct
                        state.download_progress["message"] = f"Downloading {model_info['name']}... {pct}%"

            os.replace(tmp, dest_path)
            state.download_progress["status"] = "completed"
            state.download_progress["progress"] = 100
            state.download_progress["message"] = f"{model_info['name']} downloaded successfully"
            log.info(f"Downloaded {model_info['filename']} ({total} bytes)")
            return
        except Exception as e:
            if downloaded == 0:
                log.warning(f"Download failed (attempt {attempt+1}/{max_retries}): {e}")
                state.download_progress["message"] = f"Retrying ({attempt+1}/{max_retries})..."
                if os.path.exists(tmp):
                    os.remove(tmp)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            else:
                log.warning(f"Download failed after {downloaded} bytes: {e}")
                break

    state.download_progress["status"] = "error"
    state.download_progress["message"] = f"Download failed after {max_retries} attempts"
    if os.path.exists(tmp):
        os.remove(tmp)
