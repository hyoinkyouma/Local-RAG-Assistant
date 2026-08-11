"""Embedding model: GGUF-backed LangChain Embeddings + auto-download."""
import os
import time
import logging

import requests
from llama_cpp import Llama
from langchain_core.embeddings import Embeddings

from .config import EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_INFO
from .gpu import get_gpu_layers

log = logging.getLogger(__name__)


class LlamaCppEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.client = Llama(model_path=model_path, embedding=True, verbose=False, n_gpu_layers=get_gpu_layers())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.client.create_embedding(text)["data"][0]["embedding"] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.client.create_embedding(text)["data"][0]["embedding"]


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
