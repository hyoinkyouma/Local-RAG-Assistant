"""Domain management endpoints."""
import os
import shutil
import logging

from fastapi import APIRouter, HTTPException

from .. import state
from ..domains import load_domains, save_domains, get_domain_path, get_domain_files

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/domains")
async def list_domains():
    domains = load_domains()
    result = []
    for d in domains:
        files = get_domain_files(d)
        result.append({"name": d, "file_count": len(files)})
    return {"domains": result}


@router.post("/v1/domains")
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


@router.delete("/v1/domains/{name}")
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
    if state.vector_store is not None:
        try:
            state.vector_store._collection.delete(where={"domain": name})
        except Exception as e:
            log.warning(f"Failed to delete ChromaDB entries for domain '{name}': {e}")
    return {"status": "deleted", "domain": name}


@router.get("/v1/domains/{name}/files")
async def list_domain_files(name: str):
    domains = load_domains()
    if name not in domains:
        raise HTTPException(404, f"Domain '{name}' not found")
    files = get_domain_files(name)
    return {"domain": name, "files": files}


@router.delete("/v1/domains/{name}/files/{filename:path}")
async def delete_domain_file(name: str, filename: str):
    domains = load_domains()
    if name not in domains:
        raise HTTPException(404, f"Domain '{name}' not found")
    fpath = os.path.join(get_domain_path(name), filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")
    os.remove(fpath)
    # Remove chunks for this file from ChromaDB
    if state.vector_store is not None:
        try:
            result = state.vector_store._collection.get(where={"domain": name})
            all_ids = result.get("ids", [])
            all_metadatas = result.get("metadatas", [])
            file_ids = []
            for doc_id, meta in zip(all_ids, all_metadatas):
                source = os.path.basename(meta.get("source", "")) if meta else ""
                if source == filename:
                    file_ids.append(doc_id)
            if file_ids:
                state.vector_store._collection.delete(ids=file_ids)
        except Exception as e:
            log.warning(f"Failed to delete ChromaDB entries for file '{filename}': {e}")
    return {"status": "deleted"}
