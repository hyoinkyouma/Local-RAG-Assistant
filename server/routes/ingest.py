"""Ingestion endpoints: start a background ingestion job and poll its progress."""
import os
import threading

from fastapi import APIRouter, HTTPException

from .. import state
from ..config import UPLOAD_PATH
from ..domains import load_domains
from ..schemas import IngestRequest
from ..index import run_ingestion

router = APIRouter()


@router.post("/v1/ingest")
async def start_ingestion(req: IngestRequest = IngestRequest()):
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

    if state.ingestion_progress["status"] == "running":
        raise HTTPException(400, "Ingestion already in progress")

    state.ingestion_progress = {"status": "running", "current": 0, "total": len(file_paths), "current_file": "", "message": "Starting..."}
    thread = threading.Thread(target=run_ingestion, args=(file_paths, req.domain))
    thread.start()
    return {"status": "started", "file_count": len(file_paths), "domain": req.domain}


@router.get("/v1/ingest/progress")
async def get_ingestion_progress():
    return state.ingestion_progress
