"""Serve ingested documents (PDF/TXT) for in-browser viewing."""
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..domains import load_domains, get_domain_path

router = APIRouter()


@router.get("/v1/documents/{filename:path}")
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
