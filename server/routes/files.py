"""Upload staging: upload/list/delete/clear files before ingestion."""
import os

from fastapi import APIRouter, UploadFile, File, HTTPException

from ..config import UPLOAD_PATH

router = APIRouter()


@router.post("/v1/files/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved.append(file.filename)
    return {"status": "ok", "files": saved}


@router.get("/v1/files")
async def list_uploaded_files():
    files_list = []
    for f in sorted(os.listdir(UPLOAD_PATH)):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            files_list.append({"name": f, "size": os.path.getsize(fpath)})
    return {"files": files_list}


@router.delete("/v1/files/{filename:path}")
async def delete_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    os.remove(file_path)
    return {"status": "deleted"}


@router.post("/v1/files/clear")
async def clear_uploaded_files():
    for f in os.listdir(UPLOAD_PATH):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"status": "cleared"}
