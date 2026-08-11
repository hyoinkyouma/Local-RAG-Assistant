"""Chat history persistence endpoints (stored as JSON files)."""
import json
import os
import time
import uuid

from fastapi import APIRouter, HTTPException

from ..config import CHATS_DIR

router = APIRouter()


@router.get("/v1/chats")
async def list_chats():
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CHATS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            chats.append({
                "id": data["id"],
                "title": data.get("title", "New Chat"),
                "updated_at": data.get("updated_at", ""),
                "msg_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


@router.post("/v1/chats")
async def create_chat(body: dict):
    cid = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    data = {
        "id": cid,
        "title": body.get("title", "New Chat"),
        "created_at": now,
        "updated_at": now,
        "messages": body.get("messages", []),
    }
    with open(os.path.join(CHATS_DIR, f"{cid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"id": cid}


@router.get("/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Chat not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.put("/v1/chats/{chat_id}")
async def update_chat(chat_id: str, body: dict):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Chat not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "title" in body:
        data["title"] = body["title"]
    if "messages" in body:
        data["messages"] = body["messages"]
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"ok": True}


@router.patch("/v1/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, body: dict):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Chat not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["title"] = body.get("title", "New Chat")
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"ok": True}


@router.delete("/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}
