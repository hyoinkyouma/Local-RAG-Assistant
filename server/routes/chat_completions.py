"""OpenAI-compatible chat completions endpoint."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import state
from ..schemas import ChatRequest
from ..chat import stream_chat, non_stream_chat

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if state.llm_instance is None:
        raise HTTPException(503, "No chat model loaded. Download one from Settings.")

    if req.stream:
        return StreamingResponse(stream_chat(req), media_type="text/event-stream")
    return non_stream_chat(req)
