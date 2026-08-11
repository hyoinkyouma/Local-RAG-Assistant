"""Pydantic request/response schemas for the API."""
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    web_search: bool | None = False
    enable_thinking: bool = False
    disable_rag: bool = False
    domains: list[str] | None = None


class Citation(BaseModel):
    source: str
    page: int | None = None
    content: str
    url: str | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict
    citations: list[Citation] = []


class IngestRequest(BaseModel):
    domain: str = "General"
