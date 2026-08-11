"""App settings endpoints (currently the web search master switch)."""
from fastapi import APIRouter
from pydantic import BaseModel

from .. import state
from ..settings import set_web_search_enabled

router = APIRouter()


class SettingsUpdate(BaseModel):
    web_search_enabled: bool


@router.get("/v1/settings")
async def get_settings():
    return {"web_search_enabled": bool(state.web_search_enabled)}


@router.patch("/v1/settings")
async def update_settings(body: SettingsUpdate):
    return {"web_search_enabled": set_web_search_enabled(body.web_search_enabled)}
