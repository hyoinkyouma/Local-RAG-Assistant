"""Health / startup-status endpoint."""
from fastapi import APIRouter

from .. import state
from ..llm import get_current_model_param_size, supports_function_calling

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.llm_instance is not None,
        "current_model": state.CURRENT_MODEL,
        "param_size_b": get_current_model_param_size(),
        "supports_function_calling": supports_function_calling(),
        "gpu_available": state._gpu_info["available"],
        "gpu_name": state._gpu_info["name"],
    }
