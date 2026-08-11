"""GPU detection and offload configuration."""
import logging

from . import state

log = logging.getLogger(__name__)


def detect_gpu():
    if not llama_supports_gpu_offload():
        log.info("GPU offload not available (llama-cpp-python compiled without CUDA)")
        return
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip().split("\n")[0]
            state._gpu_info["available"] = True
            state._gpu_info["name"] = name
            log.info(f"GPU detected: {name}")
        else:
            log.info("nvidia-smi failed — no NVIDIA GPU or driver issue")
    except Exception as e:
        log.info(f"GPU detection skipped: {e}")


def get_gpu_layers() -> int:
    return -1 if state._gpu_info["available"] else 0


def llama_supports_gpu_offload() -> bool:
    from llama_cpp import llama_supports_gpu_offload as _support
    return _support()
