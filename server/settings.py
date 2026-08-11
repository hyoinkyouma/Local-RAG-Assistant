"""Persisted app settings, stored in a single settings.json file.

Holds the web search master switch, the domain list, the selected chat model,
and the embedding index marker. Legacy files (domains.json, models/current_model.txt,
chroma_db/embedding_model.txt) are migrated into settings.json on first load.
"""
import json
import logging
import os

from . import state
from .config import (
    SETTINGS_FILE,
    LEGACY_DOMAINS_FILE,
    LEGACY_CURRENT_MODEL_FILE,
    LEGACY_EMBEDDING_MODEL_FILE,
)

log = logging.getLogger(__name__)

DEFAULTS = {
    "web_search_enabled": True,
    "domains": ["General"],
}


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _remove_file(path):
    try:
        os.remove(path)
        log.info(f"Migrated and removed legacy file: {path}")
    except OSError:
        pass


def save_settings():
    data = {
        "web_search_enabled": bool(state.web_search_enabled),
        "domains": list(state.domains),
        "current_model": state.CURRENT_MODEL,
        "embedding_model": state.embedding_model_marker,
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        log.warning(f"Could not save settings: {e}")


def load_settings():
    data = _read_json(SETTINGS_FILE) or {}
    migrated = False

    state.web_search_enabled = bool(data.get("web_search_enabled", DEFAULTS["web_search_enabled"]))

    if isinstance(data.get("domains"), list) and data["domains"]:
        state.domains = list(data["domains"])
    else:
        legacy = _read_json(LEGACY_DOMAINS_FILE)
        if isinstance(legacy, list) and legacy:
            state.domains = list(legacy)
            data["domains"] = state.domains
            migrated = True
        else:
            state.domains = list(DEFAULTS["domains"])

    if data.get("current_model"):
        state.CURRENT_MODEL = data["current_model"]
    else:
        legacy = _read_text(LEGACY_CURRENT_MODEL_FILE)
        if legacy:
            state.CURRENT_MODEL = legacy
            data["current_model"] = legacy
            migrated = True
        else:
            state.CURRENT_MODEL = None

    if data.get("embedding_model"):
        state.embedding_model_marker = data["embedding_model"]
    else:
        legacy = _read_text(LEGACY_EMBEDDING_MODEL_FILE)
        if legacy:
            state.embedding_model_marker = legacy
            data["embedding_model"] = legacy
            migrated = True
        else:
            state.embedding_model_marker = None

    if migrated:
        save_settings()
        _remove_file(LEGACY_DOMAINS_FILE)
        _remove_file(LEGACY_CURRENT_MODEL_FILE)
        _remove_file(LEGACY_EMBEDDING_MODEL_FILE)

    log.info(
        f"Loaded settings: web_search_enabled={state.web_search_enabled} "
        f"domains={len(state.domains)} current_model={state.CURRENT_MODEL} "
        f"embedding_model={state.embedding_model_marker}"
    )


def set_web_search_enabled(enabled: bool) -> bool:
    state.web_search_enabled = bool(enabled)
    save_settings()
    log.info(f"Web search enabled -> {state.web_search_enabled}")
    return state.web_search_enabled
