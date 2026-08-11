"""Domain helpers: load/save the domain list and locate per-domain file stores."""
import os
import shutil
import logging

from . import state
from .config import DATA_PATH
from .settings import save_settings

log = logging.getLogger(__name__)


def load_domains() -> list[str]:
    return list(state.domains)


def save_domains(domains: list[str]):
    state.domains = list(domains)
    save_settings()


def get_domain_path(domain: str) -> str:
    return os.path.join(DATA_PATH, domain)


def get_domain_files(domain: str) -> list[dict]:
    dpath = get_domain_path(domain)
    if not os.path.isdir(dpath):
        return []
    files = []
    for fname in sorted(os.listdir(dpath)):
        fpath = os.path.join(dpath, fname)
        if os.path.isfile(fpath):
            files.append({"name": fname, "size": os.path.getsize(fpath)})
    return files


def ensure_domains():
    """Migrate any flat files in data/ into the General domain on first run."""
    domains = load_domains()
    changed = False
    if "General" not in domains:
        domains.insert(0, "General")
        changed = True
    dst = get_domain_path("General")
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(DATA_PATH):
        fpath = os.path.join(DATA_PATH, fname)
        if os.path.isfile(fpath):
            shutil.move(fpath, os.path.join(dst, fname))
            changed = True
    if changed:
        save_domains(domains)
