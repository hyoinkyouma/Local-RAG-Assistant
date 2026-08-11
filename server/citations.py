"""Citation grounding: keep only citations that actually overlap the answer."""
import logging
from urllib.parse import quote

from .schemas import Citation
from .text import significant_tokens

log = logging.getLogger(__name__)


def document_url(source: str, page) -> str:
    """Build a URL that opens a local document in the browser, jumping to page."""
    url = f"/v1/documents/{quote(source)}"
    if page is not None:
        url += f"#page={page + 1}"
    return url


def grounded_citations(candidates: list[dict], answer: str) -> list[Citation]:
    """Keep only citations whose source content actually overlaps the generated answer."""
    if not candidates or not answer:
        return []
    ans_tokens = significant_tokens(answer)
    if not ans_tokens:
        return []
    seen = set()
    grounded = []
    for c in candidates:
        chunk_tokens = significant_tokens(c["content"])
        if not chunk_tokens:
            continue
        matched = sum(1 for t in chunk_tokens if t in ans_tokens)
        ratio = matched / len(chunk_tokens)
        key = f"{c['source']}:{c['page']}"
        log.info(f"[citations] source={c['source']} page={c['page']} matched={matched} ratio={ratio:.2f}")
        if (matched >= 3 and ratio >= 0.08) or ratio >= 0.3:
            if key not in seen:
                seen.add(key)
                grounded.append(Citation(
                    source=c["source"],
                    page=c["page"],
                    content=c["content"][:300],
                    url=c.get("url") or document_url(c["source"], c["page"]),
                ))
    return grounded
