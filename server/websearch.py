"""Web search via DuckDuckGo (ddgs), plus the intent classifier."""
import logging
import re

from ddgs import DDGS

from .config import SEARCH_INTENT_PATTERNS

log = logging.getLogger(__name__)


def requires_web_search(query: str) -> bool:
    for pat in SEARCH_INTENT_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            log.info(f"Intent classifier matched: {pat!r} -> enabling web search")
            return True
    return False


def web_search(query: str, max_results: int = 3) -> tuple[str, list[dict]]:
    """Return (formatted_text, citations_list) where each citation has source and content."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        snippets = []
        citations = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            link = r.get("link", r.get("href", ""))
            snippets.append(f"Title: {title}\n{body}")
            if link:
                citations.append({"source": link, "content": (title + " — " + body)[:300], "url": link})
        text = "\n\n".join(snippets) if snippets else "No results found."
        log.info(f"Web search got {len(results)} results, {len(text)} chars, {len(citations)} citations")
        return text, citations
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return "Web search failed.", []
