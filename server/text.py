"""Small text utilities: thinking stripping, tool-call parsing, token helpers."""
import json
import logging
import math
import re

log = logging.getLogger(__name__)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and orphan </think> tags with leading text."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?\s*think\s*/?>', '', text, flags=re.IGNORECASE)
    return text.strip()


def parse_qwen_tool_call(xml_text: str) -> dict | None:
    """Parse Qwen tool call (JSON or XML inside <tool_call>) into {name, arguments} dict."""
    content = xml_text.strip()
    if content.startswith("<tool_call>"):
        content = content[len("<tool_call>"):]
    if content.endswith("</tool_call>"):
        content = content[:-len("</tool_call>")]
    content = content.strip()

    log.info(f"[parse_tool] inner content (first 200): {content[:200]}")

    # JSON format: {"name": "web_search", "arguments": {"query": "..."}}
    if content.startswith("{"):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                log.info(f"[parse_tool] parsed JSON: name={obj['name']} args={args}")
                return {"name": obj["name"], "arguments": args}
        except (json.JSONDecodeError, TypeError) as e:
            log.info(f"[parse_tool] JSON parse failed: {e}")

    # Fallback XML format: <function=web_search><parameter=query>...</parameter>
    m = re.search(r'<function=(\w+)>', content)
    if m:
        name = m.group(1)
        args = {}
        for param_m in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', content, re.DOTALL):
            args[param_m.group(1)] = param_m.group(2).strip()
        log.info(f"[parse_tool] parsed XML: name={name} args={args}")
        return {"name": name, "arguments": args}
    log.info(f"[parse_tool] no format matched")
    return None


_STOPWORDS = set((
    "a an and or but if of to in on for with as at by from up down is are was were "
    "be been being this that these those it its not no so than then them they their "
    "he she we you i do does did have has had will would can could should may might "
    "about into over under the your my our what when where which who whom how why "
    "between among during before after because while each some any all both few more "
    "most other such only own same too very just also".split()
))


def significant_tokens(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if (len(t) >= 3 or t.isdigit()) and t not in _STOPWORDS}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)
