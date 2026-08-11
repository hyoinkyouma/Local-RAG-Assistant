"""Chat orchestration: RAG retrieval, prompt building, streaming and non-streaming completion."""
import json
import os
import time
import logging

from . import state
from .config import RETRIEVAL_K, RELEVANCE_THRESHOLD, FC_THRESHOLD_B, WEB_SEARCH_TOOL
from .schemas import ChatRequest, ChatResponse
from .text import cosine_similarity, strip_thinking, parse_qwen_tool_call
from .citations import grounded_citations
from .websearch import web_search, requires_web_search
from .llm import supports_function_calling, get_current_model_param_size

log = logging.getLogger(__name__)


def retrieve_docs(query: str) -> list:
    """Vector retrieval plus cosine-similarity filtering so irrelevant chunks never reach the LLM."""
    if state.retriever is None:
        return []
    doc_chunks = state.retriever.invoke(query)
    if not doc_chunks or state.embeddings_instance is None:
        return doc_chunks
    try:
        q_emb = state.embeddings_instance.embed_query(query)
        kept = []
        for d in doc_chunks:
            sim = cosine_similarity(q_emb, state.embeddings_instance.embed_query(d.page_content))
            log.info(f"[retrieve] sim={sim:.3f} source={os.path.basename(d.metadata.get('source',''))} | {d.page_content[:60]!r}")
            if sim >= RELEVANCE_THRESHOLD:
                kept.append(d)
        if kept:
            log.info(f"[retrieve] kept {len(kept)}/{len(doc_chunks)} chunks (threshold={RELEVANCE_THRESHOLD})")
        else:
            log.info("[retrieve] no chunks above threshold — answering without context")
        return kept
    except Exception as e:
        log.warning(f"Relevance filtering failed: {e}")
        return doc_chunks


def build_context(req: ChatRequest):
    """Vector retrieval plus web-search decision. Does not execute the web search."""
    query = ""
    for m in reversed(req.messages):
        if m.content and m.role == "user":
            query = m.content
            break

    doc_chunks = []
    if not req.disable_rag:
        if state.vector_store is not None and req.domains:
            search_kwargs = {"k": RETRIEVAL_K}
            search_kwargs["filter"] = {"domain": {"$in": req.domains}}
            local_retriever = state.vector_store.as_retriever(search_kwargs=search_kwargs)
            doc_chunks = retrieve_docs(query) if state.retriever else local_retriever.invoke(query)
        elif state.retriever is not None:
            doc_chunks = retrieve_docs(query)

    rag_context = "\n\n".join(d.page_content for d in doc_chunks)

    candidate_citations = []
    seen = set()
    for d in doc_chunks:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            candidate_citations.append({
                "source": os.path.basename(src),
                "page": page,
                "content": d.page_content,
                "url": None,
            })

    use_fc = supports_function_calling()
    ws_enabled = state.web_search_enabled
    tool_available = use_fc and ws_enabled
    # Auto-trigger web search by intent only for models that cannot call tools
    # themselves (allow_search=false). Models with allow_search=true decide via
    # tool calling, so the intent classifier stays out of the way.
    # The master switch (state.web_search_enabled) disables web search entirely.
    do_web_search = ws_enabled and (req.web_search or (requires_web_search(query) if not use_fc else False))
    log.info(f"[build] query={query[:80]} do_web_search={do_web_search} use_fc={use_fc} ws_enabled={ws_enabled} model_size={get_current_model_param_size()}B rag_chunks={len(doc_chunks)}")

    return query, candidate_citations, do_web_search, use_fc, tool_available, rag_context


def apply_web_search(query: str, candidate_citations: list, rag_context: str):
    """Execute a web search and merge results into the context and candidate citations."""
    log.info(f"Web search triggered — prompt injection (<{FC_THRESHOLD_B}B)")
    web_results, web_citations = web_search(query)
    for wc in web_citations:
        candidate_citations.append({
            "source": wc["source"],
            "page": None,
            "content": wc["content"],
            "url": wc.get("url") or wc["source"],
        })
    rag_context = rag_context + "\n\n---\nWeb search results:\n" + web_results if rag_context else f"Web search results:\n{web_results}"
    return candidate_citations, rag_context


def build_messages(req: ChatRequest, rag_context: str, do_web_search: bool, use_fc: bool, tool_available: bool) -> list:
    """Build the OpenAI-style message list from the retrieved context."""
    if rag_context:
        extra_inst = ""
        if tool_available and do_web_search:
            extra_inst = "\n- Use the web_search tool ONLY if the information above is insufficient or the user needs current/real-time information."
        elif tool_available:
            extra_inst = "\n- You have access to the web_search tool. Use it only if the information above is insufficient."
        system_content = (
            "You are a helpful assistant. Answer the user's question using the information provided below.\n"
            "\n"
            "Information:\n"
            f"{rag_context}\n"
            "\n"
            "Instructions:\n"
            "- Answer based on the information above.\n"
            "- If the information does not contain the answer, say so.\n"
            "- If both local information and web search results are present, prefer the local information unless it is outdated or incomplete.\n"
            "- Keep answers concise.\n"
            "- Prefer plain prose. Use bullet points or numbered lists ONLY when the answer genuinely contains multiple distinct items; never use a single bullet for a simple answer.\n"
            f"{extra_inst}\n"
            "- Do NOT mention or discuss the format, source, or limitations of the information provided. Just answer the question.\n"
            "- Do NOT include any thinking, reasoning, or analysis. Only provide the final answer."
        )
    else:
        tool_note = (" You have access to the web_search tool — use it when you need current or up-to-date information."
                     if tool_available else "")
        force_search = (" The user needs current information. You MUST use the web_search tool to find the answer before responding."
                        if tool_available and do_web_search else "")
        system_content = (
            "You are a helpful assistant. Answer the user's question concisely in plain prose; only use bullet points or numbered lists when the answer genuinely contains multiple distinct items."
            f"{tool_note}{force_search}\n"
            "Do NOT include any thinking, reasoning, or analysis. Only propyvide the final answer."
        )

    messages = [{"role": "system", "content": system_content}]
    for m in req.messages:
        msg = {"role": m.role}
        if m.content:
            msg["content"] = m.content
        if m.tool_calls:
            tcs = []
            for tc in m.tool_calls:
                tc = tc.model_dump() if hasattr(tc, 'model_dump') else tc
                if tc.get("function", {}).get("arguments") and isinstance(tc["function"]["arguments"], str):
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                tcs.append(tc)
            msg["tool_calls"] = tcs
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        messages.append(msg)

    return messages


def build_messages_and_context(req: ChatRequest):
    """Full pipeline: retrieval + (if needed) web search + prompt building."""
    query, candidate_citations, do_web_search, use_fc, tool_available, rag_context = build_context(req)
    if do_web_search and not use_fc:
        candidate_citations, rag_context = apply_web_search(query, candidate_citations, rag_context)
    messages = build_messages(req, rag_context, do_web_search, use_fc, tool_available)
    return query, candidate_citations, do_web_search, use_fc, messages


def _yield_tool_call_chunks(chat_id, model_name, now, tcd, sse):
    """Yield SSE chunks for a tool call."""
    tool_id = f"call_{now}_web_search"
    yield sse({
        "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": None}, "finish_reason": None}]
    })
    yield sse({
        "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
        "choices": [{"index": 0, "delta": {
            "role": None, "content": None,
            "tool_calls": [{"index": 0, "id": tool_id, "type": "function", "function": {"name": tcd["name"], "arguments": json.dumps(tcd["arguments"])}}]
        }, "finish_reason": None}]
    })
    yield sse({
        "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    })


def _execute_and_feed(kwargs, content_acc, tcd, query, candidate_citations):
    """Execute a tool call, feed the result into kwargs messages, and append citations."""
    name = tcd["name"]
    args = tcd.get("arguments", {})
    search_query = args.get("query", query)
    log.info(f"[exec_tool] name={name} search_query={search_query} content_acc_len={len(content_acc)}")
    if name == "web_search" and not state.web_search_enabled:
        result, web_citations = "Web search is disabled by the user.", []
    elif name == "web_search":
        result, web_citations = web_search(search_query)
    else:
        result, web_citations = f"Unknown tool: {name}", []
    log.info(f"[exec_tool] web_search returned {len(result)} chars, {len(web_citations)} citations")

    for wc in web_citations:
        candidate_citations.append({
            "source": wc["source"],
            "page": None,
            "content": wc["content"],
            "url": wc.get("url") or wc["source"],
        })

    assistant_msg = {"role": "assistant", "content": content_acc or None, "tool_calls": [
        {"id": f"call_{int(time.time())}_{name}", "type": "function",
         "function": {"name": name, "arguments": args}}
    ]}
    kwargs["messages"].append(assistant_msg)
    kwargs["messages"].append({"role": "tool", "tool_call_id": assistant_msg["tool_calls"][0]["id"], "content": result})


def stream_chat(req: ChatRequest):
    now = int(time.time())
    model_name = req.model
    chat_id = f"chatcmpl-{now}"
    resp_parts = []

    def sse(event: dict):
        return f"data: {json.dumps(event, default=str)}\n\n"

    def status_event(phase: str, message: str):
        return sse({"type": "status", "phase": phase, "message": message})

    # Phase 1: local document retrieval (status is shown while it runs)
    rag_active = (not req.disable_rag) and (state.retriever is not None or state.vector_store is not None)
    if rag_active:
        yield status_event("searching_documents", "Searching your documents...")
    query, candidate_citations, do_web_search, use_fc, tool_available, rag_context = build_context(req)

    # Phase 2: web search (prompt injection path for models that can't call tools)
    if do_web_search and not use_fc:
        yield status_event("web_search", "Searching the web...")
        candidate_citations, rag_context = apply_web_search(query, candidate_citations, rag_context)

    messages = build_messages(req, rag_context, do_web_search, use_fc, tool_available)

    kwargs = {
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.3,
        "max_tokens": min(req.max_tokens or 1024, 2048),
    }
    if use_fc and state.web_search_enabled:
        kwargs["tools"] = [WEB_SEARCH_TOOL]

    yield status_event("generating", "Generating response...")

    def content_chunk(text: str, finish: str | None = None):
        if text:
            resp_parts.append(text)
        return sse({
            "id": chat_id, "object": "chat.completion.chunk", "created": now, "model": model_name,
            "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}]
        })

    for iteration in range(4 if use_fc else 1):
        if iteration > 0:
            yield status_event("generating", "Generating response...")
        log.info(f"[stream] iteration={iteration} use_fc={use_fc} do_web_search={do_web_search} messages={len(kwargs['messages'])} tools={'tools' in kwargs}")
        stream = state.llm_instance.create_chat_completion(**kwargs, stream=True)
        content_acc = ""
        finish_reason = None
        think_buf = ""
        confirm_buf = None
        tc_buf = ""
        in_tc = False
        iter_called_tool = False

        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if not delta.get("content"):
                continue
            raw_text = delta["content"]

            if use_fc and not in_tc:
                # Check for tool call start in accumulated content
                check = content_acc + raw_text
                idx = check.find("<tool_call>")
                if idx != -1:
                    log.info(f"[stream] tool_call detected at idx={idx} in accumulated buffer (len={len(check)})")
                    content_acc = ""
                    tc_buf = check[idx + len("<tool_call>"):]
                    in_tc = True
                    if "</tool_call>" in tc_buf:
                        tc_text = "<tool_call>" + tc_buf.split("</tool_call>")[0] + "</tool_call>"
                        tcd = parse_qwen_tool_call(tc_text)
                        log.info(f"[stream] tool_call complete in first chunk, parsed={tcd is not None}")
                        if tcd:
                            yield from _yield_tool_call_chunks(chat_id, model_name, now, tcd, sse)
                            yield status_event("web_search", "Searching the web...")
                            _execute_and_feed(kwargs, content_acc, tcd, query, candidate_citations)
                            iter_called_tool = True
                            content_acc = ""
                            in_tc = False
                            kwargs.pop("tools", None)
                            break
                    continue

            if use_fc and in_tc:
                tc_buf += raw_text
                log.info(f"[stream] in_tc accumulated tc_buf len={len(tc_buf)}")
                if "</tool_call>" in tc_buf:
                    tc_text = "<tool_call>" + tc_buf.split("</tool_call>")[0] + "</tool_call>"
                    tcd = parse_qwen_tool_call(tc_text)
                    log.info(f"[stream] tool_call closed, parsed={tcd is not None}")
                    if tcd:
                        yield from _yield_tool_call_chunks(chat_id, model_name, now, tcd, sse)
                        yield status_event("web_search", "Searching the web...")
                        _execute_and_feed(kwargs, content_acc, tcd, query, candidate_citations)
                        kwargs.pop("tools", None)
                        iter_called_tool = True
                        content_acc = ""
                    in_tc = False
                    break
                continue

            # Normal content path
            if not req.enable_thinking:
                if think_buf is not None:
                    think_buf += raw_text
                    if "</think>" in think_buf:
                        if "<think>" in think_buf:
                            resp = think_buf.split("</think>", 1)[1]
                            if resp:
                                log.info(f"[stream] think block closed, yielding {len(resp)} chars after </think>")
                                content_acc += resp
                                yield content_chunk(resp)
                            else:
                                log.info(f"[stream] think block closed, nothing after </think>")
                            think_buf = None
                        else:
                            log.info(f"[stream] orphan </think> in think_buf (no <think>), redirecting to confirm_buf")
                            _, _, after = think_buf.partition("</think>")
                            think_buf = None
                            if after:
                                confirm_buf = after
                    elif len(think_buf) > 4000:
                        log.info(f"[stream] think_buf overflow at 4000, flushing")
                        content_acc += think_buf
                        yield content_chunk(think_buf)
                        think_buf = None
                elif confirm_buf is not None:
                    # Suspicious mode: buffering to check if this is more thinking
                    if "</think>" in raw_text:
                        log.info(f"[stream] orphan </think> - discarded {len(confirm_buf)} chars of thinking, entering direct mode")
                        _, _, after = raw_text.partition("</think>")
                        confirm_buf = None
                        if after:
                            content_acc += after
                            yield content_chunk(after)
                    else:
                        confirm_buf += raw_text
                else:
                    if "</think>" in raw_text:
                        log.info(f"[stream] orphan </think> in direct stream, entering suspicious mode")
                        before, _, after = raw_text.partition("</think>")
                        if before:
                            content_acc += before
                            yield content_chunk(before)
                        confirm_buf = after if after else ""
                    else:
                        content_acc += raw_text
                        yield content_chunk(raw_text)
            else:
                content_acc += raw_text
                yield content_chunk(raw_text)

        if think_buf and not confirm_buf:
            log.info(f"[stream] flushing final think_buf ({len(think_buf)} chars)")
            content_acc += think_buf
            yield content_chunk(think_buf)
            think_buf = None
        if confirm_buf:
            log.info(f"[stream] flushing final confirm_buf ({len(confirm_buf)} chars)")
            content_acc += confirm_buf
            yield content_chunk(confirm_buf)
            confirm_buf = None

        log.info(f"[stream] end of iteration {iteration}: iter_called_tool={iter_called_tool} in_tc={in_tc} finish_reason={finish_reason} content_acc_len={len(content_acc)}")
        if iter_called_tool:
            log.info(f"[stream] tool was called, continuing outer loop")
            continue

        if not in_tc:
            log.info(f"[stream] no tool call this iteration, yielding stop + usage")
            yield sse({"type": "citations", "citations": [
                {"source": c.source, "page": c.page, "content": c.content, "url": c.url} for c in grounded_citations(candidate_citations, "".join(resp_parts))
            ]})
            yield content_chunk("", finish_reason or "stop")
            yield sse({"type": "usage", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        if not in_tc:
            break
    else:
        log.info(f"[stream] outer loop exhausted (max iterations)")
        yield sse({"type": "citations", "citations": [
            {"source": c.source, "page": c.page, "content": c.content, "url": c.url} for c in grounded_citations(candidate_citations, "".join(resp_parts))
        ]})
        yield content_chunk("", "stop")

    grounded = grounded_citations(candidate_citations, "".join(resp_parts))
    yield sse({"type": "citations", "citations": [
        {"source": c.source, "page": c.page, "content": c.content, "url": c.url} for c in grounded
    ]})


def non_stream_chat(req: ChatRequest):
    query, candidate_citations, do_web_search, use_fc, messages = build_messages_and_context(req)

    kwargs = {
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.7,
        "max_tokens": min(req.max_tokens or 1024, 2048),
    }
    if use_fc and state.web_search_enabled:
        kwargs["tools"] = [WEB_SEARCH_TOOL]

    for _ in range(4 if use_fc else 1):
        resp = state.llm_instance.create_chat_completion(**kwargs)
        answer = resp["choices"][0]["message"].get("content", "")

        if use_fc and "<tool_call>" in answer:
            tcd = parse_qwen_tool_call(answer)
            if tcd:
                _execute_and_feed(kwargs, "", tcd, query, candidate_citations)
                kwargs.pop("tools", None)
                continue

        if not req.enable_thinking:
            answer = strip_thinking(answer)

        now = int(time.time())
        return ChatResponse(
            id=f"chatcmpl-{now}", created=now, model=req.model,
            choices=[{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            citations=grounded_citations(candidate_citations, answer),
        )

    now = int(time.time())
    return ChatResponse(
        id=f"chatcmpl-{now}", created=now, model=req.model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        citations=[],
    )
