import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import tiktoken
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from app.core.compressor import generate_rolling_summary
from app.core.tokenizer import calculate_tokens

load_dotenv()
router = APIRouter()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# --- Whole-conversation compression config ---
SUMMARY_MARKER = "--- CONVERSATION SUMMARY STATE ---"
SUMMARY_MARKER_END = "--- END OF STATE ---"

MODEL_CONTEXT_LIMITS = {
    # NOTE: This reflects your Groq account's actual per-request TPM cap (6000),
    # not the model's theoretical context window (which is much larger). The TPM
    # cap is the real binding constraint that causes 413s, so we trigger off that.
    "llama-3.1-8b-instant": 6000,
}
DEFAULT_CONTEXT_LIMIT = int(os.getenv("DEFAULT_CONTEXT_LIMIT", "6000"))
COMPRESSION_TRIGGER_RATIO = float(os.getenv("COMPRESSION_TRIGGER_RATIO", "0.6"))
TAIL_MESSAGES_TO_KEEP = int(os.getenv("TAIL_MESSAGES_TO_KEEP", "4"))
PER_MESSAGE_BLOAT_THRESHOLD = int(os.getenv("PER_MESSAGE_BLOAT_THRESHOLD", "1500"))

# --- Session-keyed summary cache (in-memory; swap for Redis if you scale to multiple workers) ---
# shape: { session_key: {"summary": str, "covered_count": int} }
SESSION_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}


class ChatMessage(BaseModel):
    role: str
    # Accept both plain string content and content-part arrays
    # (LibreChat sends arrays for some flows, e.g. Upload as Text / vision messages)
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


def normalize_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Flattens list-format content (content parts) into a plain string for tokenizing/compression."""
    if isinstance(content, str):
        return content
    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
    return " ".join(text_parts).strip()


MAX_CHUNK_TOKENS = 3000  # keep each Groq compression call safely under the TPM limit


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = MAX_CHUNK_TOKENS) -> list:
    """Splits text into token-bounded chunks so no single Groq call exceeds the rate limit."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    return [tokenizer.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


async def fast_text_compressor(text: str, tokenizer=None) -> str:
    prompt = (
        "You are an active compression middleware. Summarize the following bloated text "
        "into an ultra-dense, factual Markdown format. Retain ALL names, numbers, code, and core concepts. "
        "Remove all conversational filler. DO NOT output any thinking steps."
    )

    tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
    chunks = chunk_text_by_tokens(text, tokenizer)

    if len(chunks) == 1:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunks[0]}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

    print(f"[PROXY] Message too large for a single call - compressing in {len(chunks)} chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[PROXY]   Compressing chunk {i + 1}/{len(chunks)}...")
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.0
        )
        chunk_summaries.append(response.choices[0].message.content)

    return "\n\n".join(chunk_summaries)


def get_session_key(raw_messages: List[Dict[str, str]]) -> Optional[str]:
    """
    LibreChat doesn't send a conversation ID to custom endpoints, but it always
    replays the conversation from message #1. We fingerprint the first couple
    of non-system messages (stable/append-only across turns) as a pseudo session key.
    """
    non_system = [m for m in raw_messages if m["role"] != "system"]
    if not non_system:
        return None
    fingerprint_source = non_system[:2]
    raw = "|".join(f"{m['role']}:{m['content']}" for m in fingerprint_source)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def compress_conversation_if_needed(
    raw_messages: List[Dict[str, str]],
    model_name: str
) -> tuple[List[Dict[str, str]], Optional[Dict[str, int]]]:
    """
    Checks total conversation tokens against COMPRESSION_TRIGGER_RATIO of the model's
    (Groq TPM) limit. If exceeded, partitions into system / historical-head / verbatim-tail,
    incrementally summarizes only the NEW portion of the head since the last
    compression (via a session-keyed cache), merges it with any cached prior
    summary, and reconstructs a compact message list.

    Returns (messages_to_use, conversation_compression_stats | None)
    """
    context_limit = MODEL_CONTEXT_LIMITS.get(model_name, DEFAULT_CONTEXT_LIMIT)
    trigger_threshold = int(context_limit * COMPRESSION_TRIGGER_RATIO)

    total_tokens = calculate_tokens(raw_messages, model_name)
    print(f"[PROXY] Conversation token check: {total_tokens} / trigger={trigger_threshold} (limit={context_limit})")

    if total_tokens <= trigger_threshold:
        return raw_messages, None

    session_key = get_session_key(raw_messages)
    cached = SESSION_SUMMARY_CACHE.get(session_key) if session_key else None

    system_messages = [m for m in raw_messages if m["role"] == "system"]
    non_system_messages = [m for m in raw_messages if m["role"] != "system"]

    tail_size = min(TAIL_MESSAGES_TO_KEEP, len(non_system_messages))
    if tail_size == 0 or tail_size == len(non_system_messages):
        # Nothing safe to summarize (conversation too short) — fall through untouched.
        return raw_messages, None

    historical_head = non_system_messages[:-tail_size]
    verbatim_tail = non_system_messages[-tail_size:]

    # Strip any stray summary marker that might have leaked into system messages
    # (defensive — shouldn't normally happen since LibreChat doesn't resend ours).
    system_messages = [m for m in system_messages if SUMMARY_MARKER not in m["content"]]

    existing_summary = None
    to_summarize = historical_head

    if cached and cached["covered_count"] <= len(historical_head):
        existing_summary = cached["summary"]
        to_summarize = historical_head[cached["covered_count"]:]

        if not to_summarize:
            # Head hasn't grown since last compression — reuse cached summary verbatim, no LLM call.
            summary_message = {
                "role": "system",
                "content": f"{SUMMARY_MARKER}\n{existing_summary}\n{SUMMARY_MARKER_END}"
            }
            reconstructed = system_messages + [summary_message] + verbatim_tail
            compressed_tokens = calculate_tokens(reconstructed, model_name)
            return reconstructed, {
                "original_tokens": total_tokens,
                "new_tokens": compressed_tokens,
                "trigger_threshold": trigger_threshold,
                "context_limit": context_limit,
            }
    # else: no usable cache (first compression, cache miss, or edited/branched history)
    # — to_summarize stays as the full historical_head, existing_summary stays None.

    new_summary_text = await generate_rolling_summary(to_summarize, existing_summary)

    if session_key:
        SESSION_SUMMARY_CACHE[session_key] = {
            "summary": new_summary_text,
            "covered_count": len(historical_head),
        }

    summary_message = {
        "role": "system",
        "content": f"{SUMMARY_MARKER}\n{new_summary_text}\n{SUMMARY_MARKER_END}"
    }

    reconstructed = system_messages + [summary_message] + verbatim_tail
    compressed_tokens = calculate_tokens(reconstructed, model_name)

    stats = {
        "original_tokens": total_tokens,
        "new_tokens": compressed_tokens,
        "trigger_threshold": trigger_threshold,
        "context_limit": context_limit,
    }
    return reconstructed, stats


def build_compression_summary(
    per_message_stats: list,
    conversation_stats: Optional[Dict[str, int]]
) -> Optional[str]:
    """Builds a markdown status block covering both whole-conversation and per-message compression."""
    compressed_msgs = [s for s in per_message_stats if s["compressed"]]
    if not compressed_msgs and not conversation_stats:
        return None

    lines = []

    if conversation_stats:
        before = conversation_stats["original_tokens"]
        after = conversation_stats["new_tokens"]
        saved_pct = round((1 - after / before) * 100, 1) if before else 0
        lines.append(
            f"🗜️ **Auto Context Compression** — conversation hit "
            f"{before:,}/{conversation_stats['context_limit']:,} tokens "
            f"(>{int(COMPRESSION_TRIGGER_RATIO * 100)}% threshold). "
            f"Older turns summarized: {before:,} → {after:,} tokens (**{saved_pct}%** saved)"
        )

    if compressed_msgs:
        total_before = sum(s["original_tokens"] for s in compressed_msgs)
        total_after = sum(s["new_tokens"] for s in compressed_msgs)
        saved_pct = round((1 - total_after / total_before) * 100, 1) if total_before else 0
        lines.append(
            f"🗜️ **Bloated Message Compression** — {len(compressed_msgs)} message(s), "
            f"{total_before:,} → {total_after:,} tokens (**{saved_pct}%** saved)"
        )
        for s in compressed_msgs:
            lines.append(f"  - `{s['role']}`: {s['original_tokens']:,} → {s['new_tokens']:,} tokens")

    return "\n".join(lines) + "\n\n---\n\n"


@router.post("/v1/chat/completions")
async def proxy_chat_completions(req: ChatCompletionRequest):
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Step 1: normalize incoming messages to plain dicts
        raw_messages = [
            {"role": msg.role, "content": normalize_content(msg.content)}
            for msg in req.messages
        ]

        # Step 2: whole-conversation threshold check + incremental rolling-summary compression
        working_messages, conversation_stats = await compress_conversation_if_needed(
            raw_messages, req.model
        )
        if conversation_stats:
            print(
                f"[PROXY] Conversation compressed: "
                f"{conversation_stats['original_tokens']} → {conversation_stats['new_tokens']} tokens"
            )

        # Step 3: secondary safety net — per-message bloat check on whatever remains
        # (covers a single oversized message hiding inside the protected tail, etc.)
        optimized_messages = []
        per_message_stats = []
        for msg in working_messages:
            content_text = msg["content"]
            token_count = len(tokenizer.encode(content_text))

            if token_count > PER_MESSAGE_BLOAT_THRESHOLD:
                print(f"[PROXY] Intercepted bloated message ({token_count} tokens). Compressing...")
                compressed_content = await fast_text_compressor(content_text, tokenizer)
                new_token_count = len(tokenizer.encode(compressed_content))
                per_message_stats.append({
                    "role": msg["role"],
                    "original_tokens": token_count,
                    "new_tokens": new_token_count,
                    "compressed": True
                })
                optimized_messages.append({"role": msg["role"], "content": compressed_content})
            else:
                per_message_stats.append({
                    "role": msg["role"],
                    "original_tokens": token_count,
                    "new_tokens": token_count,
                    "compressed": False
                })
                optimized_messages.append({"role": msg["role"], "content": content_text})

        # Step 4: final hard safety net — if, even after all compression, the payload
        # still exceeds the Groq TPM cap (e.g. tokenizer mismatch undercounted), abort
        # with a clear error instead of letting Groq reject it with a raw 413.
        final_total = sum(len(tokenizer.encode(m["content"])) for m in optimized_messages)
        hard_cap = MODEL_CONTEXT_LIMITS.get(req.model, DEFAULT_CONTEXT_LIMIT)
        if final_total > hard_cap:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Even after compression, the payload is {final_total} tokens, "
                    f"still over the {hard_cap}-token limit. Try starting a new conversation "
                    f"or shortening your message."
                )
            )

        summary = build_compression_summary(per_message_stats, conversation_stats)
        print(f"[PROXY] Forwarding optimized payload to Groq ({final_total} tokens)...")

        if not req.stream:
            response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=optimized_messages,
                temperature=req.temperature,
                stream=False
            )
            data = response.model_dump()
            if summary:
                data["choices"][0]["message"]["content"] = summary + data["choices"][0]["message"]["content"]
            return data

        async def event_stream():
            try:
                chunk_id = f"chatcmpl-{uuid.uuid4()}"

                if summary:
                    lead_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": summary},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(lead_chunk)}\n\n"

                stream = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=optimized_messages,
                    temperature=req.temperature,
                    stream=True
                )
                async for chunk in stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Stream Error: {str(e)}")
                error_chunk = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Proxy Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))