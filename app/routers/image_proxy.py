import time
import uuid
import json
import base64
from typing import List, Optional, Union, Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.core.image_compressor import compress_image_semantic

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    # LibreChat sends vision messages as a list of content parts
    # (text + image_url), or a plain string for text-only messages.
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


def extract_image_and_text(content: Union[str, List[Dict[str, Any]]]):
    """Pulls the first base64 image and any accompanying text out of a vision-format message."""
    if isinstance(content, str):
        return None, content

    image_bytes = None
    text_parts = []

    for part in content:
        part_type = part.get("type")
        if part_type == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:") and "base64," in url:
                b64_data = url.split("base64,", 1)[1]
                try:
                    image_bytes = base64.b64decode(b64_data)
                except Exception:
                    image_bytes = None
        elif part_type == "text":
            text_parts.append(part.get("text", ""))

    return image_bytes, " ".join(text_parts).strip()


def build_stats_line(original_tokens: Optional[int], compressed_tokens: Optional[int], ratio: Optional[int]) -> str:
    """Builds the visible compression banner shown above the caption."""
    if original_tokens is not None and compressed_tokens is not None:
        return (
            f"🗜️ **Image Compression** — {original_tokens:,} → {compressed_tokens:,} tokens "
            f"(**{ratio}%** saved)"
        )
    return "🗜️ **Image Compression** — converted to semantic caption:"


def build_response_payload(answer_text: str, model: str):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": "stop"
        }]
    }


@router.post("/image/v1/chat/completions")
async def image_chat_completions(req: ChatCompletionRequest):
    try:
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found.")

        image_bytes, accompanying_text = extract_image_and_text(user_messages[-1].content)

        if image_bytes is None:
            answer_text = (
                "Attach an image to this chat and I'll compress it into a dense semantic "
                "caption you can use as low-token context."
            )
        else:
            print("[IMAGE-PROXY] Compressing uploaded image...")
            result = await compress_image_semantic(image_bytes)

            if isinstance(result, dict):
                caption = result.get("semantic_caption", "")
                original_tokens = result.get("original_tokens")
                compressed_tokens = result.get("compressed_tokens")
                ratio = result.get("compression_ratio")
            else:
                caption = result
                original_tokens = compressed_tokens = ratio = None

            if not caption or not caption.strip():
                answer_text = (
                    "🗜️ **Image Compression** — the model returned an empty response. "
                    "Try again or use a smaller/clearer image."
                )
            else:
                stats_line = build_stats_line(original_tokens, compressed_tokens, ratio)
                answer_text = f"{stats_line}\n\n---\n\n{caption}"

        if not req.stream:
            return build_response_payload(answer_text, req.model)

        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4()}"
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": answer_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[IMAGE-PROXY] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))