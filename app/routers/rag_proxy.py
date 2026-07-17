import time
import uuid
import json
from typing import List, Optional, Union, Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.core.rag_engine import query_rag_knowledge_base

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


def get_text_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    parts = [p.get("text", "") for p in content if p.get("type") == "text"]
    return " ".join(parts).strip()


@router.post("/rag/v1/chat/completions")
async def rag_chat_completions(req: ChatCompletionRequest):
    """
    Adapts LibreChat's chat request into a RAG query: takes the latest user
    message as the query, retrieves relevant chunks from ChromaDB, and returns
    the generated answer in OpenAI's chat completion shape.
    """
    try:
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found to use as RAG query.")

        query = get_text_content(user_messages[-1].content)
        print(f"[RAG-PROXY] Querying knowledge base with: {query[:80]}...")

        answer = await query_rag_knowledge_base(query)

        if not req.stream:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop"
                }]
            }

        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4()}"

            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": answer},
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
        print(f"[RAG-PROXY] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))