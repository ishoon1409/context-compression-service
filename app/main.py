import os
import shutil
import json
import time
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, Form, APIRouter, HTTPException, UploadFile, File, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --- Internal Core Imports ---
from app.routers import openai_proxy
from app.routers import pdf_proxy
from app.core.mcp_compressor import compress_mcp_output
from app.core.mcp_client import fetch_and_compress_mcp
from app.core.image_compressor import compress_image_semantic
from app.core.tokenizer import calculate_tokens
from app.core.compressor import generate_rolling_summary
from app.documents.pdf_processor import extract_text_from_pdf_bytes, execute_map_reduce_summary
from app.documents.nlp_processor import extractive_summarize
from app.core.pdf_compressor import evaluate_pdf_compression
from app.evaluate_differential import run_system_benchmark
from app.core.rag_engine import build_rag_knowledge_base, query_rag_knowledge_base
from app.core.evaluator import run_differential_evaluation
from app.routers import rag_proxy
from app.routers import image_proxy
from app.routers import mcp_proxy


app = FastAPI(
    title="Context Compression Service",
    description="High-performance middleware service designed to dynamically optimize LLM payload sizes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf_proxy.router)
app.include_router(openai_proxy.router)
app.include_router(rag_proxy.router)
app.include_router(image_proxy.router)
app.include_router(mcp_proxy.router)


class ChatMessage(BaseModel):
    role: str
    content: str

class RAGQuery(BaseModel):
    query: str

class ChatCompressionRequest(BaseModel):
    messages: List[ChatMessage]
    model_name: str = "gpt-4"
    trigger_threshold: int = Field(2000, description="Token trigger bounds execution limit")
    tail_messages_to_keep: int = Field(4, description="Messages protected against summary extraction")

class ChatCompressionResponse(BaseModel):
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    was_compressed: bool
    messages: List[ChatMessage]

class DocumentCompressionResponse(BaseModel):
    original_estimated_tokens: int
    compressed_summary_tokens: int
    compression_ratio: float
    summary: str

class EvaluationRequest(BaseModel):
    chat_history: List[Dict[str, str]]
    new_question: str


@app.get("/")
async def root():
    return {"message": "Context Compression Service is running. Point your chat app to /v1/chat/completions"}

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the clean unified user interface."""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return """<h1>Dashboard file not found.</h1><p>Please ensure index.html exists in the frontend folder.</p>"""
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/compress/image")
async def image_endpoint(file: UploadFile = File(...)):
    """Unified Image Compression Endpoint"""
    try:
        image_bytes = await file.read()

        # Route to semantic Vision pipeline
        result = await compress_image_semantic(image_bytes)

        # Guardrail: The frontend specifically expects `data.semantic_caption`
        # If your function returned a plain string, we wrap it in a dict here!
        if isinstance(result, str):
            return {"semantic_caption": result}

        return result
    except Exception as e:
        print(f"Image Pipeline Crash: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress/mcp")
async def mcp_compress_endpoint(request: Request):
    try:
        payload = await request.json()
        raw_json = payload.get("raw_json", "")
        tool_name = payload.get("tool_name", "generic")

        if not raw_json:
            raise HTTPException(status_code=400, detail="raw_json is required")

        result = await compress_mcp_output(raw_json, tool_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress/mcp-external")
async def mcp_external_endpoint(request: Request):
    try:
        payload = await request.json()
        mcp_url = payload.get("mcp_url")
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments", {})

        if not mcp_url or not tool_name:
            raise HTTPException(status_code=400, detail="mcp_url and tool_name are required")

        result = await fetch_and_compress_mcp(mcp_url, tool_name, arguments)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress/pdf", response_model=DocumentCompressionResponse)
async def compress_pdf_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    try:
        pdf_bytes = await file.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)

        if raw_text.startswith("[Error"):
            raise HTTPException(status_code=500, detail=raw_text)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="The PDF contains no extractable text.")

        compression_result = await execute_map_reduce_summary(raw_text)

        return DocumentCompressionResponse(
            original_estimated_tokens=compression_result["original_token_count"],
            compressed_summary_tokens=compression_result["compressed_token_count"],
            compression_ratio=compression_result["compression_ratio"],
            summary=compression_result["text"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing failed: {str(e)}")

@app.post("/compress/pdf/local", response_model=DocumentCompressionResponse)
async def compress_pdf_local_nlp(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    try:
        pdf_bytes = await file.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)

        if raw_text.startswith("[Error"):
            raise HTTPException(status_code=500, detail=raw_text)

        compression_result = extractive_summarize(raw_text, target_tokens=1500)

        return DocumentCompressionResponse(
            original_estimated_tokens=compression_result["original_token_count"],
            compressed_summary_tokens=compression_result["compressed_token_count"],
            compression_ratio=compression_result["compression_ratio"],
            summary=compression_result["text"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local NLP Processing failed: {str(e)}")

@app.post("/compress/chat", response_model=ChatCompressionResponse)
async def compress_chat_context(payload: ChatCompressionRequest):
    raw_dicts = [{"role": m.role, "content": m.content} for m in payload.messages]
    total_tokens = calculate_tokens(raw_dicts, payload.model_name)

    if total_tokens <= payload.trigger_threshold:
        return ChatCompressionResponse(
            original_token_count=total_tokens,
            compressed_token_count=total_tokens,
            compression_ratio=0.0,
            was_compressed=False,
            messages=payload.messages
        )

    system_messages = [m for m in raw_dicts if m["role"] == "system"]
    non_system_messages = [m for m in raw_dicts if m["role"] != "system"]

    tail_size = min(payload.tail_messages_to_keep, len(non_system_messages))
    if tail_size == 0:
        raise HTTPException(status_code=400, detail="Insufficient conversation sequence depth to partition safely.")

    historical_head = non_system_messages[:-tail_size]
    verbatim_tail = non_system_messages[-tail_size:]

    existing_summary = None
    for msg in system_messages:
        if "--- CONVERSATION SUMMARY STATE ---" in msg["content"]:
            existing_summary = msg["content"]
            system_messages.remove(msg)

    new_summary_text = await generate_rolling_summary(historical_head, existing_summary)

    summary_message_dict = {
        "role": "system",
        "content": f"--- CONVERSATION SUMMARY STATE ---\n{new_summary_text}\n--- END OF STATE ---"
    }

    reconstructed_raw = system_messages + [summary_message_dict] + verbatim_tail
    compressed_tokens = calculate_tokens(reconstructed_raw, payload.model_name)
    savings_ratio = round((1.0 - (compressed_tokens / total_tokens)) * 100, 2)

    return ChatCompressionResponse(
        original_token_count=total_tokens,
        compressed_token_count=compressed_tokens,
        compression_ratio=savings_ratio,
        was_compressed=True,
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in reconstructed_raw]
    )

@app.post("/evaluate/chat")
async def evaluate_chat_endpoint(request: EvaluationRequest):
    try:
        results = await run_differential_evaluation(request.chat_history, request.new_question)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/pdf")
async def evaluate_pdf_endpoint(
    question: str = Form(...),
    summary: str = Form(...),
    file: UploadFile = File(...)
):
    temp_file_path = f"eval_temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await evaluate_pdf_compression(temp_file_path, summary, question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.websocket("/ws/benchmark")
async def websocket_benchmark(websocket: WebSocket):
    await websocket.accept()
    async def ws_logger(msg: str):
        try:
            await websocket.send_text(msg)
        except Exception:
            pass
    try:
        await run_system_benchmark(logger=ws_logger)
        await websocket.send_text("\n✅ [SYSTEM] CI/CD Benchmark Pipeline Completed.")
    except Exception as e:
        await websocket.send_text(f"\n❌ [SYSTEM ERROR] {str(e)}")
    finally:
        await websocket.close()

@app.post("/rag/ingest")
async def rag_ingest_endpoint(
    kb_name: str = Form("master_knowledge_base"),
    file: UploadFile = File(...)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDFs supported.")
    temp_path = f"rag_temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = await build_rag_knowledge_base(temp_path, kb_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/rag/query")
async def rag_endpoint(req: RAGQuery):
    """Cleaned Endpoint triggered by the UI to query ChromaDB and the LLM"""
    try:
        result = await query_rag_knowledge_base(req.query)
        return {"answer": result}
    except Exception as e:
        # Pushing the error explicitly to help debugging in the terminal
        print(f"RAG Query Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))