import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv
# ADD THIS NEW LINE:
load_dotenv() # Load variables from .env file

from app.core.tokenizer import calculate_tokens
from app.core.compressor import generate_rolling_summary
from app.documents.pdf_processor import extract_text_from_pdf_bytes, execute_map_reduce_summary
from app.documents.nlp_processor import extractive_summarize
from app.core.document_processor import extract_text_from_pdf, chunk_text_by_tokens
from app.core.pdf_compressor import run_map_reduce_pipeline
import tiktoken
from app.core.evaluator import run_differential_evaluation

from fastapi.responses import HTMLResponse
import os

app = FastAPI(
    title="Context Compression Service",
    description="High-performance middleware service designed to dynamically optimize LLM payload sizes.",
    version="1.0.0"
)

# --- API Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

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

# --- Endpoints ---
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

@app.post("/compress/pdf", response_model=DocumentCompressionResponse)
async def compress_pdf_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    try:
        pdf_bytes = await file.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        # Catch the extraction error before it hits the Map-Reduce engine
        if raw_text.startswith("[Error"):
            raise HTTPException(status_code=500, detail=raw_text)
            
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="The PDF contains no extractable text.")
            
        # Execute compression
        compression_result = await execute_map_reduce_summary(raw_text)
        
        # Map the dictionary keys to your Pydantic Response model explicitly
        return DocumentCompressionResponse(
            original_estimated_tokens=compression_result["original_token_count"],
            compressed_summary_tokens=compression_result["compressed_token_count"],
            compression_ratio=compression_result["compression_ratio"],
            summary=compression_result["text"]  # <-- This fixes the validation error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing failed: {str(e)}")

@app.post("/compress/pdf")
async def compress_pdf_endpoint(file: UploadFile = File(...)):
    """Accepts a PDF upload, chunks it, and runs a Map-Reduce summary pipeline."""
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. Read the file into memory
        file_bytes = await file.read()
        
        # 2. Extract raw text
        raw_text = extract_text_from_pdf(file_bytes)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF. It may be an image-based scan.")
            
        # 3. Calculate original token size for metrics
        encoding = tiktoken.get_encoding("cl100k_base")
        original_token_count = len(encoding.encode(raw_text))
        
        # 4. Chunk the text
        chunks = chunk_text_by_tokens(raw_text, chunk_size=2000, overlap=200)
        
        # 5. Run Map-Reduce Pipeline
        final_summary = await run_map_reduce_pipeline(chunks)
        
        # 6. Calculate compressed token size
        compressed_token_count = len(encoding.encode(final_summary))
        compression_ratio = round((1.0 - (compressed_token_count / original_token_count)) * 100, 2)
        
        return {
            "filename": file.filename,
            "original_token_count": original_token_count,
            "compressed_token_count": compressed_token_count,
            "compression_ratio": compression_ratio,
            "total_chunks_processed": len(chunks),
            "summary": final_summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing Error: {str(e)}")
    
@app.post("/compress/pdf/local", response_model=DocumentCompressionResponse)
async def compress_pdf_local_nlp(file: UploadFile = File(...)):
    """
    Zero-Cost Extractive Summarization Endpoint.
    Uses TF-IDF and PageRank instead of an LLM.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    try:
        # Extract the text using the exact same pypdf function from earlier
        pdf_bytes = await file.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        if raw_text.startswith("[Error"):
            raise HTTPException(status_code=500, detail=raw_text)
            
        # Execute the Local NLP Engine (Targeting a strict 1,500 token output limit)
        compression_result = extractive_summarize(raw_text, target_tokens=1500)
        
        return DocumentCompressionResponse(
            original_estimated_tokens=compression_result["original_token_count"],
            compressed_summary_tokens=compression_result["compressed_token_count"],
            compression_ratio=compression_result["compression_ratio"],
            summary=compression_result["text"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local NLP Processing failed: {str(e)}")
    
    
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the clean unified user interface directly on the root endpoint."""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    
    if not os.path.exists(frontend_path):
        return """<h1>Dashboard file not found.</h1><p>Please ensure index.html exists in the frontend folder.</p>"""
        
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()
    
@app.post("/evaluate/chat")
async def evaluate_chat_endpoint(request: EvaluationRequest):
    try:
        results = await run_differential_evaluation(request.chat_history, request.new_question)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))