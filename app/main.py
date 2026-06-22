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
    """
    Accepts a PDF file upload, extracts the text, and runs it through 
    the Map-Reduce compression pipeline.
    """
    # 1. Validate file extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF.")
        
    try:
        # 2. Read the binary file stream from the HTTP request
        pdf_bytes = await file.read()
        
        # 3. Extract the raw text using your existing helper function
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        # Check if extraction failed or returned your custom error string
        if raw_text.startswith("[Error"):
            raise HTTPException(status_code=500, detail=raw_text)
            
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="The PDF contains no extractable text.")
            
        # 4. Pass the text into your Map-Reduce engine
        compression_result = await execute_map_reduce_summary(raw_text)
        
        # 5. Return the final metrics and master summary to the user
        return {
            "success": True,
            "filename": file.filename,
            "original_token_count": compression_result["original_token_count"],
            "compressed_token_count": compression_result["compressed_token_count"],
            "compression_ratio": compression_result["compression_ratio"],
            "master_summary": compression_result["text"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing failed: {str(e)}")
    
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