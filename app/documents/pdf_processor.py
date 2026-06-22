import os
import io
import math
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
import tiktoken

# Safely catch and support common open-source PDF parsing engines
try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf # type: ignore
    except ImportError:
        pypdf = None

# Initialize the shared free inference client (Groq)
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Initialize tokenizer for mathematical tracking
tokenizer = tiktoken.get_encoding("cl100k_base")

# Inside app/documents/pdf_processor.py

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts raw text strings from binary PDF data streams.
    """
    if pypdf is None:
        return "[Error: 'pypdf' is not installed in your venv.]"
    
    try:
        file_stream = io.BytesIO(pdf_bytes)
        # Strictly use PdfReader for modern pypdf compatibility
        reader = pypdf.PdfReader(file_stream)
        text_slices = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_slices.append(page_text)
                
        return "\n".join(text_slices)
    except Exception as e:
        return f"[Error parsing PDF stream payload: {str(e)}]"

def calculate_token_len(text: str) -> int:
    return len(tokenizer.encode(text))

def chunk_text_by_tokens(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Slices large documents into overlapping token buckets."""
    tokens = tokenizer.encode(text)
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break
            
    return chunks

async def summarize_chunk(chunk_text: str, chunk_target_tokens: int) -> str:
    """Summarizes an individual chunk using a dynamically scaled token budget."""
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a dense technical note-taking engine. Summarize the text using verbatim facts only."},
                {"role": "user", "content": f"Compress this section concisely into roughly {chunk_target_tokens} tokens:\n\n{chunk_text}"}
            ],
            temperature=0.1,
            max_tokens=max(50, chunk_target_tokens + 50)  # Slipped padding avoids truncation cuts
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Chunk processing error: {str(e)}]"

async def execute_map_reduce_summary(large_text: str) -> Dict[str, Any]:
    """
    Executes a dynamically scaled Map-Reduce workflow to enforce a 70% reduction matrix.
    Named back to 'execute_map_reduce_summary' to fix the application routing error.
    """
    # 1. Gauge initial scale
    original_tokens = calculate_token_len(large_text)
    
    # 2. Establish targets based on success criteria (70% reduction = 30% retention budget)
    target_total_output = int(original_tokens * 0.30)
    
    # Safety ceiling: Groq/Llama-3.1 output token generation cap protection
    final_max_tokens_budget = min(target_total_output, 4000)
    
    # Fallback to prevent processing errors on very small files
    if original_tokens < 1500:
        return {
            "original_token_count": original_tokens,
            "compressed_token_count": original_tokens,
            "compression_ratio": 0.0,
            "was_compressed": False,
            "text": large_text
        }

    # 3. Dynamic Chunking Strategy Configuration
    chunks = chunk_text_by_tokens(large_text, chunk_size=1200, overlap=150)
    num_chunks = len(chunks)
    chunk_target = max(40, int(final_max_tokens_budget / num_chunks))

    # 4. The MAP Phase: Process all chunks concurrently using async routines
    tasks = [summarize_chunk(chunk, chunk_target) for chunk in chunks]
    intermediate_summaries = await asyncio.gather(*tasks)
    
    map_combined_text = "\n\n".join(intermediate_summaries)
    
    # 5. The REDUCE Phase: Consolidate intermediate notes into the final master compression text
    reduce_prompt = (
        "You are a master document context optimizer.\n"
        "Consolidate the following intermediate chunk notes into a coherent, highly informative master summary.\n"
        f"CRITICAL BUDGET: Your output must be comprehensive and target roughly {final_max_tokens_budget} tokens.\n"
        "Do not leave out core metrics, names, configurations, or technical definitions.\n\n"
        f"INTERMEDIATE NOTES:\n{map_combined_text}"
    )

    reduce_response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You compress technical knowledge cleanly without losing architectural info."},
            {"role": "user", "content": reduce_prompt}
        ],
        temperature=0.2,
        max_tokens=final_max_tokens_budget
    )
    
    final_compressed_text = reduce_response.choices[0].message.content.strip()
    compressed_tokens = calculate_token_len(final_compressed_text)
    
    # Calculate real mathematical savings matrix
    compression_ratio = round((1 - (compressed_tokens / original_tokens)) * 100, 2)

    return {
        "original_token_count": original_tokens,
        "compressed_token_count": compressed_tokens,
        "compression_ratio": compression_ratio,
        "was_compressed": True,
        "text": final_compressed_text
    }