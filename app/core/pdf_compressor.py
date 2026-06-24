import asyncio
import os
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

async def summarize_chunk(chunk_text: str, chunk_index: int) -> str:
    """The MAP step: Summarizes a single document chunk."""
    prompt = (
        "You are an expert data extraction AI.\n"
        "Analyze the following document fragment and extract the core concepts, "
        "critical metrics, proper nouns, and key decisions.\n\n"
        f"Document Fragment:\n{chunk_text}"
    )
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=600
        )
        # Prefix the summary so we know its order when reconstructing
        return f"--- Summary of Part {chunk_index + 1} ---\n{response.choices[0].message.content.strip()}"
    except Exception as e:
        return f"--- Summary of Part {chunk_index + 1} ---\n[Extraction Error: {str(e)}]"

async def run_map_reduce_pipeline(chunks: list[str]) -> str:
    """Orchestrates the concurrent Map phase and the final Reduce phase."""
    
    # ==========================================
    # 1. THE MAP PHASE (Concurrent Execution)
    # ==========================================
    # Create a list of async tasks
    tasks = [summarize_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    
    # Run all API calls simultaneously
    chunk_summaries = await asyncio.gather(*tasks)
    
    # Combine all individual summaries into one massive text block
    combined_summaries = "\n\n".join(chunk_summaries)
    
    # ==========================================
    # 2. THE REDUCE PHASE (Final Distillation)
    # ==========================================
    reduce_prompt = (
        "You are a principal executive assistant. I am providing you with "
        "sequential summaries of a larger document.\n\n"
        "Synthesize these fragments into a single, cohesive Executive Summary. "
        "You MUST retain specific hard data, numerical metrics, tool names, and critical conclusions.\n\n"
        f"Sequential Summaries:\n{combined_summaries}"
    )
    
    final_response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a professional document synthesizer."},
            {"role": "user", "content": reduce_prompt}
        ],
        temperature=0.2, # Slight temperature increase for better narrative flow
        max_tokens=1500
    )
    
    return final_response.choices[0].message.content.strip()