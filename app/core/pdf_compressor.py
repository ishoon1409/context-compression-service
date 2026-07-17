import os
import asyncio
import re
import json
import tiktoken
import pymupdf4llm
import openai
from openai import AsyncOpenAI

# Point the OpenAI client to Groq's high-speed endpoint
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

async def call_llm_with_backoff(messages: list, response_format=None, max_retries: int = 6) -> str:
    """
    A robust network wrapper that intercepts RateLimitErrors (429) from Groq 
    and automatically pauses the script until the API bucket refills.
    """
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "temperature": 0.0
            }
            if response_format:
                kwargs["response_format"] = response_format
                
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                sleep_time = 10 * (2 ** attempt)
                print(f"⚠️ [Rate Limit] Groq TPM bucket empty. Sleeping for {sleep_time}s...")
                await asyncio.sleep(sleep_time)
            else:
                raise e

def semantic_chunk_markdown(text: str, max_tokens: int = 2500) -> list:
    """
    Intelligently splits Markdown text based on headers (## or ###) 
    instead of blindly cutting words in half.
    """
    sections = re.split(r'(?=\n##\s|\n###\s)', text)
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for section in sections:
        section_tokens = len(tokenizer.encode(section))
        
        if section_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
                
            words = section.split(" ")
            temp_chunk = ""
            temp_tokens = 0
            for word in words:
                word_tokens = len(tokenizer.encode(word + " "))
                if temp_tokens + word_tokens > max_tokens:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word + " "
                    temp_tokens = word_tokens
                else:
                    temp_chunk += word + " "
                    temp_tokens += word_tokens
            if temp_chunk:
                chunks.append(temp_chunk.strip())
            continue
            
        if current_tokens + section_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = section
            current_tokens = section_tokens
        else:
            current_chunk += section
            current_tokens += section_tokens
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c.strip()]

async def summarize_chunk(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    """
    The MAP Phase: Summarizes a single, semantic section of the document.
    """
    prompt = (
        f"You are an expert technical compression engine. This is section {chunk_index}/{total_chunks} of a larger document.\n"
        "Extract the core concepts, hard metrics, architectures, and entities from this section.\n"
        "Ignore conversational filler. Output in highly dense Markdown.\n\n"
        f"SECTION TEXT:\n{chunk_text}"
    )
    
    messages = [
        {"role": "system", "content": "You are a precise data extraction API."},
        {"role": "user", "content": prompt}
    ]
    return await call_llm_with_backoff(messages)

async def compress_pdf(pdf_path: str) -> dict:
    """
    The main Map-Reduce orchestration pipeline.
    """
    try:
        # 1. Convert visual PDF layout directly into clean Markdown using PyMuPDF
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # 2. Intelligently chunk the markdown by headers
        chunks = semantic_chunk_markdown(md_text, max_tokens=2500)
        
        # 3. Map Phase: Summarize every chunk SEQUENTIALLY to avoid TPM limit crashes
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"🔄 Processing chunk {i+1} of {len(chunks)}...")
            summary = await summarize_chunk(chunk, i+1, len(chunks))
            chunk_summaries.append(summary)
            await asyncio.sleep(2) 
        
        # 4. Reduce Phase: Combine the chunk summaries into one final master state
        combined_text = "\n\n--- NEXT SECTION ---\n\n".join(chunk_summaries)
        
        reduce_prompt = (
            "You are a master technical compression engine.\n"
            "I am providing you with several chronological summaries representing individual sections of a complete PDF document.\n"
            "Synthesize them into a single, cohesive, highly dense Markdown executive summary.\n"
            "CRITICAL INSTRUCTION: You MUST preserve all hard numerical values, proper nouns, architectures, and metrics.\n\n"
            f"COMBINED SUMMARIES:\n{combined_text}"
        )
        
        final_summary = await call_llm_with_backoff([
            {"role": "system", "content": "You are a structured Markdown summarizer."},
            {"role": "user", "content": reduce_prompt}
        ])
        
        # 5. Calculate Metrics
        tokenizer = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(tokenizer.encode(md_text))
        compressed_tokens = len(tokenizer.encode(final_summary))
        ratio = 0 if original_tokens == 0 else round((1 - (compressed_tokens / original_tokens)) * 100)
        
        return {
            "summary": final_summary,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": ratio
        }
    except Exception as e:
        raise Exception(f"PDF Compression failed: {str(e)}")

async def evaluate_pdf_compression(pdf_path: str, compressed_summary: str, question: str) -> dict:
    """
    Runs the differential evaluation pipeline for a PDF document.
    Path A: Answers using the FULL Original PDF text.
    Path B: Answers using ONLY the compressed summary.
    """
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        encoded_md = tokenizer.encode(md_text)
        MAX_EVAL_TOKENS = 3000
        if len(encoded_md) > MAX_EVAL_TOKENS:
            md_text = tokenizer.decode(encoded_md[:MAX_EVAL_TOKENS]) + "\n\n...[TRUNCATED TO FIT CONTEXT LIMIT]..."
            
        path_a_prompt = (
            "You are an expert technical assistant. Use the following complete original document to answer the user's question.\n\n"
            f"ORIGINAL DOCUMENT:\n{md_text}\n\n"
            f"QUESTION: {question}"
        )
        answer_a = await call_llm_with_backoff([{"role": "user", "content": path_a_prompt}])

        path_b_prompt = (
            "You are an expert technical assistant. Use the following compressed executive summary to answer the user's question.\n\n"
            f"COMPRESSED SUMMARY:\n{compressed_summary}\n\n"
            f"QUESTION: {question}"
        )
        answer_b = await call_llm_with_backoff([{"role": "user", "content": path_b_prompt}])

        scoring_rules = (
            "1. Start at a perfect 100 points.\n"
            "2. First, determine the nature of the user's question ('Factual' vs 'Conceptual').\n"
            "3. If the question is Factual (asking for specific metrics, names, configurations):\n"
            "   - Deduct exactly 14 points for EVERY specific numerical metric missing or altered in Answer B.\n"
            "   - Deduct exactly 7 points for EVERY system identifier, proper noun, or method name missing in Answer B.\n"
            "4. If the question is Conceptual (asking for summaries, rationale, or semantic themes):\n"
            "   - Deduct exactly 15 points if the primary core takeaway or overarching narrative is completely missing in Answer B.\n"
            "   - Deduct exactly 7 points for each missing secondary supporting argument, concept, or timeline milestone.\n"
            "5. Deduct exactly 3 points for minor stylistic or contextual omissions."
        )

        judge_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a strict, mathematical AI Judge grading a document compression engine. "
                    "Compare Answer B against the Gold Standard Answer A.\n\n"
                    "### SCORING RUBRIC:\n"
                    f"{scoring_rules}\n\n"
                    "You MUST respond in valid JSON containing exactly two keys: 'reasoning' and 'score'.\n\n"
                    "EXAMPLE OUTPUT FORMAT:\n"
                    "{\n"
                    "  \"reasoning\": \"Missing specific identifier (-7) but retained overall concept. 100 - 7 = 93.\",\n"
                    "  \"score\": 93\n"
                    "}"
                )
            },
            {
                "role": "user",
                "content": f"Answer A (Gold Standard):\n{answer_a}\n\nAnswer B (Compressed Context):\n{answer_b}"
            }
        ]

        score_response = await call_llm_with_backoff(judge_prompt, response_format={"type": "json_object"})
        
        try:
            judge_data = json.loads(score_response)
            raw_score_str = str(judge_data.get("score", judge_data.get("Score", "0")))
            score_match = re.search(r'\d+', raw_score_str)
            score = int(score_match.group()) if score_match else 0
            reasoning = judge_data.get("reasoning", judge_data.get("Reasoning", "Failed to parse reasoning."))
        except Exception as e:
            print(f"Failed to parse JSON judge output: {e}")
            score = 50
            reasoning = f"JSON parsing error occurred during AI judgement. Raw output: {score_response}"

        return {
            "path_a_answer": answer_a,
            "path_b_answer": answer_b,
            "judge_score": score,
            "judge_reasoning": reasoning
        }
    except Exception as e:
        raise Exception(f"Evaluation failed: {str(e)}")