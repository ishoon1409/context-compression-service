import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

# 1. Point the OpenAI client to Groq's free endpoint
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

async def generate_rolling_summary(historical_head: List[Dict[str, Any]], existing_summary: Optional[str] = None) -> str:
    """
    Consolidates the historical tail of a conversation into a unified,
    dense summary string using a highly efficient utility model.
    """
    formatted_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in historical_head])
    
    context_prefix = f"Existing Continuous Summary State:\n{existing_summary}\n\n" if existing_summary else ""
    
    # UPGRADED: Fierce Data Retention Prompt (Markdown Format)
    prompt = (
        "You are an advanced technical compression engine. "
        "Your task is to distill the chat history into a dense, structured Markdown summary.\n\n"
        "CRITICAL INSTRUCTION: You must extract and perfectly preserve all technical parameters, "
        "numerical metrics, proper nouns, system identifiers, and configurations.\n\n"
        "You MUST adhere strictly to this format. Do not use JSON. Use the exact markdown headers below:\n\n"
        "### 🚨 Core Issue\n"
        "[1-2 sentences explaining the primary problem or goal]\n\n"
        "### 💾 Critical State Variables\n"
        "* **Environment:** [OS, Frameworks, Cloud instances]\n"
        "* **Files Modified:** [List specific files]\n"
        "* **Hard Metrics:** [List specific numbers, IDs, ARNs, or parameters]\n\n"
        "### ✅ Resolution / Current State\n"
        "[1-2 sentences explaining exactly how the issue was resolved or what the final agreed-upon state is.]\n\n"
        f"{context_prefix}New conversation turns:\n{formatted_history}"
    )

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a highly precise Markdown extraction and compression API."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=800
        # The JSON response_format parameter has been completely removed
    )
    
    # Return the pure, beautifully formatted Markdown string directly to the frontend
    return response.choices[0].message.content.strip()