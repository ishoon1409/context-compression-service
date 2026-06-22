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
    
    # UPGRADED: Fierce Data Retention Prompt
    prompt = (
        "You are an advanced technical compression engine. "
        "Your task is to compress the chat history into a dense STATE object.\n\n"
        "CRITICAL INSTRUCTION: You must extract and perfectly preserve all technical parameters, numerical metrics, proper nouns, system identifiers, and configurations.\n\n"
        "Return ONLY a valid JSON object with exactly two keys:\n"
        "1. 'entities': An array of strings containing every specific metric, ARN, identifier, and proper noun mentioned.\n"
        "2. 'narrative': A highly abbreviated summary of the events.\n\n"
        f"{context_prefix}New conversation turns:\n{formatted_history}"
    )

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a JSON-only data extraction and compression API."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=800, # Increased slightly to accommodate the JSON bracket structure
        response_format={"type": "json_object"} # <--- THE CRITICAL FIX
    )
    
    # Parse the JSON and format it into a dense string for the next turn
    import json
    try:
        data = json.loads(response.choices[0].message.content)
        entities_str = " | ".join(data.get("entities", []))
        narrative = data.get("narrative", "")
        return f"CRITICAL ENTITIES: {entities_str}\nNARRATIVE: {narrative}"
    except json.JSONDecodeError:
        return response.choices[0].message.content.strip()
    
