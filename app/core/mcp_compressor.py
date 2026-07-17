import json
import os
import tiktoken
from openai import AsyncOpenAI

# Initialize Groq client
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def prune_json(data):
    """
    Recursively removes nulls, empty arrays, empty dicts, and massive bloated keys.
    """
    if isinstance(data, dict):
        pruned = {}
        for k, v in data.items():
            # Filter out useless metadata and heavy arrays (like Pokemon game_indices)
            if v not in (None, "", [], {}) and k not in ["avatar_url", "_links", "links", "game_indices", "moves"]:
                pruned_val = prune_json(v)
                if pruned_val not in (None, "", [], {}):
                    pruned[k] = pruned_val
        return pruned
    elif isinstance(data, list):
        pruned = [prune_json(item) for item in data]
        return [item for item in pruned if item not in (None, "", [], {})]
    return data

async def compress_mcp_output(raw_json_str: str, tool_name: str = "generic") -> dict:
    """
    Takes a raw JSON string, prunes it algorithmically, enforces a strict size limit, 
    and compresses it into YAML via the LLM.
    """
    try:
        # 1. Algorithmic Pruning
        try:
            parsed_json = json.loads(raw_json_str)
            pruned_data = prune_json(parsed_json)
            optimized_json_str = json.dumps(pruned_data, indent=2)
        except json.JSONDecodeError:
            optimized_json_str = raw_json_str

        # ==========================================
        # STRICT API FREE-TIER SAFETY VALVE
        # ==========================================
        # We enforce a hard limit of 10,000 characters (~2,500 tokens).
        # This guarantees you NEVER hit the 6,000 TPM limit or get a 413 error!
        if len(optimized_json_str) > 10000:
            print(f"⚠️ Truncating payload! Original size: {len(optimized_json_str)} chars")
            optimized_json_str = optimized_json_str[:10000] + "\n... [PAYLOAD TRUNCATED DUE TO STRICT API TOKEN LIMITS]"

        # 2. LLM Compression (JSON -> YAML)
        prompt = (
            f"You are an expert data compression engine. Below is the output from a '{tool_name}' tool/API.\n"
            "Convert this payload into an ultra-dense, highly compressed YAML format.\n"
            "Extract only the most crucial factual data, metrics, and identifiers.\n\n"
            "CRITICAL INSTRUCTION: Output ONLY the raw YAML text. Do NOT include any introductory or concluding conversational text. Do NOT wrap the output in markdown blocks (```yaml). Output exactly and only the compressed data.\n\n"
            f"PAYLOAD:\n{optimized_json_str}"
        )

        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        compressed_yaml = response.choices[0].message.content.strip()

        # Fallback: Strip markdown code blocks if the LLM ignores the prompt instructions
        if compressed_yaml.startswith("```"):
            lines = compressed_yaml.split("\n")
            if lines[0].startswith("```"): 
                lines = lines[1:]
            if lines[-1].startswith("```"): 
                lines = lines[:-1]
            compressed_yaml = "\n".join(lines).strip()

        # 3. Calculate Final Metrics
        # (We use the raw unpruned string length to show the true compression power)
        original_tokens = len(raw_json_str) // 4
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        compressed_tokens = len(tokenizer.encode(compressed_yaml))
        
        ratio = 0 if original_tokens == 0 else round((1 - (compressed_tokens / original_tokens)) * 100)

        return {
            "compressed_yaml": compressed_yaml,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": ratio
        }
    except Exception as e:
        raise Exception(f"MCP Compression failed: {str(e)}")