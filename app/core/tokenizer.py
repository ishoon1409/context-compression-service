import tiktoken
from typing import List, Dict, Any

def calculate_tokens(messages: List[Dict[str, Any]], model_name: str = "gpt-4") -> int:
    """
    Accurately computes token usage for a list of chat messages, 
    accounting for structural role metadata and message padding.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_count = 0
    for message in messages:
        token_count += 4  # Overhead wrapper tags: <im_start>, role, tags
        for key, value in message.items():
            token_count += len(encoding.encode(str(value)))
            if key == "name":
                token_count -= 1  # Role is omitted if a custom name is explicitly provided
    token_count += 2  # Every response sequence is primed with <im_start>assistant
    return token_count

def chunk_text_by_tokens(text: str, max_tokens: int = 1000, overlap: int = 150, model_name: str = "gpt-4") -> List[str]:
    """
    Slices raw text into segments based on token counts rather than characters,
    preserving semantic continuity with an overlap window.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        
        if end == len(tokens):
            break
        start += (max_tokens - overlap)

    return chunks