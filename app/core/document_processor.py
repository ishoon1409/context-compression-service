import fitz  # PyMuPDF
import tiktoken

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text from a PDF file buffer."""
    text = ""
    # Open the PDF from memory
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def chunk_text_by_tokens(text: str, chunk_size: int = 2000, overlap: int = 200, model_name: str = "gpt-3.5-turbo") -> list[str]:
    """
    Splits text into chunks of a specific token size with a defined overlap.
    """
    # We use cl100k_base as a reliable proxy for Llama 3 tokenization
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        # Grab a slice of tokens
        chunk_tokens = tokens[i : i + chunk_size]
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        # Move the index forward, accounting for the overlap
        i += chunk_size - overlap
        
    return chunks