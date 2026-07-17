import os
import chromadb
import pymupdf4llm
from openai import AsyncOpenAI
from app.core.pdf_compressor import semantic_chunk_markdown
from dotenv import load_dotenv

# Force load the .env file so os.getenv can find your keys
load_dotenv()

# Point the OpenAI client to Groq's high-speed endpoint
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Initialize Local Vector Database (ChromaDB)
# This will create a persistent folder named 'vector_db' in your root directory to save your data permanently.
db_client = chromadb.PersistentClient(path="./vector_db")

# ... existing code ...
async def build_rag_knowledge_base(pdf_path: str, kb_name: str = "master_knowledge_base") -> dict:
    """
    INGESTION PIPELINE: Extracts text from a PDF, chunks it semantically, 
    embeds it into vectors, and stores it permanently in the Vector DB.
    """
    try:
        # 1. Get or create the vector collection (like a SQL table)
        collection = db_client.get_or_create_collection(name=kb_name)
        
        # 2. Extract and chunk (Reusing your awesome semantic chunker!)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # We use a smaller token limit (1000) for RAG to keep the retrieved context highly specific
        chunks = semantic_chunk_markdown(md_text, max_tokens=1000) 
        
        # 3. Load into Vector DB
        # ChromaDB automatically handles the complex mathematical embedding process locally!
        # We generate a unique ID for every single chunk based on the filename
        base_filename = os.path.basename(pdf_path).replace(".pdf", "")
        ids = [f"{base_filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": base_filename, "chunk_index": i} for i in range(len(chunks))]
        
        collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        return {
            "status": "success", 
            "message": f"Successfully embedded and stored {len(chunks)} semantic chunks into '{kb_name}'.",
            "kb_name": kb_name
        }
    except Exception as e:
        raise Exception(f"Failed to build Knowledge Base: {str(e)}")
    
    # ... existing code ...

async def query_rag_knowledge_base(query: str, kb_name: str = "master_knowledge_base") -> str:
    """Queries ChromaDB for context and asks the LLM to generate an answer."""
    try:
        # 1. Access the database collection
        try:
            collection = db_client.get_collection(name=kb_name)
        except Exception:
            return "I don't have any documents in my Knowledge Base yet! Please upload a PDF first."

        # 2. Perform Semantic Vector Search (Retrieve top 3 chunks)
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # 3. Extract the text context from the search results
        if not results['documents'] or not results['documents'][0]:
            return "I couldn't find any relevant information in the uploaded documents."
            
        retrieved_context = "\n\n---\n\n".join(results['documents'][0])
        
        # 4. Construct the Augmented Prompt
        system_prompt = (
            "You are a highly precise Context-Aware AI Assistant. "
            "Use ONLY the provided Context to answer the user's question. "
            "If the answer is not contained in the context, explicitly state: 'I cannot answer this based on the provided documents.' "
            "Do not hallucinate external knowledge."
        )
        
        user_prompt = f"Context:\n{retrieved_context}\n\nQuestion:\n{query}"
        
        # 5. Generate the Final Answer using Groq (Fast Inference)
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant", # Or your preferred model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # This allows the exact error to bubble up to your terminal for debugging!
        print(f"RAG Engine Crash: {str(e)}")
        raise Exception(f"RAG Engine Query Failed: {str(e)}")

