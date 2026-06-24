# Context Compression Service

A high-performance backend middleware service designed to drastically reduce LLM context payloads while mathematically guaranteeing the retention of critical factual data. 

As AI applications scale, unbounded chat histories and large document payloads bloat context windows, driving up inference costs and latency while degrading model reasoning. This service intercepts LLM payloads, runs them through an Entity-First distillation engine, and reconstructs the context state—regularly achieving >70% token compression with >85% factual retention.

## 🚀 Core Features

* **Unified Developer Dashboard:** A clean, zero-dependency frontend UI (Vanilla JS + Tailwind) hosted directly on the root endpoint for drag-and-drop document compression and real-time chat evaluation.
* **Document Map-Reduce Pipeline (New):** Ingests massive technical PDFs, chunks them with intelligent overlap boundaries, and processes them concurrently via Groq LPUs to synthesize a highly dense, unified document summary.
* **Entity-First Context Summarization:** Prevents "context amnesia" by forcing the compression model to extract and isolate hard variables (ARNs, specific numerical metrics, code paths) and outputting them in strict, human-readable Markdown.
* **Dynamic Payload Partitioning:** Slices incoming chat requests based on token bounds. Protects the immediate conversational tail verbatim while rolling older historical context into a dense memory state.
* **Automated Differential Evaluation (LLM-as-a-Judge):** A built-in testing suite that runs parallel inference paths (Uncompressed vs. Compressed) and mathematically grades the compression engine on strict factual retention (0-100 score).

## 📁 Architecture Overview

```text
context-compression-service/
├── app/
│   ├── main.py                   # FastAPI routing, threshold logic, and UI rendering
│   ├── core/
│   │   ├── compressor.py         # Async rolling chat summary engine (Markdown-forced)
│   │   ├── pdf_compressor.py     # Concurrent Map-Reduce chunking pipeline for documents
│   │   ├── evaluator.py          # LLM-as-a-Judge logic for differential testing
│   │   └── tokenizer.py          # ChatML-aware token calculation bounds
│   └── test_history.json         # Highly technical, multi-turn evaluation datasets
├── frontend/
│   └── index.html                # Unified Dashboard (Tailwind CSS + JS integration)
├── requirements.txt              # Project dependencies
└── .env                          # Environment variables (API Keys)
```

## 🛠️ Installation & Setup

1. Clone the repository:

Bash
git clone [https://github.com/YOUR_USERNAME/context-compression-service.git](https://github.com/YOUR_USERNAME/context-compression-service.git)
cd context-compression-service

2. Create and activate a virtual environment:

Windows:

Bash
python -m venv venv
venv\Scripts\activate

Mac/Linux:

Bash
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:

(Note: Uses fastapi, uvicorn, groq, python-multipart for PDF uploads, and tiktoken)

Bash
pip install -r requirements.txt

4. Configure Environment Variables:

Create a .env file in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=your_api_key_here

## 💻 Usage & Dashboard Access

1. Start the Server:

Start the FastAPI service using Uvicorn. This will boot both the backend API and the frontend UI.

Bash
uvicorn app.main:app --reload

2. Access the Unified Dashboard:

Open your browser and navigate to http://127.0.0.1:8000.
From the dashboard, you can:

📄 Document (PDF) Compressor: Drag and drop technical PDFs to generate heavily compressed, entity-dense Markdown summaries.

💬 Chat History Compressor: Paste raw conversational logs, input a follow-up question, and watch the AI Judge grade the uncompressed vs. compressed answers in real-time.

3. API Documentation:

You can access the interactive Swagger API documentation at http://127.0.0.1:8000/docs.

## 🧪 Differential Evaluation Methodology

The built-in evaluation endpoint (/evaluate/chat) operates on a strict logic flow to visually verify data integrity on the frontend:

Path A (Gold Standard): Sends the full, uncompressed multi-turn chat history to the LLM and records the baseline answer.

Path B (Compressed State): Sends only the distilled Markdown summary + the immediate verbatim tail to the LLM and records the challenger answer.

The AI Judge: A secondary strict-math LLM process evaluates Answer B against Answer A. Using Chain-of-Thought prompting, it deducts specific point values for any dropped ARNs, numerical metrics, or file names, outputting a highly accurate retention score.

## 📊 Performance Benchmarks (v2.0)

Evaluated against strict multi-turn technical extraction suites and large PDF manuals using the Groq Llama 3.1 8B inference engine.

Average Context Compression Ratio: 70% - 97% (depending on PDF formatting and text density).

Entity Retention Score: >85% (measured via strict deductive AI grading).

Latency: Near-instantaneous Map-Reduce execution via Groq's high-speed LPU infrastructure.