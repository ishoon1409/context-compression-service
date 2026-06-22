Context Compression Service
An intelligent, high-performance API designed to optimize and compress massive context payloads for Large Language Models (LLMs). This service dynamically crushes long conversation histories and massive technical documents into dense architectural summaries, saving token costs and preventing context-window overflow while protecting factual accuracy.

🎯 Core Success Metrics
This architecture is engineered to balance the classic AI system triad:

Compression: Reduce overall context token size by at least 70%.

Accuracy: Retain at least 85% of factual data and goal alignment.

Performance: Execute the entire compression pipeline in under 10 seconds.

🚀 Key Features
The "Sawtooth" Chat Window: Dynamically compresses old conversation turns while locking the most recent "tail" messages to preserve immediate user intent.

Dynamic Map-Reduce Document Engine: Slices massive documents (up to 100k+ tokens) into mathematically overlapping chunks, processes them in parallel asynchronously, and reduces them into a master state based on a scalable retention budget.

Token Inversion Protection: Utilizes strict system prompts to prevent AI models from generating bloated formatting (markdown lists/templates) that accidentally increases token counts on small payloads.

Ultra-Fast Inference: Powered by Meta's llama-3.1-8b-instant model hosted on the Groq API for blazing-fast, sub-second latency.

🛠️ Tech Stack
Framework: FastAPI / Python 3.10+

Server: Uvicorn

LLM Engine: Groq API (llama-3.1-8b-instant)

Token Math: tiktoken (cl100k_base encoding)

Concurrency: Native Python asyncio

📂 Project Structure
Plaintext
context-compression-service/
│
├── app/
│   ├── main.py                     # FastAPI application routing and threshold logic
│   ├── core/
│   │   └── compressor.py           # Chat compression logic (Sawtooth Window)
│   └── documents/
│       └── pdf_processor.py        # Map-Reduce dynamic document compression
│
├── .env                            # Environment variables (API Keys)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
⚙️ Installation & Setup
1. Clone or Create the Directory
Navigate to your desired workspace and ensure you are in the project root.

2. Activate Virtual Environment

PowerShell
python -m venv venv
.\venv\Scripts\Activate
3. Install Dependencies

PowerShell
pip install fastapi uvicorn openai tiktoken pypdf
4. Environment Configuration
Create a .env file in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=gsk_your_actual_key_here
5. Start the Server

PowerShell
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000.

📡 API Endpoints
POST /compress/chat
Analyzes a stream of chat messages and compresses the historical head into a dense system state if the total token count exceeds the defined threshold.

Request Payload:

JSON
{
  "trigger_threshold": 1000,
  "tail_messages_to_keep": 2,
  "messages": [
    { "role": "system", "content": "You are a DevOps engineer." },
    { "role": "user", "content": "Let's debug my Kubernetes cluster..." },
    { "role": "assistant", "content": "Sure, let's look at the pod logs." },
    { "role": "user", "content": "The logs show an OOMKill error." }
  ]
}
Response Payload:

JSON
{
  "original_token_count": 1450,
  "compressed_token_count": 320,
  "compression_ratio": 77.93,
  "was_compressed": true,
  "messages": [
    { "role": "system", "content": "You are a DevOps engineer." },
    { "role": "system", "content": "--- CONVERSATION SUMMARY STATE ---\nUser is debugging a Kubernetes cluster experiencing OOMKill errors..." },
    { "role": "assistant", "content": "Sure, let's look at the pod logs." },
    { "role": "user", "content": "The logs show an OOMKill error." }
  ]
}
🧪 Testing with PowerShell
To test the API locally, open a secondary PowerShell terminal and run the following command to send a mock JSON payload:

PowerShell
$body = @{
    trigger_threshold = 500
    tail_messages_to_keep = 2
    messages = @(
        @{ role = "system"; content = "You are a system architect." }
        @{ role = "user"; content = "Here is my massive 4000 word system design..." }
        @{ role = "assistant"; content = "I have reviewed it." }
        @{ role = "user"; content = "What is the next step?" }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/compress/chat" -Method Post -Body $body -ContentType "application/json"
🔮 Next Steps (Roadmap)
Phase 3: Implement an automated LLM-as-a-Judge Python script to mathematically grade the 85% accuracy retention against a golden dataset.

Phase 4: Upgrade the Document Processor to a Hierarchical Reduce architecture to safely output summaries larger than the 4,000 token physical LLM limits without data loss.