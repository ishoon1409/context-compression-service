import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI

# Direct execution script configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
API_URL = "http://127.0.0.1:8000/compress/chat"
judge_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def run_inference(messages: list, question: str) -> str:
    """Simulates production pipeline call execution."""
    payload = messages + [{"role": "user", "content": question}]
    response = await judge_client.chat.completions.create(
        model="gpt-4o",
        messages=payload,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

async def evaluate_system():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Target Evaluation dataset missing at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for item in dataset:
            print(f"\nEvaluating Context Pipeline ID: {item['conversation_id']}")
            
            # Pass A: Establish control answers using standard history
            control_answer = await run_inference(item["history"], item["evaluation_question"])
            
            # Pass B: Apply context compression pipeline first
            compression_payload = {
                "messages": item["history"],
                "trigger_threshold": 100,  # Explicitly force compression trigger
                "tail_messages_to_keep": 2
            }
            
            try:
                comp_res = await http_client.post(API_URL, json=compression_payload)
                comp_res_json = comp_res.json()
                compressed_history = comp_res_json["messages"]
                print(f"-> Compression Ratio Achieved: {comp_res_json['compression_ratio']}%")
            except Exception as e:
                print(f"API Middleware Unreachable. Ensure uvicorn server is running. Error: {e}")
                return

            experimental_answer = await run_inference(compressed_history, item["evaluation_question"])
            
            # Judge Step: Evaluate factual preservation
            judge_prompt = (
                "You are an objective evaluation metric system.\n"
                "Compare an experimental answer against a control answer and reference ground truth text data.\n"
                f"Ground Truth: {item['ground_truth_answer']}\n"
                f"Control Answer (Full History): {control_answer}\n"
                f"Experimental Answer (Compressed History): {experimental_answer}\n\n"
                "Rate the factual retention quality of the Experimental Answer on a scale from 0.0 (total information loss) to 1.0 (perfect preservation).\n"
                "Output only the raw floating point number string value (e.g., '0.95') and nothing else."
            )
            
            judge_res = await judge_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            
            score = judge_res.choices[0].message.content.strip()
            print(f"-> LLM-as-a-Judge Factual Retention Alignment Score: {score}/1.0")

if __name__ == "__main__":
    load_dotenv = lambda: None # placeholder if using native shell run environment
    asyncio.run(evaluate_system())