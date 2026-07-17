import asyncio
import requests
import json
import os

FASTAPI_URL = "http://127.0.0.1:8000/compress/chat"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Default logger just prints to terminal if run normally
async def default_logger(msg: str):
    print(msg)

async def query_groq(messages, logger, is_json=False, max_retries=5, initial_backoff=5):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.0}
    
    # NEW: Force Groq to return pure JSON when the Judge is speaking
    if is_json:
        data["response_format"] = {"type": "json_object"}
        
    for attempt in range(max_retries):
        try:
            res = await asyncio.to_thread(requests.post, GROQ_API_URL, headers=headers, json=data)
            
            if res.status_code == 429:
                sleep_time = initial_backoff * (2 ** attempt)
                await logger(f"\n   ⚠️ [Rate Limit Hit] Throttling for {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(sleep_time)
                continue
            
            res.raise_for_status()
            await asyncio.sleep(2) 
            return res.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(initial_backoff * (2 ** attempt))
                continue
            raise e
    raise Exception("🚫 System Failure: Unable to clear rate-limiting constraints.")

async def evaluate_suite(suite, logger):
    await logger(f"\n============================================================")
    await logger(f" STARTING BENCHMARK SUITE: {suite['name']}")
    await logger(f"============================================================")
    
    payload = {
        "trigger_threshold": suite["trigger_threshold"],
        "tail_messages_to_keep": suite["tail_messages_to_keep"],
        "messages": suite["messages"]
    }
    
    compress_res = await asyncio.to_thread(requests.post, FASTAPI_URL, json=payload)
    compress_res.raise_for_status()
    compression_data = compress_res.json()
    
    await logger(f" -> System Prompt + History reduced by: {compression_data.get('compression_ratio')}%")
    
    compressed_summary = ""
    for msg in compression_data.get("messages", []):
        if msg["role"] == "system" and "--- CONVERSATION SUMMARY STATE ---" in msg["content"]:
            compressed_summary = msg["content"]
            break
            
    if not compressed_summary:
        await logger("\n❌ ERROR: Failed to locate the summary state!")
        return 0

    suite_scores = []
    original_messages = suite["messages"]
    system_prompt = next((m["content"] for m in original_messages if m["role"] == "system"), "")

    for idx, item in enumerate(suite["test_questions"]):
        if isinstance(item, dict):
            question_text = item.get("query", "")
            q_type = item.get("type", "exact_match")
        else:
            question_text = item
            q_type = "exact_match"

        await logger(f"\n  [Test {idx + 1}] Question: '{question_text}'")
        
        path_a_messages = original_messages + [{"role": "user", "content": question_text}]
        gold_answer = await query_groq(path_a_messages, logger)
        
        stitched_messages = [{"role": "system", "content": f"{system_prompt}\n\nHistorical Context Summary:\n{compressed_summary}"}]
        tail_count = suite.get("tail_messages_to_keep", 2)
        recent_messages = [m for m in original_messages if m["role"] != "system"][-tail_count:]
        stitched_messages.extend(recent_messages)
        stitched_messages.append({"role": "user", "content": question_text})
        
        compressed_answer = await query_groq(stitched_messages, logger)

        if q_type == "exact_match":
            scoring_rules = "Deduct exactly 14 points for every specific numerical metric missing. Deduct exactly 7 points for missing files/identifiers."
        else:
            scoring_rules = "Deduct exactly 15 points if the primary core takeaway is completely missing."

        judge_prompt = [
            {
                "role": "system", 
                "content": (
                    f"You are a strict mathematical AI judge. \n{scoring_rules}\n"
                    "You MUST output valid JSON with EXACTLY two keys: "
                    "'reasoning' (your step-by-step math proof) and 'score' (the final integer)."
                )
            },
            {"role": "user", "content": f"Gold Standard:\n{gold_answer}\n\nCompressed:\n{compressed_answer}"}
        ]
        
        # NEW: Trigger the is_json=True flag we just built!
        score_response = (await query_groq(judge_prompt, logger, is_json=True)).strip()
        
        try:
            judge_data = json.loads(score_response)
            score = int(judge_data.get("score", 0))
            await logger(f"   🔍 Judge Math: {judge_data.get('reasoning')}")
        except Exception:
            await logger(f"   ⚠️ Judge parsing error. Raw output: {score_response}")
            score = 50 
            
        await logger(f"   -> Alignment Score: {score}/100")
        suite_scores.append(score)

        await logger("   ⏳ Cooling down API bucket for 10 seconds...")
        await asyncio.sleep(10)
        
    suite_avg = sum(suite_scores) / len(suite_scores) if suite_scores else 0
    await logger(f"\n >>> Suite Subtotal Score: {suite_avg}%")
    return suite_avg

async def run_system_benchmark(logger=default_logger):
    if not GROQ_API_KEY:
        await logger("ERROR: Please load your GROQ_API_KEY environment variable.")
        return

    with open("app/test_history.json", "r") as f:
        data = json.load(f)
        
    suites = data.get("test_suites", [])
    all_suite_averages = []
    
    for suite in suites:
        avg_score = await evaluate_suite(suite, logger)
        all_suite_averages.append(avg_score)
        
    total_system_accuracy = sum(all_suite_averages) / len(all_suite_averages) if all_suite_averages else 0
    
    await logger("\n============================================================")
    await logger("      GLOBAL ARCHITECTURAL COMPRESSION BENCHMARK REPORT")
    await logger("============================================================")
    await logger(f"Total Test Cases Evaluated:    {len(suites)} Domains")
    await logger(f"Global System Accuracy Mean:   {round(total_system_accuracy, 2)}%")
    await logger(f"System Success Criteria Goal:  >= 85.0%")
    await logger(f"Overall Production Status:     {'PASSED ✅' if total_system_accuracy >= 85 else 'FAILED ❌'}")
    await logger("============================================================")

if __name__ == "__main__":
    asyncio.run(run_system_benchmark())