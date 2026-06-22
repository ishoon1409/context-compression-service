import requests
import json
import os
import time  # New dependency for managing execution intervals

# Configuration
FASTAPI_URL = "http://127.0.0.1:8000/compress/chat"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def query_groq(messages, max_retries=5, initial_backoff=5): # Increased initial backoff
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.0 
    }
    
    for attempt in range(max_retries):
        try:
            res = requests.post(GROQ_API_URL, headers=headers, json=data)
            
            if res.status_code == 429:
                # FORCE a longer wait. Groq's header is often too optimistic for free tiers.
                # Calculate exponential backoff: 5s, 10s, 20s, 40s...
                sleep_time = initial_backoff * (2 ** attempt)
                
                print(f"\n   ⚠️ [Rate Limit Hit] Throttling for {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                continue
            
            res.raise_for_status()
            
            # Increase safety padding between successful calls from 1s to 2s
            time.sleep(2) 
            return res.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(initial_backoff * (2 ** attempt))
                continue
            raise e
            
    raise Exception("🚫 System Failure: Unable to clear rate-limiting constraints.")

def evaluate_suite(suite):
    """Runs differential context analysis for a single multi-turn test case suite."""
    print(f"\n============================================================")
    print(f" STARTING BENCHMARK SUITE: {suite['name']}")
    print(f"============================================================")
    
    # Fetch compression components from local FastAPI application server
    payload = {
        "trigger_threshold": suite["trigger_threshold"],
        "tail_messages_to_keep": suite["tail_messages_to_keep"],
        "messages": suite["messages"]
    }
    
    compress_res = requests.post(FASTAPI_URL, json=payload)
    compress_res.raise_for_status()
    compression_data = compress_res.json()
    
    print(f" -> System Prompt + History reduced by: {compression_data.get('compression_ratio')}%")
    
    # Extract the summary state string directly from the reconstructed messages array
    compressed_summary = ""
    for msg in compression_data.get("messages", []):
        if msg["role"] == "system" and "--- CONVERSATION SUMMARY STATE ---" in msg["content"]:
            compressed_summary = msg["content"]
            break
            
    if not compressed_summary:
        print("\n❌ ERROR: Failed to locate the summary state inside the returned messages array!")
        import sys
        sys.exit(1)
    
    suite_scores = []
    original_messages = suite["messages"]
    system_prompt = next((m["content"] for m in original_messages if m["role"] == "system"), "")

    # Iterate through all diagnostic questions mapped to this specific scenario
    for idx, item in enumerate(suite["test_questions"]):
        
        # Unpack the dictionary format if present; fallback to raw string if old format
        if isinstance(item, dict):
            question_text = item.get("query", "")
            q_type = item.get("type", "exact_match")
        else:
            question_text = item
            q_type = "exact_match"

        print(f"\n  [Test {idx + 1}] Question: '{question_text}' (Type: {q_type.upper()})")
        
        # --- PATH A: Pure Uncompressed Complete Stream ---
        path_a_messages = original_messages + [{"role": "user", "content": question_text}]
        gold_answer = query_groq(path_a_messages)
        
        # --- DEBUG: Print the verified context summary passed to Path B ---
        print(f"\n      🔍 DEBUG - Historical Context Summary Passed to Path B:")
        print(f"      {repr(compressed_summary)}") 
        print("      --------------------------------------------------")

        # --- PATH B: Stitched Compressed Engine Stream ---
        stitched_messages = [
            {"role": "system", "content": f"{system_prompt}\n\nHistorical Context Summary:\n{compressed_summary}"}
        ]
        tail_count = suite.get("tail_messages_to_keep", 2)
        recent_messages = [m for m in original_messages if m["role"] != "system"][-tail_count:]
        stitched_messages.extend(recent_messages)
        stitched_messages.append({"role": "user", "content": question_text})
        
        compressed_answer = query_groq(stitched_messages)

        # --- DISPLAY ANSWERS SIDE-BY-SIDE IN TERMINAL ---
        print("\n      [Path A] Gold Standard Answer:")
        print(f"      {gold_answer.strip()}")
        print("\n      [Path B] Compressed Context Answer:")
        print(f"      {compressed_answer.strip()}")
        print("      --------------------------------------------------")
        
        # --- DYNAMIC JUDGE RUBRIC SELECTION ---
        if q_type == "exact_match":
            rubric_instructions = (
                "Grade factual retention on an integer scale from 0 to 100:\n"
                " - 100: Identical factual parameters, metrics, settings, configurations, and identifiers are preserved.\n"
                " - 70-90: Conceptually valid, but minor naming or descriptive quality is omitted.\n"
                " - 40-60: Missing critical metrics, specific configurations, proper nouns, or parameters.\n"
                " - 0: Completely contradictory, hallucinated, or states information is missing."
            )
        else:
            rubric_instructions = (
                "Evaluate thematic and semantic alignment on an integer scale from 0 to 100:\n"
                " - 100: The Compressed Answer captures the complete rationale, timeline, and core narrative of the Gold Standard. Paraphrasing is fully acceptable.\n"
                " - 70-90: The main point is present, but a supporting argument or secondary reason is missing.\n"
                " - 40-60: The answer captures only a fragment of the narrative or fundamentally misunderstands the core rationale.\n"
                " - 0: The answer is entirely hallucinated or unrelated."
            )

        # --- AI JUDGE VERDICT ---
        judge_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI quality assurance judge checking for data loss during context compression.\n"
                    "Compare the 'Compressed Context Answer' against the 'Gold Standard Answer'.\n"
                    f"{rubric_instructions}\n\n"
                    "Output ONLY the raw integer numerical score between 0 and 100. Do not write markdown or conversational filler."
                )
            },
            {
                "role": "user",
                "content": f"Gold Standard Answer:\n{gold_answer}\n\nCompressed Context Answer:\n{compressed_answer}"
            }
        ]
        
        score_str = query_groq(judge_prompt).strip()
        try:
            score = int(''.join(filter(str.isdigit, score_str)) or 0)
        except ValueError:
            score = 0
            
        print(f"   -> Factual Equivalence Alignment Score: {score}/100")
        suite_scores.append(score)

        print("   ⏳ Cooling down API bucket for 10 seconds before next question...")
        time.sleep(10)
        
    suite_avg = sum(suite_scores) / len(suite_scores)
    print(f"\n >>> Suite Subtotal Performance Score: {suite_avg}%")
    return suite_avg

def run_system_benchmark():
    # Load multi-scenario payload configuration mapping
    with open("test_history.json", "r") as f:
        data = json.load(f)
        
    suites = data.get("test_suites", [])
    all_suite_averages = []
    
    # Run automation across every configured functional area
    for suite in suites:
        avg_score = evaluate_suite(suite)
        all_suite_averages.append(avg_score)
        
    # Compile multi-scenario statistical mean metric
    total_system_accuracy = sum(all_suite_averages) / len(all_suite_averages)
    
    print("\n" + "============================================================")
    print("      GLOBAL ARCHITECTURAL COMPRESSION BENCHMARK REPORT")
    print("============================================================")
    print(f"Total Test Cases Evaluated:    {len(suites)} Architectural Domains")
    print(f"Global System Accuracy Mean:   {round(total_system_accuracy, 2)}%")
    print(f"System Success Criteria Goal:  >= 85.0%")
    print(f"Overall Production Status:     {'PASSED' if total_system_accuracy >= 85 else 'FAILED'}")
    print("============================================================")

if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("ERROR: Please load your GROQ_API_KEY environment variable setup.")
    else:
        run_system_benchmark()