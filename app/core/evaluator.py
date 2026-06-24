import os
import json
from openai import AsyncOpenAI
from app.core.compressor import generate_rolling_summary

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

async def run_differential_evaluation(messages: list, new_question: str) -> dict:
    # ---------------------------------------------------------
    # PATH A: The Gold Standard (Uncompressed)
    # ---------------------------------------------------------
    path_a_messages = messages + [{"role": "user", "content": new_question}]
    
    response_a = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=path_a_messages,
        temperature=0.0
    )
    answer_a = response_a.choices[0].message.content.strip()

    # ---------------------------------------------------------
    # PATH B: The Compressed State
    # ---------------------------------------------------------
    # 1. Compress the history
    compressed_summary = await generate_rolling_summary(messages)
    
    # 2. Build the new prompt using ONLY the summary + the new question
    path_b_messages = [
        {"role": "system", "content": f"Use this compressed context to answer the user:\n{compressed_summary}"},
        {"role": "user", "content": new_question}
    ]
    
    response_b = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=path_b_messages,
        temperature=0.0
    )
    answer_b = response_b.choices[0].message.content.strip()

    # ---------------------------------------------------------
    # THE AI JUDGE
    # ---------------------------------------------------------
    judge_prompt = (
        "You are a strict, mathematical AI Judge grading a context compression engine. "
        "Your task is to compare Answer B against the Gold Standard Answer A.\n\n"
        f"**Question Asked:** {new_question}\n"
        f"**Answer A (Gold Standard):** {answer_a}\n"
        f"**Answer B (Compressed Context):** {answer_b}\n\n"
        "### SCORING RUBRIC (Start at 100 points):\n"
        "- Deduct exactly 14 points for EVERY specific numerical metric missing or altered in Answer B.\n"
        "- Deduct exactly 7 points for EVERY system identifier, file name, or proper noun missing in Answer B.\n"
        "- Deduct exactly 3 points for minor contextual omissions that do not ruin the overall technical accuracy.\n\n"
        "You MUST respond in valid JSON containing exactly two keys. To ensure mathematical accuracy, "
        "you MUST generate the 'reasoning' key FIRST (showing your explicit deduction math), and the 'score' key SECOND."
    )

    judge_response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a strict JSON-only grading API."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    evaluation = json.loads(judge_response.choices[0].message.content)

    return {
        "compressed_summary": compressed_summary,
        "path_a_answer": answer_a,
        "path_b_answer": answer_b,
        "judge_score": evaluation.get("score", 0),
        "judge_reasoning": evaluation.get("reasoning", "Evaluation failed to parse.")
    }