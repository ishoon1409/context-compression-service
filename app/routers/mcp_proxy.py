import re
import time
import uuid
import json
import os
from typing import List, Optional, Union, Any, Dict, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from openai import AsyncOpenAI

from app.core.mcp_compressor import compress_mcp_output
from app.core.mcp_client import fetch_and_compress_mcp, list_available_tools, call_mcp_tool_raw

router = APIRouter()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

URL_PATTERN = re.compile(r"https?://\S+")

MAX_CHAIN_STEPS = 4
MAX_RESULT_CHARS_IN_CONTEXT = 2000

USAGE_HINT = (
    "**MCP / JSON Compression Engine**\n\n"
    "Simplest way to use this: just paste an MCP server URL, optionally with what you want, e.g.:\n\n"
    "```\n"
    "https://your-mcp-server.com get the open issues for project ENG\n"
    "```\n\n"
    "The tool (or chain of tools, if one depends on another) is auto-discovered and auto-run. "
    "If the server has multiple tools and you don't describe what you want, you'll get a list of "
    "available tools to choose from instead.\n\n"
    "For full manual control, you can still send raw JSON:\n\n"
    "1) Compress JSON you already have:\n"
    "```json\n"
    "{\"raw_json\": \"<paste your JSON blob here>\", \"tool_name\": \"jira\"}\n"
    "```\n\n"
    "2) Fetch live with an explicit tool name/arguments:\n"
    "```json\n"
    "{\"mcp_url\": \"https://your-mcp-server\", \"tool_name\": \"jira_search\", "
    "\"arguments\": {\"project\": \"ENG\"}}\n"
    "```"
)


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


def get_text_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    parts = [p.get("text", "") for p in content if p.get("type") == "text"]
    return " ".join(parts).strip()


def format_compression_result(result: Any, heading: str) -> str:
    """
    Renders a compress_mcp_output()-shaped dict into a clean chat message:
    a stats banner followed by the actual compressed YAML in a code block.
    Falls back gracefully if the shape is ever different than expected.
    """
    if isinstance(result, dict) and "compressed_yaml" in result:
        yaml_text = result.get("compressed_yaml", "")
        original_tokens = result.get("original_tokens")
        compressed_tokens = result.get("compressed_tokens")
        ratio = result.get("compression_ratio")

        if original_tokens is not None and compressed_tokens is not None:
            stats_line = (
                f"🗜️ **{heading}** — {original_tokens:,} → {compressed_tokens:,} tokens "
                f"(**{ratio}%** saved)"
            )
        else:
            stats_line = f"🗜️ **{heading}**"

        return f"{stats_line}\n\n---\n\n```yaml\n{yaml_text}\n```"

    if isinstance(result, dict):
        return f"🗜️ **{heading}**\n\n---\n\n```json\n{json.dumps(result, indent=2)}\n```"

    return f"🗜️ **{heading}**\n\n---\n\n{result}"


def format_tool_list(mcp_url: str, tools: List[dict]) -> str:
    """Shown when multiple tools exist and no intent text was given to disambiguate."""
    lines = [f"**Found {len(tools)} tools on `{mcp_url}`** — tell me which one you want, or describe what you're after:\n"]
    for t in tools:
        lines.append(f"- **{t['name']}** — {t.get('description', 'no description')}")
    lines.append(
        f"\nTry again like:\n```\n{mcp_url} <describe what you want>\n```"
    )
    return "\n".join(lines)


def strip_json_fences(raw: str) -> str:
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw


def _call_signature(tool_name: str, arguments: dict) -> str:
    """Normalized signature used to detect duplicate tool calls (same tool, same arguments)."""
    return f"{tool_name}::{json.dumps(arguments, sort_keys=True)}"


async def decide_next_step(tools: List[dict], intent_text: str, history: List[dict]) -> dict:
    """
    Given available tools (with schemas), the user's original request, and any
    steps already executed (with their real results), asks the LLM to decide
    the single next action: either call another tool, or declare the request
    already satisfied. This is what allows dependent tool chains (e.g. resolve
    an ID, then use that real ID in the next call) to work automatically.
    """
    tools_description = json.dumps(
        [{"name": t["name"], "description": t.get("description", ""), "inputSchema": t.get("inputSchema", {})} for t in tools],
        indent=2
    )

    history_description = ""
    already_called = []
    for i, step in enumerate(history, 1):
        truncated_result = step["result"][:MAX_RESULT_CHARS_IN_CONTEXT]
        history_description += (
            f"\nStep {i}: called '{step['tool_name']}' with arguments {json.dumps(step['arguments'])}\n"
            f"Result:\n{truncated_result}\n"
        )
        already_called.append(f"{step['tool_name']}({json.dumps(step['arguments'])})")

    already_called_block = (
        "\n\nCALLS ALREADY MADE (do NOT repeat any of these with the same or equivalent arguments):\n"
        + "\n".join(f"- {c}" for c in already_called)
        if already_called else ""
    )

    prompt = (
        "You are a tool-chaining orchestration engine for an MCP (Model Context Protocol) server. "
        "Some tools depend on the output of other tools — for example, one tool may resolve an ID "
        "that another tool then requires as input. Given the available tools (with schemas), the "
        "user's original request, and the results of any steps already executed, decide the SINGLE "
        "next action.\n\n"
        f"AVAILABLE TOOLS:\n{tools_description}\n\n"
        f"USER REQUEST:\n{intent_text}\n\n"
        f"STEPS EXECUTED SO FAR:{history_description if history else ' none yet'}"
        f"{already_called_block}\n\n"
        "CRITICAL RULES:\n"
        "- If a previous step's result already gave you a real value (an ID, key, or identifier) "
        "that a later tool needs, REUSE that exact real value. Never invent, guess, or reconstruct "
        "a similar-looking value from your own knowledge instead of the one actually returned.\n"
        "- Never call a tool with the same or equivalent arguments as a call already listed above — "
        "if you're tempted to repeat a call, that means you already have the answer; respond with "
        "'done' instead.\n"
        "- If the most recent successful result already substantively answers the user's request, "
        "respond with 'done' immediately rather than calling more tools speculatively.\n\n"
        "If another tool call is genuinely needed, respond with exactly this shape:\n"
        '{"action": "call_tool", "tool_name": "<name>", "arguments": {<matching that tool\'s inputSchema>}}\n\n'
        "If no further call is needed, respond with exactly:\n"
        '{"action": "done"}\n\n'
        "Respond with ONLY the raw JSON object. No markdown fences, no explanation."
    )

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = strip_json_fences(response.choices[0].message.content.strip())

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        raise Exception(f"Chain step decision failed — model returned unparseable output: {raw}")

    return decision


async def run_tool_chain(mcp_url: str, tools: List[dict], intent_text: str) -> Tuple[str, str]:
    """
    Runs an iterative tool-calling loop: asks the LLM what to do next given
    results so far, executes real MCP tool calls, and stops when the LLM
    says the request is satisfied, MAX_CHAIN_STEPS is reached, or the LLM
    tries to repeat an identical prior call (treated as an implicit "done",
    since a repeat means no new information would be gained). Returns
    (raw_text_of_final_result, name_of_final_tool_called).
    """
    valid_names = {t["name"] for t in tools}
    history: List[dict] = []
    seen_signatures: set = set()
    last_tool_name = None

    for step_num in range(MAX_CHAIN_STEPS):
        decision = await decide_next_step(tools, intent_text, history)

        if decision.get("action") == "done":
            break

        if decision.get("action") != "call_tool" or decision.get("tool_name") not in valid_names:
            raise Exception(f"Chain step {step_num + 1} produced an invalid decision: {decision}")

        tool_name = decision["tool_name"]
        arguments = decision.get("arguments", {})
        signature = _call_signature(tool_name, arguments)

        if signature in seen_signatures:
            print(f"[MCP-PROXY] Chain step {step_num + 1}: duplicate call detected ('{tool_name}' with same arguments) — stopping, reusing last result.")
            break

        seen_signatures.add(signature)
        print(f"[MCP-PROXY] Chain step {step_num + 1}: calling '{tool_name}' with {arguments}")

        raw_result = await call_mcp_tool_raw(mcp_url, tool_name, arguments)
        history.append({"tool_name": tool_name, "arguments": arguments, "result": raw_result})
        last_tool_name = tool_name

    if not history:
        raise Exception("Tool chain finished without ever successfully calling a tool.")

    return history[-1]["result"], last_tool_name


def build_response_payload(answer_text: str, model: str):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": "stop"
        }]
    }


@router.post("/mcp/v1/chat/completions")
async def mcp_chat_completions(req: ChatCompletionRequest):
    try:
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found.")

        raw_text = get_text_content(user_messages[-1].content).strip()

        # --- Path 1: legacy full-control JSON blob ---
        try:
            payload = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            payload = None

        if isinstance(payload, dict) and "raw_json" in payload:
            print("[MCP-PROXY] Compressing pasted JSON...")
            result = await compress_mcp_output(payload["raw_json"], payload.get("tool_name", "generic"))
            answer_text = format_compression_result(result, "MCP JSON Compression")

        elif isinstance(payload, dict) and "mcp_url" in payload and "tool_name" in payload:
            print(f"[MCP-PROXY] Fetching live from {payload['mcp_url']} (explicit tool)...")
            result = await fetch_and_compress_mcp(
                payload["mcp_url"], payload["tool_name"], payload.get("arguments", {})
            )
            answer_text = format_compression_result(result, "MCP Live Fetch + Compression")

        else:
            # --- Path 2: plain text containing a bare URL, auto-discover + auto-run (with chaining) ---
            url_match = URL_PATTERN.search(raw_text)

            if not url_match:
                answer_text = USAGE_HINT
            else:
                mcp_url = url_match.group(0)
                intent_text = raw_text.replace(mcp_url, "").strip()

                print(f"[MCP-PROXY] Discovering tools on {mcp_url}...")
                tools = await list_available_tools(mcp_url)

                if not tools:
                    answer_text = f"No tools were found on `{mcp_url}`."
                elif len(tools) == 1:
                    tool_name = tools[0]["name"]
                    print(f"[MCP-PROXY] Single tool auto-selected: {tool_name}")
                    result = await fetch_and_compress_mcp(mcp_url, tool_name, {})
                    answer_text = format_compression_result(result, f"MCP Auto-Compression ({tool_name})")
                elif not intent_text:
                    answer_text = format_tool_list(mcp_url, tools)
                else:
                    print(f"[MCP-PROXY] Running tool chain for: '{intent_text}'...")
                    final_raw_result, last_tool_name = await run_tool_chain(mcp_url, tools, intent_text)
                    result = await compress_mcp_output(final_raw_result, last_tool_name)
                    answer_text = format_compression_result(result, f"MCP Auto-Compression ({last_tool_name})")

        if not req.stream:
            return build_response_payload(answer_text, req.model)

        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4()}"
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": answer_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[MCP-PROXY] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))