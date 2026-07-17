import json
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.core.mcp_compressor import compress_mcp_output


def _extract_text_from_content(content) -> str:
    """
    MCP tool results come back as a list of content blocks (TextContent,
    ImageContent, etc). We pull out any text blocks and join them, and
    fall back to a JSON dump of the raw structure for anything else.
    """
    text_parts = []
    for block in content:
        block_text = getattr(block, "text", None)
        if block_text is not None:
            text_parts.append(block_text)
        else:
            try:
                text_parts.append(json.dumps(block.model_dump(), default=str))
            except Exception:
                text_parts.append(str(block))
    return "\n".join(text_parts)


async def call_mcp_tool_raw(
    mcp_url: str,
    tool_name: str,
    arguments: Optional[dict] = None,
    auth_token: Optional[str] = None,
) -> str:
    """
    Connects to a remote MCP server, performs the handshake, calls a single
    tool, and returns the RAW text result with no compression applied. This
    is the building block used both for one-shot fetches and for multi-step
    tool chains, where intermediate results must stay uncompressed so the
    orchestrator can read real values (e.g. a resolved ID) out of them.
    """
    if arguments is None:
        arguments = {}

    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    async with streamablehttp_client(mcp_url, headers=headers) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

            if result.isError:
                error_text = _extract_text_from_content(result.content)
                raise Exception(f"MCP tool '{tool_name}' returned an error: {error_text}")

            return _extract_text_from_content(result.content)


async def fetch_and_compress_mcp(
    mcp_url: str,
    tool_name: str,
    arguments: Optional[dict] = None,
    auth_token: Optional[str] = None,
):
    """
    Single-shot convenience wrapper: calls one tool and pipes its result
    through the compression engine. Kept for the manual JSON-blob flow and
    for the "only one tool exists" auto-discovery case, where no chaining
    is needed.
    """
    try:
        raw_json_str = await call_mcp_tool_raw(mcp_url, tool_name, arguments, auth_token)
        compressed_result = await compress_mcp_output(raw_json_str, tool_name)

        if isinstance(compressed_result, dict):
            compressed_result["source_url"] = mcp_url
            compressed_result["raw_json"] = raw_json_str

        return compressed_result

    except Exception as e:
        raise Exception(f"Failed to fetch/compress from MCP server '{mcp_url}': {str(e)}")


async def list_available_tools(mcp_url: str, auth_token: Optional[str] = None):
    """
    Convenience helper: connects to an MCP server and lists its available
    tools, since tool names aren't discoverable without this call.
    """
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        async with streamablehttp_client(mcp_url, headers=headers) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return [
                    {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                    for t in tools_result.tools
                ]
    except Exception as e:
        raise Exception(f"Failed to list tools from MCP server '{mcp_url}': {str(e)}")