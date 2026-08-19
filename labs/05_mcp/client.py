"""
Lab 05: MCP Client

Connects Claude to the MCP server in server.py via the MCP protocol.
The client spawns server.py as a subprocess, discovers its tools at runtime,
then runs an agentic loop letting Claude call those tools.

Quick start, just needs ANTHROPIC_API_KEY:
  python labs/05_mcp/client.py

To see the server's log output in a separate terminal:
  Terminal 1: python labs/05_mcp/server.py   # waits for client to connect
  Terminal 2: python labs/05_mcp/client.py

Claude backend: uses ANTHROPIC_API_KEY (direct API) if set, otherwise
falls back to AWS Bedrock. Swap _make_client() to use AnthropicVertex for GCP.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

QUESTION = (
    "What does the Claude Platform knowledge base say about MCP and tool use? "
    "Also, what is 17 * 23?"
)


def _make_client():
    """Return an Anthropic client: direct API if a key is set, else Bedrock."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        return Anthropic(), "claude-haiku-4-5-20251001"
    else:
        from anthropic import AnthropicBedrock
        return AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1")), \
               "global.anthropic.claude-sonnet-4-6"


def mcp_tool_to_anthropic(tool) -> dict:
    """Convert an MCP Tool object to the Anthropic tool definition format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


async def run():
    # Spawn server.py as a subprocess connected via stdio
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Discover tools from the MCP server ---
            tools_result = await session.list_tools()
            anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools_result.tools]

            print(f"Discovered {len(anthropic_tools)} tools from MCP server:")
            for t in anthropic_tools:
                print(f"  • {t['name']}: {t['description']}")
            print()

            # --- Agentic loop (direct API or Bedrock) ---
            claude, model = _make_client()

            messages = [{"role": "user", "content": QUESTION}]
            print(f"Question: {QUESTION}\n")

            try:
                while True:
                    response = claude.messages.create(
                        model=model,
                        max_tokens=512,
                        tools=anthropic_tools,
                        messages=messages,
                    )

                    messages.append({"role": "assistant", "content": response.content})

                    if response.stop_reason == "end_turn":
                        for block in response.content:
                            if block.type == "text":
                                print(f"\nClaude: {block.text}")
                        break

                    if response.stop_reason == "tool_use":
                        tool_results = []
                        for block in response.content:
                            if block.type == "tool_use":
                                print(f"  -> MCP call: {block.name}({block.input})")
                                # Execute tool via MCP session (not hardcoded locally)
                                result = await session.call_tool(block.name, block.input)
                                output = result.content[0].text if result.content else ""
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": output,
                                })

                        messages.append({"role": "user", "content": tool_results})

            except Exception as exc:
                # _make_client picked its backend off the same env var, so the
                # hint has to follow it, not always Bedrock.
                hint = "ANTHROPIC_API_KEY" if os.getenv("ANTHROPIC_API_KEY") else "AWS credentials"
                print(f"\nClaude API error: {exc}")
                print(f"(Check your {hint} to run the full agentic loop)")


if __name__ == "__main__":
    asyncio.run(run())
