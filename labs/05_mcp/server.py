"""
Lab 05: MCP Server

A minimal MCP (Model Context Protocol) server that exposes two tools to any
MCP-compatible client. Run this first in its own terminal.

Usage:
  python labs/05_mcp/server.py

The server waits for a client to connect via stdio. When client.py runs, it
spawns this server as a subprocess and communicates over stdin/stdout.
The server's stderr log lines are visible in this terminal as calls arrive.

Tools exposed:
  search_docs(query): Keyword search over a small mocked doc store
  run_calculation(expression): Evaluates a safe arithmetic expression
"""

import ast
import asyncio
import logging
import operator
import sys

from mcp.server import MCPServer

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[server] %(message)s")
log = logging.getLogger(__name__)

app = MCPServer("claude-multicloud-lab")

# --- Mocked knowledge base ---

DOCS = [
    {"id": "1", "title": "What is Claude?",            "body": "Claude is a family of AI assistants built by Anthropic, designed to be helpful, harmless, and honest."},
    {"id": "2", "title": "AWS Bedrock overview",        "body": "Amazon Bedrock is a fully managed service that provides access to foundation models via API, including Claude."},
    {"id": "3", "title": "Google Vertex AI overview",   "body": "Vertex AI is Google Cloud's ML platform. Claude models are available in Model Garden."},
    {"id": "4", "title": "Azure AI Foundry overview",   "body": "Azure AI Foundry lets teams build AI apps with models including Claude via the Azure Marketplace."},
    {"id": "5", "title": "MCP protocol",                "body": "MCP (Model Context Protocol) is an open standard that lets LLMs connect to external tools and data sources through a unified interface."},
    {"id": "6", "title": "Tool use / function calling", "body": "Tool use lets Claude call external functions. The model returns a tool_use block; your code executes the function and returns the result."},
    {"id": "7", "title": "Streaming",                   "body": "Streaming returns tokens as they're generated instead of waiting for the full response, reducing time-to-first-token."},
]

# Safe arithmetic evaluator: no eval() or exec()
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# --- Tool definitions ---
# MCPServer infers the input schema from the function's type hints and docstring.

@app.tool(
    name="search_docs",
    description="Search the Claude Platform knowledge base. Returns matching documents.",
)
def search_docs(query: str) -> str:
    """Search documentation by keyword."""
    log.info("search_docs(%r)", query)
    q = query.lower()
    matches = [d for d in DOCS if q in d["title"].lower() or q in d["body"].lower()]
    if not matches:
        return f"No documents found for query: {query!r}"
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['body']}" for d in matches)


@app.tool(
    name="run_calculation",
    description="Evaluate a safe arithmetic expression and return the numeric result.",
)
def run_calculation(expression: str) -> str:
    """Evaluate arithmetic: supports +, -, *, /, ** and parentheses."""
    log.info("run_calculation(%r)", expression)
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    log.info("MCP server starting on stdio")
    asyncio.run(app.run_stdio_async())
