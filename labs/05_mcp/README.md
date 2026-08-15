# Lab 05 — MCP (Model Context Protocol)

**What you'll learn:** How to build an MCP server that exposes tools, and how to connect Claude to it so it discovers and calls those tools dynamically — without the tools being hardcoded in the client.

## What makes MCP different from regular tool use

In Labs 01–03, tools are defined in the same script that calls the model. With MCP, the tool definitions and implementations live in a separate process (the server). The client discovers them at runtime over a standard protocol. This separation enables:

- **Tool reuse**: any MCP-compatible client can use the same server
- **Independent deployment**: update tools without touching model code
- **Composability**: connect Claude to many MCP servers at once (filesystem, databases, APIs)

## Run it

```bash
# Install MCP package if needed
pip install mcp

# Terminal 1 — start the server (it will wait for a client)
python labs/05_mcp/server.py

# Terminal 2 — run the client
python labs/05_mcp/client.py
```

Watch Terminal 1 for `[server]` log lines showing each tool call as it arrives.

## What happens step by step

```
client.py starts
    │
    ├─► spawns server.py as subprocess (stdio transport)
    ├─► session.initialize()          — MCP handshake
    ├─► session.list_tools()          — discovers tools at runtime
    ├─► converts MCP tools → Anthropic format
    │
    └─► agentic loop:
            Claude decides to call search_docs("MCP") via tool_use
            client calls session.call_tool("search_docs", {...})  ──► server executes, returns text
            Claude decides to call run_calculation("17 * 23")
            client calls session.call_tool("run_calculation", {...}) ──► server executes, returns 391
            Claude synthesizes and returns end_turn
```

## Architecture diagram

```
┌─────────────────────────────────────┐     stdio     ┌─────────────────────────┐
│  client.py                          │ ◄────────────► │  server.py              │
│                                     │                │                         │
│  AnthropicBedrock                   │                │  search_docs()          │
│  ↕ messages API                     │                │  run_calculation()      │
│  session.list_tools()  (discover)   │                │                         │
│  session.call_tool()   (execute)    │                │  MCP protocol over stdio│
└─────────────────────────────────────┘                └─────────────────────────┘
```

## Exercises

1. Add a third tool to `server.py` — e.g. `get_doc_by_id(id)` — restart the server and confirm `client.py` discovers it automatically without any client changes.
2. Change the question in `client.py` to something that doesn't require any tools. Observe that `stop_reason` goes directly to `end_turn` and no MCP calls are made.
3. Replace the mocked `DOCS` list in `server.py` with reads from a local markdown file — now the server is a real retrieval tool.
4. Connect the client to two MCP servers simultaneously by initializing two sessions and merging their tool lists.
