# Lab 05: MCP (model context protocol)

**What you'll learn:** How to build an MCP server that exposes tools, and how to connect Claude to it so it discovers and calls those tools dynamically, without the tools being hardcoded in the client.

Lab 03's agentic loop had one structural weakness: the tools lived in the same file as the model call. This lab pulls them out into a separate process and has the client ask what's available at startup. The loop itself barely changes. That's the point worth watching for.

## What makes MCP different from regular tool use

In Labs 01 to 03, tools are defined in the same script that calls the model. With MCP, the tool definitions and implementations live in a separate process (the server). The client discovers them at runtime over a standard protocol. This separation enables:

- **Tool reuse**: any MCP-compatible client can use the same server
- **Independent deployment**: update tools without touching model code
- **Composability**: connect Claude to many MCP servers at once (filesystem, databases, APIs)

## Run it

```bash
# Install MCP package if needed
pip install mcp

# Terminal 1: start the server (it will wait for a client)
python labs/05_mcp/server.py

# Terminal 2: run the client
python labs/05_mcp/client.py
```

Watch Terminal 1 for `[server]` log lines showing each tool call as it arrives.

The client spawns its own copy of the server either way, the Terminal 1 instance is there so you can watch the logs. Running only `client.py` works fine.

## What happens step by step

```
client.py starts
    │
    ├─► spawns server.py as subprocess (stdio transport)
    ├─► session.initialize(): MCP handshake
    ├─► session.list_tools(): discovers tools at runtime
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

Claude never talks to the server. The client is the only thing that speaks both protocols, Anthropic Messages API on one side, MCP on the other. It's a translator, and most of `client.py` is that translation.

## Walkthrough: the server

### Step 1: Declare tools with a decorator

```python
app = MCPServer("claude-multicloud-lab")

@app.tool(
    name="search_docs",
    description="Search the Claude Platform knowledge base. Returns matching documents.",
)
def search_docs(query: str) -> str:
    """Search documentation by keyword."""
    ...
```

No hand-written JSON Schema here, unlike Lab 03. `MCPServer` builds the schema from the function's type hints, so `query: str` becomes a required string property. The `description` still carries the same weight it did in Lab 03, it's what the model uses to decide when to call this.

### Step 2: Log to stderr, never stdout

```python
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[server] %(message)s")
```

This is the one thing that will break your first MCP server without any obvious error. Under the stdio transport, **stdout is the protocol channel**: it carries JSON-RPC messages between client and server. A stray `print()` in a tool handler injects garbage into that stream and corrupts the session, usually with an error that points nowhere near the actual `print`. All diagnostics go to stderr, which is why you can watch them in Terminal 1 without disturbing anything.

### Step 3: Execute untrusted input safely

```python
_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ...}

def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    ...
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")
```

`run_calculation` takes a string the model composed and evaluates it as arithmetic. The obvious implementation is `eval(expression)`, and it's remote code execution: a model that can be talked into emitting `__import__('os').system(...)` now runs it on your machine.

Instead the expression is parsed to an AST and walked with an explicit allowlist of node types. Anything not on the list raises. This is the shape of every safe evaluator: allowlist what's permitted rather than blocklisting what isn't.

The general rule holds beyond this example. MCP tools run with your process's privileges, and the arguments come from a model that a user can influence.

### Step 4: Serve over stdio

```python
if __name__ == "__main__":
    asyncio.run(app.run_stdio_async())
```

stdio is one of several MCP transports (HTTP and SSE are the others). It's the right one for a local subprocess: no ports, no auth, and the OS cleans up the child when the parent exits.

## Walkthrough: the client

### Step 5: Spawn the server as a subprocess

```python
server_params = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

`sys.executable` rather than `"python"`: that's the interpreter currently running, so the subprocess inherits your virtualenv instead of hitting whatever `python` resolves to on `PATH`. A hardcoded `"python"` is the most common reason a working server fails to start from a different client.

`session.initialize()` is the MCP handshake: both sides exchange protocol version and capabilities before any tool call is legal.

### Step 6: Discover, then translate

```python
tools_result = await session.list_tools()
anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools_result.tools]

def mcp_tool_to_anthropic(tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
```

This is the heart of the lab. `list_tools()` asks the server what it can do, at runtime, the client has no compile-time knowledge of `search_docs` or `run_calculation`. Add a tool to the server, restart it, and the client picks it up with no code change (exercise 1).

The translation is nearly a no-op because both formats are name + description + JSON Schema. That's not a coincidence; MCP's tool shape was designed to map onto what models already accept.

### Step 7: The same loop as Lab 03, one line different

```python
if response.stop_reason == "tool_use":
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = await session.call_tool(block.name, block.input)      # ← the only change
            output = result.content[0].text if result.content else ""
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

    messages.append({"role": "user", "content": tool_results})
```

Compare this to Step 5 of Lab 03. Identical: `stop_reason` branch, block iteration, `tool_use_id` correlation, results appended as a user turn. Except `dispatch_tool(...)` became `await session.call_tool(...)`.

That's the takeaway. MCP doesn't change how the model uses tools. It changes where the tools live, and the model can't tell the difference.

`result.content` is a list of content blocks (MCP supports text, images, and embedded resources), hence `result.content[0].text` and the empty guard.

## Exercises

1. Add a third tool to `server.py`: e.g. `get_doc_by_id(id)`: restart the server and confirm `client.py` discovers it automatically without any client changes.
2. Change the question in `client.py` to something that doesn't require any tools. Observe that `stop_reason` goes directly to `end_turn` and the client makes no MCP calls.
3. Replace the mocked `DOCS` list in `server.py` with reads from a local markdown file. Now the server is a real retrieval tool.
4. Connect the client to two MCP servers simultaneously by initializing two sessions and merging their tool lists. You'll need a name-to-session map to route `call_tool` correctly.
5. Add a `print("hello")` to a tool handler in `server.py` and watch the session break. Then remove it. This failure mode is worth seeing once deliberately.

## Next

[Lab 06: Stateful Agent](../06_stateful_agent/) keeps the tools local again but gives the model tools for managing its own workflow: planning, recording, and reviewing its progress.
