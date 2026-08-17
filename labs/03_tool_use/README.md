# Lab 03 — Tool Use

**What you'll learn:** How to implement an agentic loop — letting the model decide when to call tools, executing those tools on your side, and feeding results back — across all three platforms.

This is the lab that changes how you think about the API. In Labs 01 and 02 the response was text to display. Here it's a decision to act on: the model asks for a function call, your code runs it, and the conversation continues. Everything people call "an agent" is this loop with more tools and better state.

The example asks about two cities in one question, so you'll watch the model request two tool calls in a single turn.

## Run it

```bash
python labs/03_tool_use/bedrock.py
python labs/03_tool_use/vertex.py
python labs/03_tool_use/foundry.py
```

## How the loop works

```
User prompt
    │
    ▼
Model response ──► stop_reason == "end_turn"  ──► print final text, done
                │
                └─► stop_reason == "tool_use"  ──► execute tool(s)
                                                        │
                                                        └─► append tool_result(s) as user turn
                                                                │
                                                                └─► back to model ↑
```

Each iteration, the model either finishes (returns text) or requests one or more tool calls. You execute the tools and loop. This is the same pattern used in Claude Code, Operator, and most production agents.

The model never executes anything. It emits a structured request; your code decides whether and how to honor it. That boundary is the whole security model of tool use.

## Walkthrough

Open `bedrock.py` alongside this — it's the clearest of the three.

### Step 1: Describe the tool to the model

```python
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature in Celsius and a condition string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'San Francisco'"},
            },
            "required": ["city"],
        },
    }
]
```

`input_schema` is plain JSON Schema, and the SDK validates the model's output against it before handing it to you.

Treat `description` as a prompt, because that's what it is — it's the only thing the model knows about your function. "Get the current weather for a city. Returns temperature in Celsius" tells it both when to reach for the tool and what shape comes back. A description like `"weather tool"` produces a model that calls it at the wrong times. Same for the per-property descriptions: `"City name, e.g. 'San Francisco'"` is what stops the model passing `"SF"` or `"San Francisco, CA, USA"`.

### Step 2: Implement the tool on your side

```python
def get_weather(city: str) -> dict:
    # Mocked. Replace with requests.get("https://api.openweathermap.org/...") in production.
    mock_data = {...}
    return mock_data.get(city, {"temp_c": 20, "condition": "Unknown"})

def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":
        result = get_weather(**inputs)
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {name}")
```

`dispatch_tool` is the name-to-function lookup, kept separate from the implementation so the loop below never grows an `if` chain. It raises on unknown names rather than returning an error string — the model can only request names you gave it, so an unknown name is a bug in your code, not model behavior to handle gracefully.

Results go back as a string. JSON is the convention because it's unambiguous to parse and the model reads it reliably.

### Step 3: Call, then append the assistant turn verbatim

```python
while True:
    response = client.messages.create(
        model="global.anthropic.claude-sonnet-4-6",
        max_tokens=512,
        tools=TOOLS,
        messages=messages,
    )

    messages.append({"role": "assistant", "content": response.content})
```

`response.content` — the raw block list from Lab 01 — goes straight back into `messages` untouched. Don't reconstruct it or extract just the text: the tool_use blocks carry IDs that the next turn's results must reference, and a rebuilt message loses them.

`tools=TOOLS` is passed on *every* iteration. The API is stateless, so the tool list is part of the request, not a session setting.

### Step 4: Branch on stop_reason

```python
if response.stop_reason == "end_turn":
    for block in response.content:
        if block.type == "text":
            print(block.text)
    break
```

`end_turn` means the model is done and the content is the final answer. This is the loop's only exit — which is why exercise 4 below matters.

### Step 5: Execute the tools and feed results back

```python
if response.stop_reason == "tool_use":
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"  -> calling {block.name}({block.input})")
            output = dispatch_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

    messages.append({"role": "user", "content": tool_results})
```

Three things here surprise people:

- **The inner loop is over blocks, not a single call.** Asking about San Francisco *and* Seattle produces two `tool_use` blocks in one response. Code that grabs `content[0]` works in the demo and breaks the first time a user asks a two-part question.
- **`tool_use_id` is the correlation key.** Each result must carry the ID of the request it answers, and all results for a turn go back together. Miss one and the API rejects the turn.
- **Results are sent as `"role": "user"`.** They're not user input in any human sense, but the conversation alternates user/assistant, and the model's turn was the assistant one. The tool result is the reply to it.

## SDK format differences

The Anthropic SDK (Bedrock + Vertex) and the Azure AI Inference SDK (Foundry) use different formats for tool definitions and results:

| Concept | Anthropic SDK | Azure AI Inference |
|---|---|---|
| Tool definition | `{"name": ..., "input_schema": {...}}` | `ChatCompletionsToolDefinition` |
| Stop signal | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| Tool call fields | `block.id`, `block.name`, `block.input` (dict) | `call.id`, `call.function.name`, `call.function.arguments` (JSON string) |
| Result message | `{"type": "tool_result", "tool_use_id": ...}` | `ToolMessage(tool_call_id=...)` |

Two of these bite in practice. Foundry hands you arguments as a **JSON string**, so `foundry.py` calls `json.loads` inside `dispatch_tool` where the Bedrock version receives a dict directly. And Foundry sends one `ToolMessage` per result as separate messages, rather than the Anthropic SDK's single user turn containing a list of results.

## Exercises

1. Add a second tool — for example `get_time(city)` — and ask a question that requires both.
2. Add `tool_choice={"type": "any"}` to force Claude to always use a tool on the first turn.
3. Replace `get_weather` with a real HTTP call to `wttr.in` (no API key needed): `requests.get(f"https://wttr.in/{city}?format=j1").json()`
4. Add a max-iterations guard to the while loop to prevent runaway agents. `while True` with a single exit condition is fine for a lab and not fine in production — Lab 06 shows the guarded version.
5. Ask about a city that isn't in `mock_data` and watch how the model handles the `"Unknown"` condition. Tool errors are just strings the model reads.

## Next

[Lab 04 — Cross-Platform Comparison](../04_compare/) runs all three platforms concurrently and puts the latency and token numbers side by side.
