# Lab 03 — Tool Use

**What you'll learn:** How to implement an agentic loop — letting the model decide when to call tools, executing those tools on your side, and feeding results back — across all three platforms.

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

## SDK format differences

The Anthropic SDK (Bedrock + Vertex) and the Azure AI Inference SDK (Foundry) use different formats for tool definitions and results:

| Concept | Anthropic SDK | Azure AI Inference |
|---|---|---|
| Tool definition | `{"name": ..., "input_schema": {...}}` | `ChatCompletionsFunctionToolDefinition` |
| Stop signal | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| Tool call fields | `block.id`, `block.name`, `block.input` (dict) | `call.id`, `call.function.name`, `call.function.arguments` (JSON string) |
| Result message | `{"type": "tool_result", "tool_use_id": ...}` | `ToolMessage(tool_call_id=...)` |

## Exercises

1. Add a second tool — for example `get_time(city)` — and ask a question that requires both.
2. Add `tool_choice={"type": "any"}` to force Claude to always use a tool on the first turn.
3. Replace `get_weather` with a real HTTP call to `wttr.in` (no API key needed): `requests.get(f"https://wttr.in/{city}?format=j1").json()`
4. Add a max-iterations guard to the while loop to prevent runaway agents.
