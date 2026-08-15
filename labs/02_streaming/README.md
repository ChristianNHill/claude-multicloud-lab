# Lab 02 — Streaming

**What you'll learn:** How to stream tokens as they're generated on each platform, and why the SDK differences matter for production apps.

## Run it

```bash
python labs/02_streaming/bedrock.py
python labs/02_streaming/vertex.py
python labs/02_streaming/foundry.py
```

## SDK differences worth knowing

| | Bedrock & Vertex | Foundry |
|---|---|---|
| Pattern | `client.messages.stream()` context manager + `stream.text_stream` iterator | `client.complete(stream=True)` + iterate chunks directly |
| Final metadata | `stream.get_final_message()` after context exits | Embedded in last chunk's `usage` field |
| Abstraction | Anthropic SDK normalizes the underlying SSE format | azure-ai-inference exposes raw delta chunks |

Bedrock and Vertex use identical Python because the Anthropic SDK abstracts both — that's the point of using `AnthropicBedrock` and `AnthropicVertex` over raw HTTP.

## Why streaming matters

Without streaming, users stare at a blank screen for the full generation time before seeing anything. For long responses (>200 tokens) the difference is visible. Most production chat UIs require streaming.

## Exercises

1. Measure time-to-first-token vs. time-to-last-token across platforms. Add `time.time()` calls around the stream iterator.
2. Accumulate the streamed text into a string and compare it to a non-streamed response to the same prompt — they should be identical.
3. Add error handling: what happens if the stream is interrupted mid-response? Wrap the iterator in a `try/except` and log partial output.
