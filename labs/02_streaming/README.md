# Lab 02 — Streaming

**What you'll learn:** How to stream tokens as they're generated on each platform, and why the SDK differences matter for production apps.

Same call as Lab 01, one parameter different — but the code shape changes more than you'd expect, and the two SDKs diverge here more than anywhere else in the series. The prompt is deliberately long ("Be thorough") because streaming is invisible on a two-sentence answer.

## Run it

```bash
python labs/02_streaming/bedrock.py
python labs/02_streaming/vertex.py
python labs/02_streaming/foundry.py
```

Watch the terminal as it runs — text should appear word by word rather than all at once.

## Why streaming matters

Without streaming, users stare at a blank screen for the full generation time before seeing anything. For long responses (>200 tokens) the difference is visible. Most production chat UIs require streaming.

The number that matters is **time-to-first-token**, not total latency. A response that takes 8 seconds to finish but starts rendering at 0.4s feels fast. The same response delivered in one 8-second block feels broken.

## Walkthrough

### Step 1: Same setup, longer prompt

```python
PROMPT = "Walk me through how transformer attention works. Be thorough."
```

Client construction is unchanged from Lab 01 — streaming is a property of the request, not the client.

### Step 2: Open the stream

```python
# bedrock.py / vertex.py — context manager
with client.messages.stream(
    model="global.anthropic.claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": PROMPT}],
) as stream:
    ...

# foundry.py — a flag on the normal call
response = client.complete(
    model=os.getenv("AZURE_FOUNDRY_MODEL", "claude-sonnet-4-6"),
    messages=[...],
    max_tokens=512,
    stream=True,
)
```

The Anthropic SDK gives you `messages.stream()` as a separate method returning a context manager. The `with` block matters: it holds an open HTTP connection, and exiting it closes the connection and finalizes the accumulated message. Foundry instead flips `stream=True` on the same `complete()` call, which changes the return type from a response object to an iterator.

### Step 3: Consume the deltas

```python
# bedrock.py / vertex.py
for text in stream.text_stream:
    print(text, end="", flush=True)

# foundry.py
for update in response:
    if update.choices and update.choices[0].delta.content:
        print(update.choices[0].delta.content, end="", flush=True)
```

This is the real difference. `stream.text_stream` is a convenience iterator that yields **just the text** — the SDK has already parsed the server-sent event frames, filtered out the non-text events, and handed you strings. Foundry hands you raw chunks, so you dig the text out of `delta.content` yourself and guard against empty ones: the first and last chunks in a stream typically carry role and finish metadata with no content, and unguarded that prints `None` into your output.

`end=""` stops `print` adding a newline per token; `flush=True` forces the write out immediately instead of waiting for Python's line buffer to fill. Without `flush=True` the output arrives in bursts and the whole exercise looks broken.

### Step 4: Collect the final metadata

```python
# bedrock.py / vertex.py — available once the context manager exits
final = stream.get_final_message()
print(f"\n\n  input_tokens={final.usage.input_tokens}  output_tokens={final.usage.output_tokens}")
```

The SDK accumulates the deltas into a complete message as they arrive, so after the stream closes you have the same object Lab 01 returned — text, `stop_reason`, and usage. On Foundry there's no equivalent accumulator: usage arrives on the final chunk's `usage` field, and rebuilding the full text means concatenating the deltas yourself as you go.

## SDK differences worth knowing

| | Bedrock & Vertex | Foundry |
|---|---|---|
| Pattern | `client.messages.stream()` context manager + `stream.text_stream` iterator | `client.complete(stream=True)` + iterate chunks directly |
| Final metadata | `stream.get_final_message()` after context exits | Embedded in last chunk's `usage` field |
| Abstraction | Anthropic SDK normalizes the underlying SSE format | azure-ai-inference exposes raw delta chunks |

Bedrock and Vertex use identical Python because the Anthropic SDK abstracts both — that's the point of using `AnthropicBedrock` and `AnthropicVertex` over raw HTTP.

## Exercises

1. Measure time-to-first-token vs. time-to-last-token across platforms. Add `time.perf_counter()` calls around the stream iterator.
2. Accumulate the streamed text into a string and compare it to a non-streamed response to the same prompt — they should be identical.
3. Add error handling: what happens if the stream is interrupted mid-response? Wrap the iterator in a `try/except` and log partial output.
4. Remove `flush=True` from one script and re-run. The output arrives in bursts — that's Python's line buffering, and it's the first thing to check when streaming "doesn't work."

## Next

[Lab 03 — Tool Use](../03_tool_use/) stops treating the response as text to print and starts treating it as a decision to act on.
