# Mock Cloud: run every lab offline

**What this is:** A local server that impersonates AWS Bedrock, Google Vertex AI, Azure AI Foundry, and the direct Anthropic API. Point the labs at it and all six run with no cloud account, no credentials, and no network.

Built for workshops. The most common way a session stalls is twenty minutes of `aws configure` and Model Access approvals before anyone writes a line of code. With this running, attendees clone the repo and start at Lab 01 immediately, then swap to real platforms as their access comes through, the lab code is identical either way.

## What's real and what isn't

| Real | Not real |
|---|---|
| HTTP request/response over a socket | The model: every response is canned |
| Each platform's actual URL routing and payload shape | The content, which won't address your prompt |
| SSE streaming (Vertex, Foundry, Direct) | Token counts, which are estimated from word count |
| AWS binary event-stream framing (Bedrock) | Latency, which is a fixed per-platform sleep |
| `tool_use` blocks, `tool_use_id` correlation, multi-turn loops | |
| `stop_reason` / `finish_reason` transitions | |

The point is that everything *your code* does is real. The SDKs are not stubbed or monkeypatched, `AnthropicBedrock` decodes genuine event-stream frames here, and Lab 03's agentic loop runs a genuine multi-turn tool exchange. Only the thing on the far end is fake.

## Run it

```bash
# Terminal 1
python mock_cloud/server.py

# Terminal 2: any lab, unchanged
python labs/01_basic_inference/bedrock.py
python labs/04_compare/compare.py
```

Options: `--port 9000`, `--no-delay` (drop the simulated latency), `--selftest` (assert the reply logic and exit).

Binds `127.0.0.1` only. To share one server with a room, add a `--host` argument and pass it to `ThreadingHTTPServer`: it's a one-line change, left out because nobody has asked for it.

## Point the labs at it

Add to `.env`. The credentials have to be *present*, because the SDKs won't construct a client without them, but they're never checked, so any string works.

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8787/direct
ANTHROPIC_BEDROCK_BASE_URL=http://127.0.0.1:8787/bedrock
ANTHROPIC_VERTEX_BASE_URL=http://127.0.0.1:8787/vertex/v1
AZURE_FOUNDRY_ENDPOINT=http://127.0.0.1:8787/foundry

ANTHROPIC_API_KEY=mock
AWS_ACCESS_KEY_ID=mock
AWS_SECRET_ACCESS_KEY=mock
AWS_REGION=us-east-1
VERTEX_PROJECT_ID=mock-project
VERTEX_ACCESS_TOKEN=mock
AZURE_FOUNDRY_API_KEY=mock
```

Comment those four base URLs out and the same labs hit the real platforms. That's the whole switch.

The Anthropic SDK reads `ANTHROPIC_BEDROCK_BASE_URL` and `ANTHROPIC_VERTEX_BASE_URL` itself: no lab code change was needed for them. `VERTEX_ACCESS_TOKEN` is the one exception: `AnthropicVertex` otherwise calls `google.auth.default()` and fails without real ADC, so the Vertex scripts pass `access_token=os.getenv("VERTEX_ACCESS_TOKEN")`. Leave it unset in normal use and the SDK falls back to Application Default Credentials exactly as before.

## Routes

| Platform | Path | Format |
|---|---|---|
| Bedrock | `POST /bedrock/model/{model}/invoke` | Anthropic messages JSON |
| Bedrock | `POST /bedrock/model/{model}/invoke-with-response-stream` | AWS `vnd.amazon.eventstream` frames |
| Vertex | `POST /vertex/v1/projects/{p}/locations/{r}/publishers/anthropic/models/{m}:rawPredict` | Anthropic messages JSON |
| Vertex | `...:streamRawPredict` | SSE |
| Foundry | `POST /foundry/chat/completions` | OpenAI chat completions (+ SSE when `stream=true`) |
| Direct | `POST /direct/v1/messages` | Anthropic messages JSON (+ SSE) |

Bedrock is the awkward one: `invoke-with-response-stream` doesn't return SSE, it returns binary event-stream frames: a 12-byte prelude with its own CRC, typed headers, a base64 payload, and a trailing CRC. The Anthropic SDK decodes these with botocore, so the mock has to encode them correctly or the stream fails to parse. `aws_frame()` in `server.py` does that in about ten lines of `struct` and `zlib`, and `--selftest` verifies the output round-trips through botocore's own `EventStreamBuffer`.

## How it decides what to "say"

The API is stateless: the full conversation arrives on every request, so the server can work out what should happen next without keeping any state of its own:

- **No tools in the request** → return canned text, matched to the prompt by keyword.
- **Tools, and no tool results in the history yet** → emit `tool_use` blocks. Lab 03's two-city prompt produces two calls; Lab 05 gets one `search_docs` and one `run_calculation`, with the arithmetic pulled out of the prompt by regex.
- **Tools, and results already returned** → finish with `end_turn`.
- **Lab 06's workflow tools** → `make_plan` on the first turn, then one `record_finding` per turn for each step not yet recorded, then synthesize. The plan is read back out of the earlier `make_plan` block in the history.

That last branch is why the agent terminates instead of looping: the server derives progress from the same message history the model would.

## Extending it

Adding a canned answer is a dict entry in `CANNED`, keyed by a substring of the prompt. Teaching a new tool is a branch in `decide_tool_calls`. If you add either, run `--selftest`: it covers the text path, both tool paths, agent termination, Bedrock frame encoding, and the Foundry round trip.

## Limits

Deliberately not built: authentication (signatures and keys are ignored), rate limits, error injection, and model-accurate tokenization. Error injection is the most useful next addition: a flag that returns 429s or truncated streams would make the Lab 02 and Lab 04 error-handling exercises real instead of hypothetical.
