# Lab 04 — Cross-Platform Comparison

**What you'll learn:** How to fan out the same prompt to all platforms concurrently and compare latency, token usage, and responses side by side. Also a useful diagnostic when debugging platform-specific behavior.

This is the showpiece. One prompt, every platform you've configured, answered simultaneously with the numbers next to each other. It's also the first lab where the interesting code isn't the Claude call — it's the concurrency and the graceful degradation around it.

Only have an `ANTHROPIC_API_KEY`? It still runs. Unconfigured platforms report themselves and the rest continue.

## Run it

```bash
# Default prompt
python labs/04_compare/compare.py

# Custom prompt
python labs/04_compare/compare.py "What is chain-of-thought prompting?"
```

## What the output means

```
┌──────────────────┬────────┬──────────┬───────────────┬──────────────────────────┐
│ Platform         │ Status │  Latency │ Tokens in/out │ Response (excerpt)       │
├──────────────────┼────────┼──────────┼───────────────┼──────────────────────────┤
│ Direct API       │ ok     │   0.94s  │   42 / 91     │ Enterprises hedge...     │
│ AWS Bedrock      │ ok     │   1.82s  │   42 / 98     │ Enterprises spread AI... │
│ Google Vertex AI │ ok     │   2.10s  │   42 / 103    │ Multi-cloud AI strategy… │
│ Azure Foundry    │ ok     │   1.95s  │   44 / 96     │ Organizations avoid...   │
└──────────────────┴────────┴──────────┴───────────────┴──────────────────────────┘
```

- **Latency**: wall-clock time from request to full response. All platforms run concurrently so total wall time ≈ slowest platform, not the sum.
- **Tokens in/out**: input token counts vary slightly by platform due to different tokenizers and system prompt handling.
- **Response**: same Claude model version across platforms, so variance is usually small — notable differences suggest version lag in a platform's deployment.

If a platform shows `not configured`, its required env vars are missing — only that platform is skipped.

Latency here measures *your network path to that region*, not the platform's inherent speed. Run it from a different location and the ranking changes.

## Walkthrough

### Step 1: One caller per platform, all returning the same shape

```python
def call_bedrock(prompt: str) -> dict:
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
    t0 = time.perf_counter()
    response = client.messages.create(...)
    latency = time.perf_counter() - t0
    return {
        "platform": "AWS Bedrock",
        "latency": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "text": response.content[0].text,
    }
```

Four near-identical functions, one per platform. The duplication is deliberate — each one shows its platform's real API surface, which is the point of the lab. Factoring them into one adapter would hide exactly what you're here to compare.

Two details worth copying:

- **The import is inside the function.** A missing `azure-ai-inference` install then breaks only the Foundry row, not the whole script at import time.
- **`t0` starts after the client is constructed.** You're timing the request, not SDK setup or credential resolution — those happen once in a real app and would otherwise pollute the first measurement.

Every caller returns the same dict keys, so downstream code never branches on platform. That normalization is why the Foundry caller reads `usage.prompt_tokens` and stores it under `input_tokens`.

### Step 2: Declare what each platform needs

```python
PLATFORMS = {
    "Direct API": (call_direct,  ["ANTHROPIC_API_KEY"]),
    "AWS Bedrock": (call_bedrock, []),                    # uses AWS profile
    "Vertex AI":   (call_vertex,  ["VERTEX_PROJECT_ID"]),
    "Foundry":     (call_foundry, ["AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_API_KEY"]),
}

def is_configured(required_vars: list[str]) -> bool:
    return all(os.getenv(v) for v in required_vars)
```

The registry pairs each caller with the env vars it can't run without, so "is this configured?" is a data question rather than a try/except. Bedrock's list is empty because credentials come from the AWS profile or instance metadata — there's no single variable to check, so it's attempted and allowed to fail into the error path.

### Step 3: Fan out

```python
futures = {}
with ThreadPoolExecutor(max_workers=3) as pool:
    for key, (fn, required) in PLATFORMS.items():
        if not is_configured(required):
            futures[key] = None  # skipped
        else:
            futures[key] = pool.submit(fn, prompt)
```

Threads, not `asyncio` — the SDK calls are blocking HTTP, and a thread pool needs no async plumbing through the rest of the script. For four concurrent network calls that's the right tool.

Submit everything first, collect afterward. Calling `.result()` inside this loop would serialize the whole thing and quietly turn the benchmark into a sequential one — a mistake that produces plausible-looking numbers, which is the worst kind.

Skipped platforms get a `None` placeholder rather than being dropped, so they still appear as a row in the output. Missing configuration should be visible, not silent.

### Step 4: Collect, isolating failures

```python
for key, future in futures.items():
    if future is None:
        results.append({"platform": key, "status": "not configured", "error": True})
        continue
    try:
        data = future.result()
        data["status"] = "ok"
        data["error"] = False
        results.append(data)
    except Exception as exc:
        results.append({"platform": key, "status": f"error: {exc}", "error": True})
```

`future.result()` re-raises whatever the thread raised, on this thread. The `try` per platform is what keeps one expired credential from taking down the other three rows — for a comparison tool, partial results beat an exception every time.

### Step 5: Render

`rich` handles the table. The only judgment call is truncating the excerpt to 120 characters with the full text printed below the table — the table is for scanning the numbers, the section under it is for actually reading the answers.

## Exercises

1. Add a `--stream` flag that switches all platforms to streaming mode and measures time-to-first-token instead of total latency.
2. Run with a prompt that requires long output and observe how output token count and latency scale.
3. Build a loop that runs 5 prompts and averages latency per platform — one sample is noise, and you'll see the spread is wider than the gap between platforms.
4. Swap in a different model on one platform (e.g. Haiku on Bedrock) and observe the latency and response quality trade-off.
5. Add a cost column. Multiply tokens by each platform's published per-token price — the ranking often differs from the latency ranking.

## Next

[Lab 05 — MCP](../05_mcp/) moves the tools out of your script entirely, into a separate process the client discovers at runtime.
