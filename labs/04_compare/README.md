# Lab 04 — Cross-Platform Comparison

**What you'll learn:** How to fan out the same prompt to all three platforms concurrently and compare latency, token usage, and responses side by side. Also a useful diagnostic when debugging platform-specific behavior.

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
│ AWS Bedrock      │ ok     │   1.82s  │   42 / 98     │ Enterprises spread AI... │
│ Google Vertex AI │ ok     │   2.10s  │   42 / 103    │ Multi-cloud AI strategy… │
│ Azure Foundry    │ ok     │   1.95s  │   44 / 96     │ Organizations avoid...   │
└──────────────────┴────────┴──────────┴───────────────┴──────────────────────────┘
```

- **Latency**: wall-clock time from request to full response. All three run concurrently so total wall time ≈ slowest platform, not sum of all three.
- **Tokens in/out**: input token counts will vary slightly by platform due to different tokenizers and system prompt handling.
- **Response**: same Claude model version across all three, so variance is usually small — notable differences suggest version lag in a platform's deployment.

If a platform shows `not configured`, its required env vars are missing — only that platform is skipped.

## Exercises

1. Add a `--stream` flag that switches all three platforms to streaming mode and measures time-to-first-token instead of total latency.
2. Run with a prompt that requires long output and observe how output token count and latency scale.
3. Build a loop that runs 5 prompts and averages latency per platform — which is fastest for your workload?
4. Swap in a different model on one platform (e.g. Haiku on Bedrock) and observe the latency and response quality trade-off.
