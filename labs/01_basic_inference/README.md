# Lab 01 — Basic Inference

**What you'll learn:** How to send a prompt and receive a response on each cloud platform using the Anthropic SDK's native backends and the Azure AI Inference SDK.

## Prerequisites

| Platform | What to set up |
|---|---|
| AWS Bedrock | AWS CLI configured (`aws configure`). Claude model enabled in the Bedrock console. |
| Vertex AI | `gcloud auth application-default login`. Claude enabled in Vertex Model Garden. Set `VERTEX_PROJECT_ID` in `.env`. |
| Azure Foundry | Claude deployed in an AI Foundry project. Set `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_MODEL` in `.env`. |

## Run it

```bash
# from the repo root
python labs/01_basic_inference/bedrock.py
python labs/01_basic_inference/vertex.py
python labs/01_basic_inference/foundry.py
```

## What to look for

Each script prints the model's response and three metadata fields:
- **stop_reason / finish_reason** — `end_turn` means the model finished naturally; `max_tokens` means you hit the limit
- **input_tokens** — how many tokens your prompt consumed (affects cost)
- **output_tokens** — how many tokens the response used

The same prompt hits three different inference endpoints. The response text should be nearly identical; any variation reflects model version differences between platform deployments.

## Exercises

1. Change `PROMPT` to something longer and watch `input_tokens` grow.
2. Set `max_tokens=10` and observe the truncated `stop_reason`.
3. Add a `system` parameter to `client.messages.create` on Bedrock/Vertex and a second `SystemMessage` on Foundry — how does the response change?
