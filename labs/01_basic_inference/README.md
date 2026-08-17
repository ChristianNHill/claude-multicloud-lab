# Lab 01 — Basic Inference

**What you'll learn:** How to send a prompt and receive a response on each cloud platform using the Anthropic SDK's native backends and the Azure AI Inference SDK.

You'll make the same call three times — once on Bedrock, once on Vertex AI, once on Azure Foundry — and compare what comes back. Same prompt, same model family, three different inference endpoints. Everything in the rest of the series is this call with more structure around it, so it's worth understanding the shape of it now.

## Prerequisites

| Platform | What to set up |
|---|---|
| AWS Bedrock | AWS CLI configured (`aws configure`). Claude model enabled in the Bedrock console. |
| Vertex AI | `gcloud auth application-default login`. Claude enabled in Vertex Model Garden. Set `VERTEX_PROJECT_ID` in `.env`. |
| Azure Foundry | Claude deployed in an AI Foundry project. Set `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_MODEL` in `.env`. |

Only have one of the three? Run that one. Each script is independent.

## Run it

```bash
# from the repo root
python labs/01_basic_inference/bedrock.py
python labs/01_basic_inference/vertex.py
python labs/01_basic_inference/foundry.py
```

## Walkthrough

Each script is about 25 lines and does the same four things. Open `bedrock.py` alongside this.

### Step 1: Load config and fix the inputs

```python
load_dotenv()

PROMPT = "Explain what a foundation model is in two sentences, for a software engineer who hasn't worked with AI before."
```

`load_dotenv()` reads `.env` into the environment so the SDKs can find credentials. The prompt is a module-level constant in all three scripts, deliberately identical — the only variable across the three runs should be the endpoint.

### Step 2: Create the client

This is the only place the three platforms meaningfully differ.

```python
# bedrock.py — credentials come from your AWS profile
client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))

# vertex.py — credentials come from Application Default Credentials
client = AnthropicVertex(
    project_id=os.environ["VERTEX_PROJECT_ID"],
    region=os.getenv("VERTEX_REGION", "us-east5"),
)

# foundry.py — explicit endpoint + key
client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_FOUNDRY_API_KEY"]),
)
```

`AnthropicBedrock` and `AnthropicVertex` are the Anthropic SDK with a different transport underneath — you write no auth code, the cloud SDK's existing credential chain handles it. Foundry takes an explicit endpoint and key because `azure-ai-inference` is a generic client for any model deployed in a Foundry project, not a Claude-specific one.

Note `os.environ[...]` vs `os.getenv(...)`: the required values use `os.environ` so a missing one fails loudly at startup rather than sending a malformed request.

### Step 3: Send the message

```python
# Bedrock / Vertex
response = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_tokens=256,
    messages=[{"role": "user", "content": PROMPT}],
)

# Foundry
response = client.complete(
    model=os.getenv("AZURE_FOUNDRY_MODEL", "claude-3-5-sonnet"),
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content=PROMPT),
    ],
    max_tokens=256,
)
```

`messages` is a list because the API is stateless — there's no session on the server, so every turn of a conversation is re-sent in full. Lab 03 is where that list starts growing.

`max_tokens` is a ceiling on the *response*, not the request. It's required on the Anthropic SDK, and it's how you bound cost per call.

The model ID is the same model in three naming conventions:

| Platform | Model ID | Convention |
|---|---|---|
| Bedrock | `anthropic.claude-3-5-sonnet-20241022-v2:0` | vendor prefix, date, version suffix |
| Vertex | `claude-3-5-sonnet-v2@20241022` | `@version` suffix |
| Foundry | whatever you named the deployment | your choice at deploy time |

### Step 4: Read the response and its metadata

```python
# Bedrock / Vertex
print(f"[Bedrock] {response.content[0].text}")
print(f"  stop_reason={response.stop_reason}  input_tokens={response.usage.input_tokens}  output_tokens={response.usage.output_tokens}")

# Foundry
choice = response.choices[0]
print(f"[Foundry] {choice.message.content}")
print(f"  finish_reason={choice.finish_reason}  prompt_tokens={response.usage.prompt_tokens}  completion_tokens={response.usage.completion_tokens}")
```

`response.content` is a **list of blocks**, not a string — that's why it's `content[0].text`. A plain text answer is one block, but a response can also contain `tool_use` blocks, which is exactly what Lab 03 relies on. Getting used to indexing into blocks now saves confusion later.

## What to look for

Each script prints the model's response and three metadata fields:

- **stop_reason / finish_reason** — `end_turn` means the model finished naturally; `max_tokens` means you hit the limit
- **input_tokens** — how many tokens your prompt consumed (affects cost)
- **output_tokens** — how many tokens the response used

The vocabulary drifts between the two SDKs, which is the first thing to internalize about running Claude multi-cloud:

| Concept | Anthropic SDK (Bedrock, Vertex) | Azure AI Inference (Foundry) |
|---|---|---|
| Call | `client.messages.create(...)` | `client.complete(...)` |
| Response text | `response.content[0].text` | `response.choices[0].message.content` |
| Why it stopped | `stop_reason` (`end_turn`) | `finish_reason` (`stop`) |
| Token counts | `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` |
| System prompt | `system=` parameter | `SystemMessage` in the message list |

The same prompt hits three different inference endpoints. The response text should be nearly identical; any variation reflects model version differences between platform deployments, not the platforms themselves.

## Exercises

1. Change `PROMPT` to something longer and watch `input_tokens` grow.
2. Set `max_tokens=10` and observe the truncated `stop_reason`.
3. Add a `system` parameter to `client.messages.create` on Bedrock/Vertex and a second `SystemMessage` on Foundry — how does the response change?
4. Print the whole `response` object instead of just `content[0].text`. Everything in Lab 03 is already visible in that structure.

## Next

[Lab 02 — Streaming](../02_streaming/) takes this same call and returns tokens as they're generated, instead of making the user wait for the whole response.
