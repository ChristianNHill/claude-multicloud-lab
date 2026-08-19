# Claude multi-cloud lab

A hands-on lab series for deploying and building with Claude across AWS Bedrock, Google Vertex AI, and Azure AI Foundry. Six progressive labs cover the Claude Platform stack, from first API call to MCP-connected agents.

Built as workshop-ready teaching material: each lab has a README with step-by-step instructions, a concept walkthrough, and exercises to extend the work.

I built this to learn more about AWS Bedrock, Google Vertex AI, and Azure AI Foundry. Hopefully you find it useful!!

---

## Labs

| # | Lab | Concepts | Platforms |
|---|-----|----------|-----------|
| 01 | [Basic Inference](labs/01_basic_inference/) | Messages API, token usage, stop reasons | Bedrock · Vertex · Foundry |
| 02 | [Streaming](labs/02_streaming/) | Token streaming, time-to-first-token, SDK differences | Bedrock · Vertex · Foundry |
| 03 | [Tool Use](labs/03_tool_use/) | Agentic loops, function calling, tool result handling | Bedrock · Vertex · Foundry |
| 04 | [Cross-Platform Compare](labs/04_compare/) | Concurrent requests, latency benchmarking, rich terminal output | Bedrock · Vertex · Foundry |
| 05 | [MCP](labs/05_mcp/) | MCP server + client, tool discovery at runtime, stdio transport | Bedrock |
| 06 | [Stateful Agent](labs/06_stateful_agent/) | Multi-step orchestration, external state, plan-execute-synthesize pattern | Bedrock |

---

## Quick start

```bash
git clone https://github.com/ChristianNHill/claude-multicloud-lab
cd claude-multicloud-lab
pip install -r requirements.txt
cp .env.example .env
```

### Credentials

**`.env.example` points at the mock server out of the box, so don't add your own keys yet.** Start the mock and every lab runs offline:

```bash
python mock_cloud/server.py          # terminal 1
python labs/01_basic_inference/bedrock.py   # terminal 2
```

It impersonates all four endpoints: real HTTP, real streaming, real tool loops, canned responses. Comment the mock block out of `.env` to go back to real platforms; the lab scripts are unchanged either way. See [mock_cloud/](mock_cloud/). Good for workshops where attendees are still waiting on Model Access approvals.

**Fastest path with a real model, one key.** Set `ANTHROPIC_API_KEY` in `.env` (get one at [console.anthropic.com](https://console.anthropic.com)). Labs 04-06 run immediately with only this.

**Add cloud platforms** as you get access. Each one lights up an additional row in the Lab 04 comparison table:

| Platform | What to set up |
|---|---|
| AWS Bedrock | `aws configure` + enable Claude in Bedrock console → Model access |
| Google Vertex AI | `gcloud auth application-default login` + `VERTEX_PROJECT_ID` in `.env` + enable Claude in Model Garden |
| Azure Foundry | Deploy Claude via Azure Marketplace → set `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_MODEL` in `.env` |

Labs 01-03 show all three cloud platforms side by side, run the script for whichever platform you've configured, since the others raise a credentials error. Lab 04 is the one that skips unconfigured platforms and keeps going.

---

## The showpiece demo

Run this and watch all three platforms answer the same prompt side by side, with latency and token counts:

```bash
python labs/04_compare/compare.py
python labs/04_compare/compare.py "Explain chain-of-thought prompting in two sentences."
```

---

## Lab progression

The labs are ordered by concept depth. Each one builds on the last:

```
Messages API (01)
    └─► Streaming (02): same API, streaming variant
        └─► Tool Use (03): agentic loop, multi-turn
            └─► Compare (04): all platforms, concurrent
                └─► MCP (05): tools in a separate process, discovered at runtime
                    └─► Agent (06): multi-step planning, external state
```

Labs 01-04 run on all three platforms. Labs 05-06 focus on Bedrock and demonstrate patterns that apply equally on Vertex and Foundry.

---

## SDK notes

Labs 01-03 use the Anthropic SDK's first-party cloud backends:
- `AnthropicBedrock`: same API as `Anthropic`, no extra auth code
- `AnthropicVertex`: same API, uses Application Default Credentials
- `azure-ai-inference`: OpenAI-compatible, slightly different tool call format (documented in each lab)

Lab 05 uses the `mcp` package for the MCP server and client session.

---

## Using as a workshop

Each lab is self-contained: install deps, set env vars, run the script. The README in each lab includes:
- What the lab teaches and why it matters
- Step-by-step run instructions
- Annotated explanation of the output
- 3-4 exercises to extend the lab

Labs 03, 05, and 06 are the most discussion-rich. They anchor a live session on agentic patterns well.
