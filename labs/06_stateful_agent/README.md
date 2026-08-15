# Lab 06 — Stateful Research Agent

**What you'll learn:** How to build a multi-step agent that plans its own work, tracks state across tool calls, and synthesizes a final output — without an external agent framework.

## Run it

```bash
python labs/06_stateful_agent/agent.py
python labs/06_stateful_agent/agent.py "the history of transformer models"
```

## How it works

The agent loop runs until Claude issues an `end_turn` response. Each iteration, Claude either calls a tool or finishes. Three tools let Claude manage its own workflow:

| Tool | What it does |
|---|---|
| `make_plan(steps)` | Sets the research agenda — a list of questions to answer |
| `record_finding(step, content)` | Records an answer for one step |
| `get_state()` | Returns the full plan + all findings so far |

State is a plain Python dict that lives in your code, not inside the model. The model reads and writes it through tools. This is the key pattern: **the model orchestrates, your code holds state**.

## Typical execution trace

```
step 1  → make_plan(["What is X?", "How does Y work?", "Where is it used?"])
            plan: 3 steps | findings: 0

step 2  → record_finding("What is X?", "X is...")
            plan: 3 steps | findings: 1

step 3  → record_finding("How does Y work?", "Y works by...")
            plan: 3 steps | findings: 2

step 4  → record_finding("Where is it used?", "It is used in...")
            plan: 3 steps | findings: 3

step 5  → [end_turn] → Research Report printed
```

## Why this matters

This is the skeleton of every production agent:
- **Plan** → decompose the goal into trackable steps
- **Execute** → work through steps, persisting results outside the model
- **Synthesize** → combine findings into a final output

Real agents extend this with external storage (databases), concurrent step execution, checkpointing for long-running jobs, and human-in-the-loop approval steps. The core loop is the same.

## Exercises

1. Add a `mark_complete()` tool that the agent calls when it's satisfied all steps are done — observe how the loop terminates differently.
2. Persist state to a JSON file so the agent can resume after a crash. Add a `--resume` flag to `agent.py`.
3. Add a `web_search(query)` tool (using the `httpx` library and a real search API) and ask the agent to research a current event.
4. Parallelize step execution: identify which findings don't depend on each other and run those `record_finding` calls concurrently using `asyncio`.
