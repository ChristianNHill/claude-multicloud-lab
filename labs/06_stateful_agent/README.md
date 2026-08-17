# Lab 06 — Stateful Research Agent

**What you'll learn:** How to build a multi-step agent that plans its own work, tracks state across tool calls, and synthesizes a final output — without an external agent framework.

Every previous lab gave the model tools that touch the outside world: fetch weather, search docs, do arithmetic. This one gives it tools that manipulate *its own workflow* — make a plan, record a finding, check progress. That's the whole trick behind agent frameworks, and it's about 200 lines without one.

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

## Walkthrough

### Step 1: State is a dict, and it lives in your process

```python
def new_state(topic: str) -> dict:
    return {
        "topic": topic,
        "plan": [],        # list of step strings set by make_plan
        "findings": {},    # step -> finding text
        "complete": False,
    }
```

That's the entire state layer. No database, no framework, no memory abstraction — a dict you own.

The reason this matters: the model's context window is not storage. It's lossy, it's capped, and on a long enough run the early turns fall out of it. Anything the agent must not forget lives here instead, and the model retrieves it deliberately via `get_state`.

### Step 2: Tools that operate on the agent's own process

```python
TOOLS = [
    {
        "name": "make_plan",
        "description": (
            "Set the research plan. Call this first with a list of specific questions "
            "or subtopics to investigate. Each step will be addressed in turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of research questions or subtopics to investigate.",
                }
            },
            "required": ["steps"],
        },
    },
    ...
]
```

Same JSON Schema format as Lab 03, but note `"type": "array"` with typed `items` — schemas nest, and the SDK validates the whole structure before you see it. `make_plan` gets a list of strings or it doesn't get called.

Nothing about these tools reaches outside the process. `make_plan` writes to a dict. That's enough to give the model a durable notion of "what I set out to do," which is the difference between an agent and a chatbot in a loop.

### Step 3: Tool results are feedback, not just acknowledgements

```python
def dispatch_tool(name: str, inputs: dict, state: dict) -> str:
    if name == "make_plan":
        state["plan"] = inputs["steps"]
        return f"Plan set with {len(inputs['steps'])} steps: {inputs['steps']}"

    if name == "record_finding":
        state["findings"][inputs["step"]] = inputs["content"]
        remaining = [s for s in state["plan"] if s not in state["findings"]]
        return f"Finding recorded. Steps remaining: {len(remaining)}"

    if name == "get_state":
        return json.dumps(state, indent=2)
```

`dispatch_tool` takes `state` as a parameter rather than reaching for a global, so the agent's state is passed explicitly down the call chain — that's what would let you run two agents concurrently in one process.

The return strings do real work. `"Steps remaining: 2"` goes back into the conversation and tells the model, without any prompting, that it isn't finished. Change it to `"OK"` and the agent gets noticeably worse at completing its plan. **Tool results are the steering signal for the next turn** — write them for a reader who needs to decide what to do next.

### Step 4: The system prompt is the procedure

```python
def system_prompt(topic: str) -> str:
    return f"""You are a research agent investigating: "{topic}"

Work through this process in order:
1. Call make_plan with 3–5 specific research questions to investigate.
2. For each step in your plan, call record_finding with what you know about that step.
3. Once all steps have findings, write a final synthesis as your text response.

Use get_state at any point to review your progress. Do not skip steps."""
```

Tool descriptions say what each tool does; the system prompt says what order to use them in. Both are needed. A numbered procedure with an explicit terminal condition ("write a final synthesis as your text response") is what makes the run finish rather than loop.

### Step 5: Read the content blocks, don't trust stop_reason

```python
# Collect tool calls from content regardless of stop_reason —
# models can return tool_use blocks even when stop_reason is max_tokens.
tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

if tool_use_blocks:
    ...
elif response.stop_reason == "end_turn":
    ...
```

This is the production-hardened version of Lab 03's loop, and the difference is worth understanding. Lab 03 branches on `stop_reason == "tool_use"`. That's correct until a response hits the token ceiling mid-turn: `stop_reason` comes back `max_tokens` while the content still contains complete `tool_use` blocks. Branch on `stop_reason` alone and those calls are silently dropped, leaving the model waiting for results that never arrive.

Checking the content first and using `stop_reason` only as the exit condition handles both cases.

### Step 6: Bound the loop

```python
iteration = 0
max_iterations = 20  # guard against runaway loops

while iteration < max_iterations:
    iteration += 1
    ...

if iteration >= max_iterations:
    console.print("[red]Max iterations reached — agent did not complete.[/red]")
```

Lab 03's `while True` is fine for a demo with one tool. An agent that decides its own next step needs a ceiling, because a model that gets confused about whether it's finished will happily call `get_state` forever — and every iteration is a billed API call. The guard is three lines and it's the difference between a bug and an invoice.

## Why this matters

This is the skeleton of every production agent:

- **Plan** → decompose the goal into trackable steps
- **Execute** → work through steps, persisting results outside the model
- **Synthesize** → combine findings into a final output

Real agents extend this with external storage (databases), concurrent step execution, checkpointing for long-running jobs, and human-in-the-loop approval steps. The core loop is the same.

Combine it with Lab 05 and the shape of a real system appears: workflow tools local, capability tools behind MCP servers, state in your own store.

## Exercises

1. Add a `mark_complete()` tool that the agent calls when it's satisfied all steps are done — observe how the loop terminates differently.
2. Persist state to a JSON file so the agent can resume after a crash. Add a `--resume` flag to `agent.py`.
3. Add a `web_search(query)` tool (using the `httpx` library and a real search API) and ask the agent to research a current event.
4. Parallelize step execution: identify which findings don't depend on each other and run those `record_finding` calls concurrently using `asyncio`.
5. Change `record_finding`'s return string to just `"OK"` and re-run a few times. Watch the completion rate drop — that's how much work the feedback string was doing.

## Next

That's the series. The natural extensions from here: swap the mocked tools for real ones, put the agent behind an API, and add evals so you can tell whether a prompt change made it better. See the [repo README](../../README.md) for the full lab progression.
