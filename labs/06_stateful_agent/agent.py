"""
Lab 06 — Stateful Research Agent

Demonstrates multi-step agent orchestration: the agent plans its own work,
executes each step by recording findings, and synthesizes a final report.
State lives outside the model — the model reads and writes it through tools.

This pattern scales to production agents (memory, task queues, checkpointing)
without requiring an external framework.

Quick start: just needs ANTHROPIC_API_KEY in .env.
Uses direct API if ANTHROPIC_API_KEY is set, otherwise falls back to AWS Bedrock.

Usage:
  python labs/06_stateful_agent/agent.py
  python labs/06_stateful_agent/agent.py "quantum computing"
"""

import json
import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich import box

load_dotenv()

console = Console()


# --------------------------------------------------------------------------- #
#  State                                                                        #
# --------------------------------------------------------------------------- #

def new_state(topic: str) -> dict:
    return {
        "topic": topic,
        "plan": [],        # list of step strings set by make_plan
        "findings": {},    # step -> finding text
        "complete": False,
    }


# --------------------------------------------------------------------------- #
#  Tools                                                                        #
# --------------------------------------------------------------------------- #

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
    {
        "name": "record_finding",
        "description": "Record a finding for a specific research step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": "The research step this finding addresses (must match a step from make_plan).",
                },
                "content": {
                    "type": "string",
                    "description": "The finding — what you know about this step.",
                },
            },
            "required": ["step", "content"],
        },
    },
    {
        "name": "get_state",
        "description": "Retrieve the current research state: the plan and all findings recorded so far.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


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

    return f"Unknown tool: {name}"


# --------------------------------------------------------------------------- #
#  System prompt                                                                #
# --------------------------------------------------------------------------- #

def system_prompt(topic: str) -> str:
    return f"""You are a research agent investigating: "{topic}"

Work through this process in order:
1. Call make_plan with 3–5 specific research questions to investigate.
2. For each step in your plan, call record_finding with what you know about that step.
3. Once all steps have findings, write a final synthesis as your text response.

Use get_state at any point to review your progress. Do not skip steps."""


# --------------------------------------------------------------------------- #
#  Agent loop                                                                   #
# --------------------------------------------------------------------------- #

def _make_client():
    if os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        return Anthropic(), "claude-haiku-4-5-20251001"
    else:
        from anthropic import AnthropicBedrock
        return AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1")), \
               "anthropic.claude-3-5-sonnet-20241022-v2:0"


def run(topic: str):
    console.rule(f"[bold]Stateful Research Agent[/bold]")
    console.print(f"\n[dim]Topic:[/dim] {topic}\n")

    client, model = _make_client()
    state = new_state(topic)

    messages = [{"role": "user", "content": f"Research the topic: {topic}"}]
    iteration = 0
    max_iterations = 20  # guard against runaway loops

    try:
        while iteration < max_iterations:
            iteration += 1

            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt(topic),
                tools=TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            # Collect tool calls from content regardless of stop_reason —
            # models can return tool_use blocks even when stop_reason is max_tokens.
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if tool_use_blocks:
                tool_results = []
                for block in tool_use_blocks:
                    console.print(f"  [dim]step {iteration}[/dim] → [cyan]{block.name}[/cyan]({_summarize(block.input)})")
                    output = dispatch_tool(block.name, block.input, state)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })

                console.print(f"    [dim]plan: {len(state['plan'])} steps | findings: {len(state['findings'])}[/dim]")
                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        console.print(Panel(block.text, title="[bold green]Research Report[/bold green]", box=box.ROUNDED))
                break

        if iteration >= max_iterations:
            console.print("[red]Max iterations reached — agent did not complete.[/red]")

    except Exception as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        console.print("[dim](Configure AWS credentials to run the full agent loop)[/dim]")


def _summarize(inputs: dict) -> str:
    """Compact one-line representation of tool inputs for display."""
    s = json.dumps(inputs)
    return s[:80] + "…" if len(s) > 80 else s


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "large language models"
    run(topic)
