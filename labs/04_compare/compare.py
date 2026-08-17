"""
Lab 04 — Cross-Platform Comparison

Fires the same prompt at all configured platforms concurrently and prints a
side-by-side table of latency, token usage, and response excerpts.

Quick start: set ANTHROPIC_API_KEY in .env — that's enough to run the Direct
API row. Add cloud credentials to light up Bedrock, Vertex, and Foundry rows.

Usage:
  python labs/04_compare/compare.py
  python labs/04_compare/compare.py "Your custom prompt here"
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv()

DEFAULT_PROMPT = (
    "In three sentences, explain why enterprises are deploying AI on multiple cloud "
    "platforms instead of committing to a single provider."
)

console = Console()


# --------------------------------------------------------------------------- #
#  Platform callers                                                             #
# --------------------------------------------------------------------------- #

def call_direct(prompt: str) -> dict:
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    t0 = time.perf_counter()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    return {
        "platform": "Direct API",
        "latency": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "text": response.content[0].text,
    }


def call_bedrock(prompt: str) -> dict:
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
    t0 = time.perf_counter()
    response = client.messages.create(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    return {
        "platform": "AWS Bedrock",
        "latency": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "text": response.content[0].text,
    }


def call_vertex(prompt: str) -> dict:
    from anthropic import AnthropicVertex
    client = AnthropicVertex(
        project_id=os.environ["VERTEX_PROJECT_ID"],
        region=os.getenv("VERTEX_REGION", "us-east5"),
        # Unset in normal use — falls through to Application Default Credentials.
        # Set it to any string to point at the mock server (see mock_cloud/).
        access_token=os.getenv("VERTEX_ACCESS_TOKEN"),
    )
    t0 = time.perf_counter()
    response = client.messages.create(
        model="claude-3-5-sonnet-v2@20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    return {
        "platform": "Google Vertex AI",
        "latency": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "text": response.content[0].text,
    }


def call_foundry(prompt: str) -> dict:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_FOUNDRY_API_KEY"]),
    )
    t0 = time.perf_counter()
    response = client.complete(
        model=os.getenv("AZURE_FOUNDRY_MODEL", "claude-3-5-sonnet"),
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=prompt),
        ],
        max_tokens=256,
    )
    latency = time.perf_counter() - t0
    choice = response.choices[0]
    return {
        "platform": "Azure Foundry",
        "latency": latency,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "text": choice.message.content,
    }


# --------------------------------------------------------------------------- #
#  Concurrency + display                                                        #
# --------------------------------------------------------------------------- #

PLATFORMS = {
    "Direct API": (call_direct,  ["ANTHROPIC_API_KEY"]),
    "AWS Bedrock": (call_bedrock, []),                    # uses AWS profile
    "Vertex AI":   (call_vertex,  ["VERTEX_PROJECT_ID"]),
    "Foundry":     (call_foundry, ["AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_API_KEY"]),
}


def is_configured(required_vars: list[str]) -> bool:
    return all(os.getenv(v) for v in required_vars)


def run(prompt: str):
    console.rule("[bold]Claude Multi-Cloud Comparison[/bold]")
    console.print(f"\n[dim]Prompt:[/dim] {prompt}\n")

    futures = {}
    with ThreadPoolExecutor(max_workers=len(PLATFORMS)) as pool:
        for key, (fn, required) in PLATFORMS.items():
            if not is_configured(required):
                futures[key] = None  # skipped
            else:
                futures[key] = pool.submit(fn, prompt)

    results = []
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

    # --- Summary table ---
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Platform", style="bold cyan", min_width=18)
    table.add_column("Status", min_width=10)
    table.add_column("Latency", justify="right", min_width=10)
    table.add_column("Tokens in/out", justify="right", min_width=14)
    table.add_column("Response (excerpt)", min_width=40)

    for r in results:
        if r["error"]:
            table.add_row(
                r["platform"], f"[red]{r['status']}[/red]", "-", "-", "-"
            )
        else:
            excerpt = r["text"][:120].replace("\n", " ")
            if len(r["text"]) > 120:
                excerpt += "…"
            table.add_row(
                r["platform"],
                "[green]ok[/green]",
                f"{r['latency']:.2f}s",
                f"{r['input_tokens']} / {r['output_tokens']}",
                excerpt,
            )

    console.print(table)

    # --- Full responses ---
    console.rule("Full responses")
    for r in results:
        if not r["error"]:
            console.print(f"\n[bold cyan]{r['platform']}[/bold cyan]")
            console.print(r["text"])


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    run(prompt)
