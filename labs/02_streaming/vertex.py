"""
Lab 02 — Streaming: Google Vertex AI

Identical stream interface to the direct Anthropic API — the SDK abstracts
Vertex's underlying SSE format.
"""

import os
from dotenv import load_dotenv
from anthropic import AnthropicVertex

load_dotenv()

PROMPT = "Walk me through how transformer attention works. Be thorough."

def run():
    client = AnthropicVertex(
        project_id=os.environ["VERTEX_PROJECT_ID"],
        region=os.getenv("VERTEX_REGION", "us-east5"),
        # Unset in normal use — falls through to Application Default Credentials.
        # Set it to any string to point at the mock server (see mock_cloud/).
        access_token=os.getenv("VERTEX_ACCESS_TOKEN"),
    )

    print("[Vertex AI] ", end="", flush=True)

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    final = stream.get_final_message()
    print(f"\n\n  input_tokens={final.usage.input_tokens}  output_tokens={final.usage.output_tokens}")

if __name__ == "__main__":
    run()
