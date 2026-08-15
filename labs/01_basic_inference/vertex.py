"""
Lab 01 — Basic Inference: Google Vertex AI

Uses the Anthropic SDK's first-party Vertex backend (AnthropicVertex).
Auth uses Application Default Credentials (ADC): run `gcloud auth application-default login` once.
"""

import os
from dotenv import load_dotenv
from anthropic import AnthropicVertex

load_dotenv()

PROMPT = "Explain what a foundation model is in two sentences, for a software engineer who hasn't worked with AI before."

def run():
    client = AnthropicVertex(
        project_id=os.environ["VERTEX_PROJECT_ID"],
        region=os.getenv("VERTEX_REGION", "us-east5"),
    )

    # Vertex model IDs use @version suffixes instead of date suffixes
    response = client.messages.create(
        model="claude-3-5-sonnet-v2@20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": PROMPT}],
    )

    print(f"[Vertex AI] {response.content[0].text}")
    print(f"  stop_reason={response.stop_reason}  input_tokens={response.usage.input_tokens}  output_tokens={response.usage.output_tokens}")

if __name__ == "__main__":
    run()
