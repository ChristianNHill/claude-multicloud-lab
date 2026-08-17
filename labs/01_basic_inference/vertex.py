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
        # Unset in normal use — falls through to Application Default Credentials.
        # Set it to any string to point at the mock server (see mock_cloud/).
        access_token=os.getenv("VERTEX_ACCESS_TOKEN"),
    )

    # Vertex takes the bare Anthropic model ID — no vendor prefix, no version suffix
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": PROMPT}],
    )

    print(f"[Vertex AI] {response.content[0].text}")
    print(f"  stop_reason={response.stop_reason}  input_tokens={response.usage.input_tokens}  output_tokens={response.usage.output_tokens}")

if __name__ == "__main__":
    run()
