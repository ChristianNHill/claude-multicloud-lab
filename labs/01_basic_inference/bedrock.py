"""
Lab 01: Basic Inference: AWS Bedrock

Uses the Anthropic SDK's first-party Bedrock backend (AnthropicBedrock).
Credentials come from your AWS profile; no extra auth code needed.
"""

import os
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

PROMPT = "Explain what a foundation model is in two sentences, for a software engineer who hasn't worked with AI before."

def run():
    client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))

    response = client.messages.create(
        model="global.anthropic.claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": PROMPT}],
    )

    print(f"[Bedrock] {response.content[0].text}")
    print(f"  stop_reason={response.stop_reason}  input_tokens={response.usage.input_tokens}  output_tokens={response.usage.output_tokens}")

if __name__ == "__main__":
    run()
