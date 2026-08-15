"""
Lab 02 — Streaming: AWS Bedrock

Streams tokens as they're generated instead of waiting for the full response.
The Anthropic SDK handles the underlying Bedrock InvokeModelWithResponseStream
call; you get the same stream interface as the direct API.
"""

import os
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

PROMPT = "Walk me through how transformer attention works. Be thorough."

def run():
    client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))

    print("[Bedrock] ", end="", flush=True)

    with client.messages.stream(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=512,
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    # Final message is available after the stream closes
    final = stream.get_final_message()
    print(f"\n\n  input_tokens={final.usage.input_tokens}  output_tokens={final.usage.output_tokens}")

if __name__ == "__main__":
    run()
