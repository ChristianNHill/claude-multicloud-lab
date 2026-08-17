"""
Lab 03 — Tool Use: Google Vertex AI

Identical agentic loop to the Bedrock version — AnthropicVertex uses the same
messages API so tool use code is copy-portable across platforms.
"""

import os
import json
from dotenv import load_dotenv
from anthropic import AnthropicVertex

load_dotenv()

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature in Celsius and a condition string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'San Francisco'"},
            },
            "required": ["city"],
        },
    }
]

def get_weather(city: str) -> dict:
    mock_data = {
        "San Francisco": {"temp_c": 17, "condition": "Partly cloudy"},
        "New York":      {"temp_c": 22, "condition": "Sunny"},
        "Seattle":       {"temp_c": 14, "condition": "Overcast"},
    }
    return mock_data.get(city, {"temp_c": 20, "condition": "Unknown"})

def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":
        return json.dumps(get_weather(**inputs))
    raise ValueError(f"Unknown tool: {name}")

def run():
    client = AnthropicVertex(
        project_id=os.environ["VERTEX_PROJECT_ID"],
        region=os.getenv("VERTEX_REGION", "us-east5"),
        # Unset in normal use — falls through to Application Default Credentials.
        # Set it to any string to point at the mock server (see mock_cloud/).
        access_token=os.getenv("VERTEX_ACCESS_TOKEN"),
    )

    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco and Seattle? Which is warmer?"},
    ]

    print("[Vertex AI tool use]\n")

    while True:
        response = client.messages.create(
            model="claude-3-5-sonnet-v2@20241022",
            max_tokens=512,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  -> calling {block.name}({block.input})")
                    output = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
            messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    run()
