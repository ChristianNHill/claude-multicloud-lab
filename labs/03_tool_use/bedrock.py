"""
Lab 03 — Tool Use: AWS Bedrock

Demonstrates an agentic loop: Claude decides when to call a tool, we execute it,
and feed the result back. The loop runs until Claude stops requesting tools.

Tool: get_weather (mocked — swap for a real API in production)
"""

import os
import json
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

# --- Tool definition (sent to the model) ---

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

# --- Tool implementation (runs on our side) ---

def get_weather(city: str) -> dict:
    # Mocked. Replace with requests.get("https://api.openweathermap.org/...") in production.
    mock_data = {
        "San Francisco": {"temp_c": 17, "condition": "Partly cloudy"},
        "New York":      {"temp_c": 22, "condition": "Sunny"},
        "Seattle":       {"temp_c": 14, "condition": "Overcast"},
    }
    return mock_data.get(city, {"temp_c": 20, "condition": "Unknown"})

def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":
        result = get_weather(**inputs)
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {name}")

# --- Agentic loop ---

def run():
    client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))

    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco and Seattle? Which is warmer?"},
    ]

    print("[Bedrock tool use]\n")

    while True:
        response = client.messages.create(
            model="global.anthropic.claude-sonnet-4-6",
            max_tokens=512,
            tools=TOOLS,
            messages=messages,
        )

        # Append the assistant turn (may include text + tool calls)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Model is done — extract and print final text
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break

        if response.stop_reason == "tool_use":
            # Execute each requested tool and collect results
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

            # Feed results back as a user turn
            messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    run()
