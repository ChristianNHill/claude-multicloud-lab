"""
Lab 03 — Tool Use: Azure AI Foundry

Azure AI Foundry uses an OpenAI-compatible tool calling format (ChatCompletionsFunctionToolDefinition).
The agentic loop is the same concept but the SDK types differ from the Anthropic SDK.
"""

import os
import json
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ChatCompletionsFunctionToolDefinition,
    FunctionDefinition,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# OpenAI-compatible tool definition format
TOOLS = [
    ChatCompletionsFunctionToolDefinition(
        function=FunctionDefinition(
            name="get_weather",
            description="Get the current weather for a city. Returns temperature in Celsius and a condition string.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'San Francisco'"},
                },
                "required": ["city"],
            },
        )
    )
]

def get_weather(city: str) -> dict:
    mock_data = {
        "San Francisco": {"temp_c": 17, "condition": "Partly cloudy"},
        "New York":      {"temp_c": 22, "condition": "Sunny"},
        "Seattle":       {"temp_c": 14, "condition": "Overcast"},
    }
    return mock_data.get(city, {"temp_c": 20, "condition": "Unknown"})

def dispatch_tool(name: str, args_json: str) -> str:
    inputs = json.loads(args_json)
    if name == "get_weather":
        return json.dumps(get_weather(**inputs))
    raise ValueError(f"Unknown tool: {name}")

def run():
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_FOUNDRY_API_KEY"]),
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather like in San Francisco and Seattle? Which is warmer?"),
    ]

    print("[Foundry tool use]\n")

    while True:
        response = client.complete(
            model=os.getenv("AZURE_FOUNDRY_MODEL", "claude-3-5-sonnet"),
            messages=messages,
            tools=TOOLS,
            max_tokens=512,
        )

        choice = response.choices[0]
        messages.append(AssistantMessage(
            content=choice.message.content,
            tool_calls=choice.message.tool_calls,
        ))

        if choice.finish_reason == "stop":
            print(choice.message.content)
            break

        if choice.finish_reason == "tool_calls":
            for call in choice.message.tool_calls:
                fn = call.function
                print(f"  -> calling {fn.name}({fn.arguments})")
                result = dispatch_tool(fn.name, fn.arguments)
                messages.append(ToolMessage(tool_call_id=call.id, content=result))

if __name__ == "__main__":
    run()
