"""
Lab 02 — Streaming: Azure AI Foundry

The azure-ai-inference SDK uses an iterator-based streaming pattern.
Each chunk carries a delta (partial text) rather than the full accumulated response.
"""

import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

PROMPT = "Walk me through how transformer attention works. Be thorough."

def run():
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_FOUNDRY_API_KEY"]),
    )

    print("[Foundry] ", end="", flush=True)

    # stream=True switches the response to an iterator of StreamingChatCompletionsUpdate
    response = client.complete(
        model=os.getenv("AZURE_FOUNDRY_MODEL", "claude-sonnet-4-6"),
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=PROMPT),
        ],
        max_tokens=512,
        stream=True,
    )

    for update in response:
        if update.choices and update.choices[0].delta.content:
            print(update.choices[0].delta.content, end="", flush=True)

    print()  # newline after stream ends

if __name__ == "__main__":
    run()
