"""
Lab 01 — Basic Inference: Azure AI Foundry

Uses the azure-ai-inference SDK, which provides a unified interface to any
model deployed in an Azure AI Foundry project — including Claude via the
Azure Marketplace.

Auth: API key via AZURE_FOUNDRY_API_KEY (or swap AzureKeyCredential for
DefaultAzureCredential to use managed identity / Entra ID in production).
"""

import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

PROMPT = "Explain what a foundation model is in two sentences, for a software engineer who hasn't worked with AI before."

def run():
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_FOUNDRY_API_KEY"]),
    )

    response = client.complete(
        model=os.getenv("AZURE_FOUNDRY_MODEL", "claude-3-5-sonnet"),
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=PROMPT),
        ],
        max_tokens=256,
    )

    choice = response.choices[0]
    print(f"[Foundry] {choice.message.content}")
    print(f"  finish_reason={choice.finish_reason}  prompt_tokens={response.usage.prompt_tokens}  completion_tokens={response.usage.completion_tokens}")

if __name__ == "__main__":
    run()
