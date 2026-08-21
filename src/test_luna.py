import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    """Run the manual Azure Luna smoke test.

    Keeping the provider call behind this entry point prevents pytest and other
    import-time tooling from sending a billable external request during module
    discovery.
    """
    load_dotenv()
    api_key = os.getenv("GPT_5_6_LUNA_AZURE_API_KEY")
    if not api_key:
        raise RuntimeError("GPT_5_6_LUNA_AZURE_API_KEY is not configured")

    client = OpenAI(
        base_url="https://isep-thesis.services.ai.azure.com/openai/v1",
        api_key=api_key,
    )
    response = client.responses.create(
        model="gpt-5.6-luna",
        input="What is seborrheic dermatitis?",
        reasoning={
            "effort": "high",
            "summary": "auto",
        },
    )

    print(f"answer: {response.output_text}")

    for item in response.output:
        if getattr(item, "type", None) == "reasoning":
            print(f"reasoning summary: {item.summary}")

    output_token_details = getattr(response.usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_token_details, "reasoning_tokens", None)
    print(f"reasoning tokens: {reasoning_tokens}")


if __name__ == "__main__":
    main()
