from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = "https://isep-thesis.services.ai.azure.com/openai/v1"
deployment_name = "gpt-5.6-luna"

client = OpenAI(
    base_url=endpoint,
    api_key=os.getenv("GPT_5_6_LUNA_AZURE_API_KEY"),
)

response = client.responses.create(
    model=deployment_name,
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
