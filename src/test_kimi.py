from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = "https://isep-thesis.services.ai.azure.com/openai/v1"
deployment_name = "Kimi-K2.6"

client = OpenAI(
    base_url=endpoint,
    api_key=os.getenv("KIMI_K2_6_AZURE_API_KEY")
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is seborrheic dermatitis?",
        }
    ],
    temperature=0.6,
    top_p=0.95,
    # The Direct-from-Azure gateway rejects Moonshot's official
    # `thinking: {type: disabled}` object. Its accepted no-reasoning
    # compatibility control is reasoning_effort="none".
    reasoning_effort="none",
)

message = completion.choices[0].message
print("answer:", message.content)
print(
    "reasoning:",
    getattr(message, "reasoning", None)
    or getattr(message, "reasoning_content", None),
)
