"""
Responses 01: make a request and inspect every returned item.

The Responses API does not return a `choices` list. It returns an ordered list
of typed output items. A plain text request will usually produce a `message`,
but tool calls and reasoning can add other item types. Code that assumes
`response.output[0]` is a message will eventually break.

Predict before running
    Which item types will this request produce? Then compare your prediction
    with the list printed by the script.

Run
    secrun python responses/01_request_and_items.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
client = OpenAI()

response = client.responses.create(
    model=MODEL,
    instructions="Answer in one sentence for a software engineer.",
    input="What does an idempotency key prevent?",
    max_output_tokens=300,
)

print(response.output_text)
print("\nResponse metadata:")
print(f"  id: {response.id}")
print(f"  status: {response.status}")
print(f"  model: {response.model}")
print(f"  output item types: {[item.type for item in response.output]}")

if response.usage is not None:
    print(f"  input tokens: {response.usage.input_tokens}")
    print(f"  output tokens: {response.usage.output_tokens}")

print(
    "\nUse output_text when you only need aggregated text. Inspect output when "
    "item type, tool metadata, or ordering matters."
)
