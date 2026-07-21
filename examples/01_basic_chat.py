"""
Example 01: Your first chat completion.

The whole API in five lines. You send a list of messages; you get back a
message. Run it:

    secrun python examples/01_basic_chat.py

What to notice:
  - `client = OpenAI()` reads your key from the OPENAI_API_KEY environment
    variable. We load it from .env first.
  - `messages` is a list. Even a one-off question is a list with one entry.
  - The reply lives at `response.choices[0].message.content`. There can be more
    than one choice if you ask for several (the `n` parameter), hence the [0].
  - `response.usage` reports exactly how many tokens you were billed for.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "In one sentence, what is an API?"},
    ],
)

print(response.choices[0].message.content)
print("\n--- usage ---")
print(response.usage)
