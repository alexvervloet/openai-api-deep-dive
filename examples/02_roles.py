"""
Example 02: system / user / assistant roles.

A chat is a transcript of messages, each tagged with a role:

  - system    : Standing instructions. Sets the persona, rules, and tone for the
                whole conversation. Usually the first message. The user doesn't
                "see" it; it steers everything that follows.
  - user      : What the human says.
  - assistant : What the model said. You include PRIOR assistant messages to give
                the model memory of the conversation. The API itself is
                stateless, so *you* resend the history every time.

Run it:

    secrun python examples/02_roles.py

Try editing the system message (e.g. "You are a grumpy pirate") and watch the
tone of the answer change without touching the question at all. That's the power
of the system role.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

client = OpenAI()

# Note the assistant message in the middle: we're *simulating* a prior turn so
# the model continues the thread coherently. This is how you build multi-turn
# chat: keep appending messages to the list.
messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a terse math tutor. One line only."},
    {"role": "user", "content": "What is 12 * 12?"},
    {"role": "assistant", "content": "144."},
    {"role": "user", "content": "And that, doubled?"},
]

response = client.chat.completions.create(
    model="gpt-5.4-nano",
    messages=messages,
)

print(response.choices[0].message.content)
