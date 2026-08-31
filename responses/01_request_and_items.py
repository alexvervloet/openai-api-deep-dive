"""
Responses 01: make a request and inspect every returned item.

If you have worked through `examples/` you already know the Chat Completions
shape: you send `messages`, you read `choices[0].message.content`. Two things
change here. The request splits into `instructions` (the standing brief) and
`input` (this turn's content). The response drops `choices` entirely and gives
you `output`, an ordered list of typed items.

That list is the part worth slowing down for. An item has a `type`. A plain
text request produces one `message` item, so `output[0]` happens to be the
message and naive code appears to work. It is the easy case, and it is exactly
what trains the wrong habit: as soon as a tool is involved the first item is
something else. Example 05 returns `['web_search_call', 'message']`, and code
that reached for `output[0].content` there is reading a search call.

So there are two correct ways to read a response, and this script prints both:
`output_text` when you only want the words, and a filtered walk of `output`
when type, ordering, or tool metadata matters.

Predict before running
    How many items will this request produce, and of which types? Then check
    your prediction against the list the script prints.

Run
    secrun python responses/01_request_and_items.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

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

# The same text reached two ways. The naive read assumes item 0 is a message.
# The filtered read asks. Both print the same string here, which is the point:
# the assumption is invisible until a tool call puts another item in front.
messages = [item for item in response.output if item.type == "message"]
print("\nTwo ways to reach the text:")
print(f"  output[0].type is {response.output[0].type!r}, so the naive read works today")
print(f"  filtering for message items found {len(messages)} of {len(response.output)}")

print(
    "\nUse output_text when you only need aggregated text. Filter output by item "
    "type when tool metadata or ordering matters. Run example 05 to see the same "
    "code path with a non-message item first in the list."
)
