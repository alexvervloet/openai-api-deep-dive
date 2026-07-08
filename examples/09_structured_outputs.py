"""
Example 09 — JSON & structured outputs.
=======================================

Often you don't want prose — you want *data* your program can use directly. The
API can guarantee the reply is valid JSON, and even force it to match an exact
schema you define. No more fragile "please reply in JSON" prompting and hoping.

There are two levels:

  1. JSON mode — `response_format={"type": "json_object"}`.
     Guarantees the output is *syntactically valid JSON*. You still describe the
     shape you want in the prompt. Good enough for simple cases.

  2. Structured Outputs — `response_format={"type": "json_schema", ...}` with
     `"strict": True`.
     Guarantees the output *conforms to a JSON Schema you provide* — the right
     keys, the right types, every time. This is the robust choice. We use it
     below.

Either way, the content still comes back as a *string* — you parse it with
`json.loads()` yourself.

Run it:

    secrun python examples/09_structured_outputs.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

# Describe EXACTLY the structure we want back. `additionalProperties: False` and
# listing every field in `required` are needed for strict mode to be accepted.
schema = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "summary": {"type": "string"},
        "bugs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_hint": {"type": "string"},
                    "problem": {"type": "string"},
                },
                "required": ["line_hint", "problem"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["language", "summary", "bugs"],
    "additionalProperties": False,
}

code = """
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total // len(numbers)
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": f"Review this code:\n```\n{code}\n```"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "code_review",
            "schema": schema,
            "strict": True,  # the model is constrained to match the schema
        },
    },
)

# Guaranteed valid JSON matching our schema — safe to parse and use as a dict.
# `.content` is typed `str | None` (it's None when the model returns only tool
# calls), so we fall back to "{}" to keep both the runtime and the type checker
# happy.
data = json.loads(response.choices[0].message.content or "{}")

print(f"Language: {data['language']}")
print(f"Summary:  {data['summary']}")
print("Bugs:")
for bug in data["bugs"]:
    print(f"  - ({bug['line_hint']}) {bug['problem']}")
