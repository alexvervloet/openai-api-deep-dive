"""
Example 14 — Pydantic validation of responses.
===============================================

Example 09 made the model return JSON matching a hand-written JSON Schema, then
parsed it with `json.loads()` into a plain dict. That works, but you're left
holding an untyped dict: no editor autocomplete, no real validation, and a typo
like `data["sumary"]` blows up only at runtime.

The better way: define your shape as a **Pydantic model** and let the SDK do the
rest. `client.chat.completions.parse(...)` (the structured-outputs helper):

  - converts your model into a strict JSON Schema and sends it for you,
  - constrains the model to match it,
  - and hands back an *already-validated instance of your model* on
    `message.parsed` — typed attributes, not a dict.

Why this beats a raw dict:
  - **Types**: `review.bugs[0].severity` autocompletes and is checked.
  - **Validation**: Pydantic enforces constraints (enums, ranges, lengths). If
    the data doesn't fit, you get a clear `ValidationError` instead of a
    land-mine deep in your code.
  - **Less code**: no schema dict to maintain, no manual `json.loads`.

Run it:

    secrun python examples/14_pydantic_validation.py
"""

import os
import sys
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()


# 1. Describe the shape as Pydantic models. Field(...) descriptions are sent to
#    the model as part of the schema, so they double as instructions. Constraints
#    (the Enum, ge/le bounds) are enforced by Pydantic when the reply comes back.
class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Bug(BaseModel):
    line_hint: str = Field(description="Where the problem is, e.g. 'the return line'.")
    problem: str = Field(description="What's wrong, in one sentence.")
    severity: Severity


class CodeReview(BaseModel):
    language: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0, description="0–1 self-rated confidence.")
    bugs: list[Bug]


code = """
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total // len(numbers)
"""

# 2. Pass the MODEL CLASS as response_format. No schema dict, no json.loads.
response = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a meticulous code reviewer."},
        {"role": "user", "content": f"Review this code:\n```\n{code}\n```"},
    ],
    response_format=CodeReview,
)

message = response.choices[0].message

# 3. The model may decline (safety). When it does, `.parsed` is None and the
#    reason is on `.refusal` — always check before using the parsed object.
if message.refusal:
    print(f"Model refused: {message.refusal}")
    sys.exit()

review = message.parsed  # <- a validated CodeReview instance, not a dict
assert review is not None

# 4. Use it like the typed object it is — attributes, not string keys.
print(f"Language:   {review.language}")
print(f"Summary:    {review.summary}")
print(f"Confidence: {review.confidence:.0%}")
print("Bugs:")
for bug in review.bugs:
    print(f"  - [{bug.severity.value:>6}] ({bug.line_hint}) {bug.problem}")
