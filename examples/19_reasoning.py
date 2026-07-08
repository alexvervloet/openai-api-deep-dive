"""
Example 19 — reasoning models (the o-series): think first, answer second.
=========================================================================

The chat models you've used so far answer immediately, token by token. OpenAI's
*reasoning* models (the o-series — o1, o3, o4-mini — and the GPT-5 reasoning
tiers) do something different: before they write a visible answer, they generate
a private chain of **reasoning tokens** you never see, working the problem out
internally. That makes them much stronger at math, logic, coding, and multi-step
planning — at the cost of higher latency and more tokens billed.

Three things change about the request:

  1. You don't set `temperature`/`top_p` — reasoning models ignore sampling knobs.
     You steer effort with `reasoning_effort` instead ("low" | "medium" | "high"):
     more effort = more thinking tokens = better on hard problems, slower & pricier.
  2. The system role is called `developer` (system still works, but `developer`
     is the modern name for these models).
  3. `usage` now reports `reasoning_tokens` — hidden thinking you still pay for —
     under `completion_tokens_details`.

Use a reasoning model when the task is genuinely hard; a normal model like
gpt-4o-mini is cheaper and faster for everyday requests.

Run it:

    secrun python examples/19_reasoning.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

# Override with REASONING_MODEL in .env if your account has a different one
# available (o3, o3-mini, a gpt-5 reasoning tier, ...).
MODEL = os.getenv("REASONING_MODEL", "o4-mini")

# A puzzle that rewards working step-by-step rather than blurting an answer.
PROBLEM = (
    "A 3-gallon jug and a 5-gallon jug, and a tap. Measure out exactly 4 gallons. "
    "Give the shortest sequence of fill/empty/pour steps."
)

print(f"Model: {MODEL}   (reasoning_effort=high)\n")
print(f"Problem: {PROBLEM}\n")

response = client.chat.completions.create(
    model=MODEL,
    # Note: no temperature here — reasoning models don't use it.
    reasoning_effort="high",
    messages=[
        {"role": "developer", "content": "You are a careful puzzle solver. Show the final steps only."},
        {"role": "user", "content": PROBLEM},
    ],
)

print(response.choices[0].message.content)

# The hidden thinking is billed but never shown. Inspect it via usage:
usage = response.usage
assert usage is not None
details = getattr(usage, "completion_tokens_details", None)
reasoning = getattr(details, "reasoning_tokens", None) if details else None
print(f"\n[tokens — prompt: {usage.prompt_tokens}, "
      f"completion: {usage.completion_tokens}"
      + (f", of which reasoning (hidden): {reasoning}" if reasoning is not None else "") + "]")
print("Those reasoning tokens are the model 'thinking' — you pay for them but never see them.")
print("Try reasoning_effort='low' and watch them drop (and the answer sometimes get worse).")
