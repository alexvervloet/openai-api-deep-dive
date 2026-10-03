"""
Example 19: reasoning models: think first, answer second.

The chat models you've used so far answer immediately, token by token. A
*reasoning* model does something different: before it writes a visible answer it
generates a private chain of **reasoning tokens** you never see, working the
problem out internally. That makes it much stronger at math, logic, coding, and
multi-step planning, at the cost of higher latency and more tokens billed.

Reasoning used to be a separate family of models, the o-series (o1, o3,
o4-mini). It isn't any more. It's a dial on the mainline models, and the
o-series is being switched off: o1, o1-pro, and o4-mini shut down on
2026-10-23, and the 2025 o3 snapshots on 2026-12-11. So this example runs on
gpt-6-luna, the same model every other example in this repo uses. The only
difference is where the dial sits.

Three things change about the request:

  1. You don't set `temperature`/`top_p`; luna rejects them at any effort above
     "none". You steer with `reasoning_effort` instead ("none" | "low" |
     "medium" | "high" | "xhigh" | "max"): more effort = more thinking tokens =
     better on hard problems, slower and pricier.
  2. The system role is called `developer` (system still works, but `developer`
     is the modern name for these models).
  3. `usage` now reports `reasoning_tokens`: hidden thinking you still pay for
     under `completion_tokens_details`.

Note the default. Luna defaults `reasoning_effort` to "medium", so a request
that says nothing still thinks, and you pay for it. That's why every other
example here sends "none" explicitly. The GPT-5.6 tiers went the other way and
default to "none", so check the default per model rather than per family.

One sharp edge worth knowing before you build on this: on chat completions luna
rejects function tools combined with any `reasoning_effort` above "none". A
400, not a degraded answer. Tools plus reasoning is a Responses API job, which
is what `responses/04_custom_tool_loop.py` is for.

Use reasoning when the task is genuinely hard. For everyday requests the same
model at "none" is cheaper and faster.

Run it:

    secrun python examples/19_reasoning.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

client = OpenAI()

# Override with REASONING_MODEL in .env to try another tier (gpt-6-sol,
# gpt-5.6-terra, gpt-6-astra, ...). Anything on the GPT-5.6 line or later works.
MODEL = os.getenv("REASONING_MODEL", "gpt-6-luna")

# A puzzle that rewards working step-by-step rather than blurting an answer.
PROBLEM = (
    "A 3-gallon jug and a 5-gallon jug, and a tap. Measure out exactly 4 gallons. "
    "Give the shortest sequence of fill/empty/pour steps."
)

print(f"Model: {MODEL}   (reasoning_effort=high)\n")
print(f"Problem: {PROBLEM}\n")

response = client.chat.completions.create(
    model=MODEL,
    # Note: no temperature here; reasoning models don't use it.
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
print(f"\n[tokens: prompt: {usage.prompt_tokens}, "
      f"completion: {usage.completion_tokens}"
      + (f", of which reasoning (hidden): {reasoning}" if reasoning is not None else "") + "]")
print("Those reasoning tokens are the model 'thinking': you pay for them but never see them.")
print("Try reasoning_effort='low' and watch them drop (and the answer sometimes get worse).")
