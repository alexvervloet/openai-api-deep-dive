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
gpt-5.6-luna, a current tier that costs about the same as the series default.

Three things change about the request:

  1. You don't set `temperature`/`top_p`; the GPT-5.6 tiers reject them outright.
     You steer effort with `reasoning_effort` instead ("none" | "low" | "medium"
     | "high" | "xhigh"): more effort = more thinking tokens = better on hard
     problems, slower and pricier.
  2. The system role is called `developer` (system still works, but `developer`
     is the modern name for these models).
  3. `usage` now reports `reasoning_tokens`: hidden thinking you still pay for
     under `completion_tokens_details`.

Note the default. On chat completions, GPT-5.6 defaults `reasoning_effort` to
"none", so a request that says nothing gets no reasoning at all. You have to ask.

One sharp edge worth knowing before you build on this: on chat completions the
GPT-5.6 tiers reject function tools combined with any `reasoning_effort` above
"none". A 400, not a degraded answer. Tools plus reasoning is a Responses API
job, which is what `responses/04_custom_tool_loop.py` is for.

Use a reasoning model when the task is genuinely hard; a normal model like
gpt-5.4-nano is cheaper and faster for everyday requests.

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

# Override with REASONING_MODEL in .env to try another tier (gpt-5.6-terra,
# gpt-5.6-sol, gpt-6-astra, ...). Anything on the GPT-5.6 line or later works.
MODEL = os.getenv("REASONING_MODEL", "gpt-5.6-luna")

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
