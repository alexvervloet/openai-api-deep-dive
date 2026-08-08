"""
Example 04: max_completion_tokens (and finish_reason).

`max_completion_tokens` caps how many tokens the model is allowed to GENERATE.
It does NOT limit your input, and it does NOT make the model "summarize to fit".
It simply cuts the model off when the budget runs out, mid-sentence if necessary.

Why use it?
  - Cost control: output tokens are the expensive ones.
  - Latency: shorter answers come back faster.
  - Safety: stop a runaway answer from ballooning.

The companion to watch is `finish_reason`:
  - "stop"   : the model finished on its own.
  - "length" : it hit your cap, so the answer is truncated.

A NAME CHANGE WORTH KNOWING
    This parameter used to be called `max_tokens`, and most tutorials still
    call it that. On the gpt-5 line the old name is rejected outright:

        Unsupported parameter: 'max_tokens' is not supported with this model.
        Use 'max_completion_tokens' instead.

    The rename is not cosmetic. On reasoning models the budget covers tokens
    you never see: the model's internal reasoning is generated, billed, and
    counted against this cap before a single visible character is produced.
    "Completion" tokens is the honest name for what is being capped. That is
    also why a cap that looks generous can still return an EMPTY string with
    finish_reason "length": the reasoning ate the whole budget. If you get a
    blank answer, raise the cap before you suspect your prompt.

Run it:

    secrun python examples/04_max_tokens.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

prompt = "Explain how the internet works."

# If you'd like to see the reason be "stop", add a large number like 2000 to
# the loop. But remember larger numbers incur more cost.
for cap in (16, 256):
    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=cap,
    )
    choice = response.choices[0]
    print(f"--- max_completion_tokens={cap} (finish_reason={choice.finish_reason}) ---")
    print(choice.message.content)
    print()
