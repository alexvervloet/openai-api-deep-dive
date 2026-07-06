"""
Example 04 — max_tokens (and finish_reason).
============================================

`max_tokens` caps how many tokens the model is allowed to GENERATE. It does NOT
limit your input, and it does NOT make the model "summarize to fit" — it simply
cuts the model off when the budget runs out, mid-sentence if necessary.

Why use it?
  - Cost control: output tokens are the expensive ones.
  - Latency: shorter answers come back faster.
  - Safety: stop a runaway answer from ballooning.

The companion to watch is `finish_reason`:
  - "stop"   : the model finished on its own.
  - "length" : it hit your max_tokens cap — the answer is truncated.

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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cap,
    )
    choice = response.choices[0]
    print(f"--- max_tokens={cap} (finish_reason={choice.finish_reason}) ---")
    print(choice.message.content)
    print()
