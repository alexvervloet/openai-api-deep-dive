"""
Example 05 — top_p (nucleus sampling).
======================================

`top_p` is the other randomness knob. Instead of scaling probabilities like
temperature does, it *restricts the candidate pool*:

  top_p = 0.1  -> consider only the smallest set of tokens whose probabilities
                  add up to 10%. Very focused — picks from the obvious choices.
  top_p = 1.0  -> consider everything (no restriction). This is the default.

Mental model: temperature changes *how boldly* the model chooses among options;
top_p changes *how many options it's even allowed to consider*.

Important: OpenAI recommends tuning EITHER temperature OR top_p, not both at
once, because they interact in confusing ways. Pick one knob and learn it.

Run it:

    secrun python examples/05_top_p.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

prompt = "Name an unusual but real animal."

for p in (0.1, 1.0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        top_p=p,
        # We leave temperature at its default and only vary top_p here.
    )
    print(f"top_p={p:<4} -> {response.choices[0].message.content}")
