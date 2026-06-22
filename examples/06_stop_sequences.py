"""
Example 06 — stop sequences.
============================

`stop` is a string (or list of up to 4 strings) that tells the model: "the
moment you're about to produce this text, stop generating." The stop text itself
is NOT included in the output.

Uses:
  - Cut a list off after N items (stop at "4.").
  - End a structured response at a delimiter.
  - Prevent the model from running past a known boundary (e.g. "\n\n").

Run it:

    python examples/06_stop_sequences.py

The first call lets the model count freely; the second stops it the instant it
tries to write "4", so you only get items 1–3.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) and try again.")

client = OpenAI()

prompt = "Count from 1 to 10, one number per line, like '1.', '2.', ..."

print("--- without stop ---")
r1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print(r1.choices[0].message.content)

print("\n--- with stop=['4.'] ---")
r2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    stop=["4."],
)
print(r2.choices[0].message.content)
print(f"(finish_reason={r2.choices[0].finish_reason})")
