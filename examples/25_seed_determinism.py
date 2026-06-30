"""
Example 25 — seed & reproducibility: (mostly) the same answer twice.
====================================================================

Generation is random by default (that's what `temperature` controls — example 03).
Run the same prompt twice and you get two different answers. Sometimes you want the
*opposite*: the same input to produce the same output — for tests, for caching, for
debugging, for reproducible evals.

Two levers together get you there:

  - `temperature=0` — always take the most likely token. This alone removes most of
    the variation.
  - `seed=<int>` — ask the API to use a fixed random seed, so even the remaining
    tie-breaking is deterministic across calls with the same inputs.

The honest caveat: this is **best-effort, not a guarantee.** OpenAI may change the
backend (you can detect this via the `system_fingerprint` field — if it changes,
determinism can break). Treat seed as "much more reproducible," not "byte-identical
forever."

This script sends the same creative prompt twice with a fixed seed and compares the
results, then once without a seed to show the contrast.

Run it:

    python examples/25_seed_determinism.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) and try again.")

client = OpenAI()

PROMPT = "Invent a quirky name for a coffee shop run by cats. Just the name."


def generate(seed=None):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,     # take the most likely token...
        seed=seed,         # ...and fix the seed for the rest
        max_tokens=20,
    )
    return (resp.choices[0].message.content or "").strip(), resp.system_fingerprint


print("With a FIXED seed (42), temperature 0 — expect identical (or near-identical) output:\n")
a, fp_a = generate(seed=42)
b, fp_b = generate(seed=42)
print(f"  run 1: {a!r}")
print(f"  run 2: {b!r}")
print(f"  -> {'IDENTICAL ✓' if a == b else 'differed (backend may have shifted)'}")
print(f"  system_fingerprint: {fp_a} / {fp_b}"
      + ("  (same backend)" if fp_a == fp_b else "  (DIFFERENT backend — determinism not guaranteed)"))

print("\nWith NO seed (still temp 0) — usually the same here, but unpinned:\n")
c, _ = generate(seed=None)
print(f"  run 3: {c!r}")

print("\nTakeaway: temperature=0 + a fixed seed is the most reproducible the API gets.")
print("Watch system_fingerprint — if it changes, the backend changed and outputs can")
print("drift even with the same seed. Good enough for tests and caching; not a promise.")
