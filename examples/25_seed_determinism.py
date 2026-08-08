"""
Example 25: seed & reproducibility: pinning down a random model.

Generation is random by default (that's what `temperature` controls; see example 03).
Run the same prompt twice and you get two different answers. Sometimes you want the
*opposite*: the same input to produce the same output, for tests, for caching, for
debugging, for reproducible evals.

`seed=<int>` is the lever: it fixes the random state the model samples from, so the
*same* seed + same inputs reproduce the *same* output, even with temperature turned
up. This script proves that directly, running at `temperature=0.9` (real randomness)
so the seed's effect is unmistakable, rather than hiding behind `temperature=0`'s
already-deterministic greedy decoding (where a seed wouldn't visibly be doing much).

  - `seed=42` twice -> identical output (the seed pins the randomness down).
  - `seed=None` twice -> different output (nothing pins it down, so it's free to vary).

In production you'd typically combine `temperature=0` *and* a fixed `seed`. temp=0
removes most of the variation by always taking the most likely token, and the seed
locks down whatever tie-breaking remains. This script isolates the seed's own effect
by leaving temperature high, so you can see it doing the work on its own.

The honest caveat either way: this is **best-effort, not a guarantee.** OpenAI may
change the backend (detectable via the `system_fingerprint` field: if it changes,
determinism can break). Treat seed as "much more reproducible," not "byte-identical
forever."

Run it:

    secrun python examples/25_seed_determinism.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

PROMPT = (
    "Invent a quirky name and a one-sentence tagline for a coffee shop run by cats. "
    "Format: Name, tagline."
)
TEMPERATURE = 0.9  # real randomness, so a fixed seed's effect is actually visible


def generate(seed=None):
    resp = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=TEMPERATURE,
        seed=seed,
        max_completion_tokens=40,
    )
    return (resp.choices[0].message.content or "").strip(), resp.system_fingerprint


print(
    f"With temperature={TEMPERATURE} and a FIXED seed (42), expect identical (or near-identical) output:\n"
)
a, fp_a = generate(seed=42)
b, fp_b = generate(seed=42)
print(f"  run 1: {a!r}")
print(f"  run 2: {b!r}")
print(f"  -> {'IDENTICAL ✓' if a == b else 'differed (backend may have shifted)'}")
print(
    f"  system_fingerprint: {fp_a} / {fp_b}"
    + (
        "  (same backend)"
        if fp_a == fp_b
        else "  (DIFFERENT backend; determinism not guaranteed)"
    )
)

print(
    f"\nSame temperature={TEMPERATURE}, but NO seed: expect different output across runs:\n"
)
c, _ = generate(seed=None)
d, _ = generate(seed=None)
print(f"  run 3: {c!r}")
print(f"  run 4: {d!r}")
print(
    f"  -> {'differed, as expected (nothing pins the randomness down) ✓' if c != d else 'IDENTICAL (got lucky, or model defaulted similarly)'}"
)

print("\nTakeaway: it's the seed, not temperature=0, that makes output reproducible.")
print("Pair it with temperature=0 in production for the strongest guarantee, and watch")
print("system_fingerprint: if it changes, the backend changed and outputs can drift")
print("even with the same seed. Good enough for tests and caching; not a promise.")
