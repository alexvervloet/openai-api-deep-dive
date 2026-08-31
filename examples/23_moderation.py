"""
Example 23: the moderation endpoint: a free safety filter.

Before you send user text to a model (or publish a model's output), you often
want to know: does this contain hateful, violent, sexual, or self-harm content?
OpenAI exposes a dedicated **moderation** endpoint for exactly this. It's a
separate, **free** classifier, not a chat model, that returns category flags and
scores for a piece of text.

The response gives you, per input:
  - `flagged`: a single boolean. Did anything trip the policy?
  - `categories`: a bool per category (hate, violence, sexual, self-harm, ...).
  - `category_scores`: a 0-1 confidence per category, so you can set your own
    threshold instead of trusting the default flag.

A common pattern: moderate the *user's input* on the way in and the *model's
output* on the way out, and refuse / redact when `flagged` is true. It's cheap
(free) and fast, so there's little reason not to.

Run it:

    secrun python examples/23_moderation.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

client = OpenAI()

SAMPLES = [
    "I love this product, it works great!",          # clearly fine
    "I'm going to find you and hurt you.",            # a threat -> should flag
    "How do I bake sourdough bread at home?",         # fine
]


def top_categories(scores, n: int = 3):
    """The n highest-scoring categories, for a readable summary."""
    items = sorted(scores.model_dump().items(), key=lambda kv: kv[1], reverse=True)
    return [(name, score) for name, score in items[:n]]


# The moderation endpoint takes one string or a list of strings.
result = client.moderations.create(
    model="omni-moderation-latest",
    input=SAMPLES,
)

for text, item in zip(SAMPLES, result.results):
    verdict = "🚩 FLAGGED" if item.flagged else "✓ ok"
    print(f"{verdict}  {text!r}")
    if item.flagged:
        # Show which categories tripped and how confidently.
        tripped = [name for name, on in item.categories.model_dump().items() if on]
        print(f"        categories: {', '.join(tripped)}")
        print(f"        top scores: " +
              ", ".join(f"{name}={score:.2f}" for name, score in top_categories(item.category_scores)))
    print()

print("Use this as a cheap gate: moderate input on the way in and output on the way")
print("out. Set your own threshold on category_scores if the default flag is too")
print("strict or too loose for your app.")
