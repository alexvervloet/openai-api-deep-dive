"""
Example 21: prompt caching: stop paying full price for a repeated prefix.

Many real apps send the *same* long prefix on every request: a big system prompt,
a tool catalog, a document you're asking many questions about. Re-processing those
identical input tokens every time is wasteful, so OpenAI **caches** them.

The good news: on OpenAI it's **automatic**. Any request whose prompt is ≥1024
tokens has its longest matching *prefix* cached; identical prefixes on later
requests are read from cache at a **discount** (cached input tokens are billed at
a fraction of the normal input price) and processed faster. You don't call a
special endpoint. You just structure prompts so the *stable* part comes first and
the *variable* part (the user's actual question) comes last.

This script sends two requests that share a long, identical system prefix, then
reads `usage.prompt_tokens_details.cached_tokens` to show the cache kicking in on
the second call.

The one design rule: **put the constant stuff at the front, the variable stuff at
the back.** A cache only helps the prefix that's byte-for-byte identical.

Run it:

    secrun python examples/21_prompt_caching.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

# A long, STABLE prefix (must be ≥1024 tokens to be cacheable). We fake one by
# repeating a policy block; in a real app this is your big system prompt, a tool
# catalog, or a document you're answering questions about.
STABLE_PREFIX = (
    "You are Acme Corp's support assistant. Follow these policies exactly.\n"
    + "\n".join(f"Policy {i}: Always be concise, accurate, and cite the policy number when relevant. "
                f"Never share internal pricing. Escalate anything about refunds over $500."
                for i in range(1, 120))
)


def ask(question: str):
    """Same long system prefix every time; only the question changes."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": STABLE_PREFIX},  # constant -> cacheable
            {"role": "user", "content": question},          # variable -> at the end
        ],
        max_tokens=30,
    )
    usage = resp.usage
    assert usage is not None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return resp.choices[0].message.content, usage.prompt_tokens, cached


print(f"Stable prefix is ~{len(STABLE_PREFIX) // 4} tokens (needs ≥1024 to cache).\n")

# First call: writes the prefix into the cache (cached_tokens is usually 0).
ans1, total1, cached1 = ask("How do I reset my password?")
print(f"Call 1: {total1} prompt tokens, {cached1} cached")

# Second call: same prefix -> most of those tokens are now served from cache.
ans2, total2, cached2 = ask("What's your refund window?")
print(f"Call 2: {total2} prompt tokens, {cached2} cached  <-- the cache paid off")

print("\nThe second call billed most of the prefix at the cheaper cached rate, and")
print("processed faster, for free, just by keeping the constant part at the front.")
print("(Caches are best-effort and expire after minutes of inactivity; don't rely on")
print(" a specific hit, just structure prompts to make hits likely.)")
