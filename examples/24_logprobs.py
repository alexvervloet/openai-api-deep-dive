"""
Example 24 — logprobs: how *confident* was the model?
=====================================================

A normal response gives you the model's chosen words but says nothing about how
sure it was. Set `logprobs=True` and the API also returns, for each token it
generated, the **log-probability** it assigned — and with `top_logprobs=k`, the
k most likely alternatives it considered at that position.

Why you'd want this:
  - **Confidence scoring.** Turn a log-prob into a probability (`math.exp`) to get a
    0–1 confidence per token. Useful to flag shaky answers for review.
  - **Classification with calibration.** For a one-token answer ("yes"/"no",
    "positive"/"negative"), the alternatives' probabilities tell you *how close*
    the call was — far more informative than the bare label.
  - **Debugging.** See where the model was torn between two continuations.

This script asks a yes/no question, then prints the probability of the answer
token and the runners-up it weighed.

Run it:

    python examples/24_logprobs.py
"""

import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) and try again.")

client = OpenAI()


def confidence(question: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer with exactly one word: Yes or No."},
            {"role": "user", "content": question},
        ],
        max_tokens=1,
        logprobs=True,        # ask for log-probabilities...
        top_logprobs=5,       # ...and the 5 alternatives at each position
    )
    # One token out, so we look at the first (only) entry.
    logprobs = resp.choices[0].logprobs
    assert logprobs is not None and logprobs.content is not None
    token_info = logprobs.content[0]
    answer = token_info.token
    prob = math.exp(token_info.logprob)  # logprob -> probability in [0, 1]
    alternatives = [
        (alt.token, math.exp(alt.logprob)) for alt in token_info.top_logprobs
    ]
    return answer, prob, alternatives


QUESTIONS = [
    "Is the Earth larger than the Moon?",   # the model should be very sure
    "Will it rain in Paris next Tuesday?",  # genuinely uncertain
]

for q in QUESTIONS:
    answer, prob, alts = confidence(q)
    print(f"Q: {q}")
    print(f"  answer: {answer!r}   confidence: {prob:.1%}")
    print("  it also considered: " +
          ", ".join(f"{tok!r}={p:.1%}" for tok, p in alts))
    print()

print("A confident answer puts almost all probability on one token; a genuinely")
print("uncertain one spreads it across alternatives. That spread is a signal you can")
print("act on — auto-accept the confident ones, route the shaky ones to a human.")
