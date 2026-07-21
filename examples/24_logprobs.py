"""
Example 24: logprobs: how *confident* was the model?

A normal response gives you the model's chosen words but says nothing about how
sure it was. Set `logprobs=True` and the API also returns, for each token it
generated, the **log-probability** it assigned, and with `top_logprobs=k`, the
k most likely alternatives it considered at that position.

Why you'd want this:
  - **Confidence scoring.** Turn a log-prob into a probability (`math.exp`) to get a
    0–1 confidence per token. Useful to flag shaky answers for review.
  - **Classification with calibration.** For a one-token answer ("yes"/"no",
    "positive"/"negative"), the alternatives' probabilities tell you *how close*
    the call was, far more informative than the bare label.
  - **Debugging.** See where the model was torn between two continuations.

CAVEAT: confidence is not the same as correctness. A logprob measures how
peaked the model's next-token distribution was, given its training data. It
says nothing about whether the model actually *knows* the answer. Ask it
something unknowable (future weather, an unpublished number) and forced into
a one-word reply, it will often still commit to a single token with near-100%
"confidence". It's just reproducing the most statistically common phrasing
for that kind of question, not reporting epistemic uncertainty. Low-confidence,
split logprobs show up when the model is genuinely torn between plausible
continuations (ambiguous classification, or tasks it tends to get wrong) 
not when it lacks information it never had a way to access.

This script asks a yes/no question, then prints the probability of the answer
token and the runners-up it weighed.

Run it:

    secrun python examples/24_logprobs.py
"""

import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()


def confidence(question: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer with exactly one word: Yes or No."},
            {"role": "user", "content": question},
        ],
        max_tokens=1,
        logprobs=True,  # ask for log-probabilities...
        top_logprobs=5,  # ...and the 5 alternatives at each position
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
    "Is the Earth larger than the Moon?",  # the model should be very sure
    "Will it rain in Paris next Tuesday?",  # unknowable, but watch the model still
    # answer near-100% confident. See the
    # CAVEAT above
    "Is a hot dog a sandwich?",  # genuinely contested classification 
    # training data argues both ways, so the
    # logprobs are more likely to split
]

for q in QUESTIONS:
    answer, prob, alts = confidence(q)
    print(f"Q: {q}")
    print(f"  answer: {answer!r}   confidence: {prob:.1%}")
    print("  it also considered: " + ", ".join(f"{tok!r}={p:.1%}" for tok, p in alts))
    print()

print("A confident answer puts almost all probability on one token; a genuinely")
print("torn one spreads it across alternatives. That spread is a signal you can act")
print("on: auto-accept the confident ones, route the shaky ones to a human.")
print()
print("But notice the Paris weather question: the model can't know the answer, yet")
print("it was still ~100% confident. High confidence reflects a peaked training")
print("distribution, not factual certainty. Don't treat logprobs as a truth signal")
print("for questions the model has no way to answer.")
