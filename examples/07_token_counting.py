"""
Example 07 — counting tokens and estimating cost (no API call!).
================================================================

This example uses NO network and NO API key. Everything here runs locally with
tiktoken, which is the whole point: you can plan and budget a request before you
ever spend a cent.

Run it:

    python examples/07_token_counting.py

It shows three things:
  1. How a sentence breaks into tokens.
  2. How a full chat `messages` list is counted (with per-message overhead).
  3. How that token count maps to dollars at different models' prices.
"""

# Make the repo-root modules (utils/pricing.py, utils/tokens.py) importable no
# matter what directory you run this from.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pricing import PRICING, estimate_cost, format_cost
from utils.tokens import count_message_tokens, count_tokens

# 1. Raw string tokenization.
sentence = "Tokens are not the same as words!"
print(f"{sentence!r}")
print(f"  -> {count_tokens(sentence)} tokens\n")

# 2. A realistic chat request.
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the plot of Hamlet in two sentences."},
]
input_tokens = count_message_tokens(messages)
print(f"Chat request input tokens (with overhead): {input_tokens}\n")

# 3. Cost across models, assuming a 150-token answer.
assumed_output = 150
print(f"Estimated cost for {input_tokens} in + {assumed_output} out:")
for model in PRICING:
    cost = estimate_cost(model, input_tokens, assumed_output)
    print(f"  {model:<16} {format_cost(cost)}")

print(
    "\nNotice how much cheaper gpt-4o-mini is than gpt-4o for the same request — "
    "this is why picking the right model matters as much as writing a good prompt."
)
