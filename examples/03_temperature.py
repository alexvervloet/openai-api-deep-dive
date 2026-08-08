"""
Example 03: temperature.

`temperature` controls randomness, roughly 0.0 to 2.0.

  - 0.0  : The model almost always picks its single most likely next token.
           Answers are focused, repeatable, and a bit "safe". Best for facts,
           code, extraction, anything where you want consistency.
  - 0.7  : A balanced default. Some variety, still coherent.
  - 1.5+ : Wild. More surprising word choices, more risk of nonsense. Good for
           brainstorming or creative writing.

Run it:

    secrun python examples/03_temperature.py

We ask the same creative question at three temperatures and print the results
side by side. Notice how 0.0 tends to repeat itself across runs while the high
setting reinvents the answer each time.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

prompt = "Give a five-word slogan for a coffee shop on the moon."

for temp in (0.0, 0.7, 1.5, 2.0):
    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"temperature={temp:<4} -> {response.choices[0].message.content}")
