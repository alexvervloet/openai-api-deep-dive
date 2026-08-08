"""
Example 06: stop sequences, and a parameter being retired under you.

`stop` was a string (or list of up to 4 strings) that told the model: "the
moment you are about to produce this text, stop generating." The stop text
itself was not included in the output.

It was used to:
  - cut a list off after N items (stop at "4."),
  - end a structured response at a delimiter,
  - keep the model from running past a known boundary (e.g. "\\n\\n").

THE POINT OF THIS EXAMPLE HAS CHANGED
    `stop` is not supported on the gpt-5 line. Every gpt-5 model tested on
    2026-08-08 (nano, mini, and the 5.6 tiers) rejects it:

        Unsupported parameter: 'stop' is not supported with this model.

    Only the older gpt-4o line still accepts it. So this file now teaches two
    things at once: what stop sequences were, and what it feels like when a
    parameter you built on is retired.

WHY IT WENT AWAY
    Stop sequences are a text-completion idea. You could not ask the old
    completion API for a shape, so you asked for prose and chopped it at a
    marker you hoped the model would emit. It was always brittle: if the model
    phrased things differently, the marker never appeared and you got the whole
    answer anyway.

    The replacements are stricter and do not depend on hope:
      - structured outputs (`response_format`) when you want a SHAPE. See
        examples/09_structured_outputs.py.
      - `max_completion_tokens` when you want a LENGTH cap. See
        examples/04_max_tokens.py.
      - tool calling when you want the model to hand control back to your code.
        See examples/10_function_calling.py.

    Reach for a marker-and-chop approach today and you are emulating a 2022 API
    on top of a 2026 one.

Run it:

    secrun python examples/06_stop_sequences.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

CURRENT_MODEL = "gpt-5.4-nano"   # what the rest of this repo uses
LEGACY_MODEL = "gpt-4o-mini"     # still served, still accepts `stop`

prompt = "Count from 1 to 10, one number per line, like '1.', '2.', ..."

# 1. The current model refuses the parameter outright. We catch it rather than
#    crash, because seeing the actual error text is the lesson.
print(f"--- {CURRENT_MODEL} with stop=['4.'] ---")
try:
    client.chat.completions.create(
        model=CURRENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200,
        stop=["4."],
    )
    print("Accepted. The parameter has evidently been reinstated; update this file.")
except Exception as e:
    print(f"Rejected, as expected:\n  {e}\n")

# 2. The legacy model still honours it, so you can see the behaviour it used to
#    give you: generation halts the instant the model tries to write "4.".
print(f"--- {LEGACY_MODEL} with stop=['4.'] ---")
r = client.chat.completions.create(
    model=LEGACY_MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200,  # the legacy line still uses the old parameter name too
    stop=["4."],
)
print(r.choices[0].message.content)
print(f"(finish_reason={r.choices[0].finish_reason})")

# 3. The modern way to get the same guarantee: ask for a shape, not a marker.
print(f"\n--- {CURRENT_MODEL}, asking for a shape instead ---")
r2 = client.chat.completions.create(
    model=CURRENT_MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 3 as JSON."}],
    max_completion_tokens=200,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "counted",
            "schema": {
                "type": "object",
                "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}},
                "required": ["numbers"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
)
print(r2.choices[0].message.content)
print("\nThe cut-off is now guaranteed by the schema, not by a string you hoped for.")
