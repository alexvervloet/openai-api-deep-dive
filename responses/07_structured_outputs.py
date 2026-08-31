"""
Responses 07: get a validated object back instead of a string.

Examples 09 and 14 did this on Chat Completions with `response_format`. That
parameter does not exist here. The Responses equivalent is `text.format`, and
the rename is the whole migration hazard: the old key is not an error you can
see, it is a key the endpoint does not read.

Two levels, same as before. `responses.create` with a `text.format` of
`json_schema` guarantees the text conforms to a schema you wrote, and you still
call `json.loads` yourself. `responses.parse` takes a Pydantic model, builds the
schema from it, and hands back an instance on `output_parsed`.

The part worth internalising is what happens when the model runs out of room
mid-object. A schema constrains the shape of text the model may emit. It
cannot promise the model finishes emitting it. `responses.create` reports that
honestly: `status="incomplete"` and a truncated string. `responses.parse` does
not hand you a None to check, it raises `pydantic.ValidationError` from inside
the SDK, because half an object cannot be validated into a whole one. The
convenience of parsing is also a failure mode you have to catch.

Predict before running
    The third and fourth requests cap output at 16 tokens. One returns, one
    raises. Which is which, and what does the caller have to write differently?

Run
    secrun python responses/07_structured_outputs.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see docs/SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
NOTE = "checkout was down about 45 minutes last night, we called it a sev2"
client = OpenAI()


class Incident(BaseModel):
    """The shape we want back. The SDK turns this into a strict JSON schema."""

    service: str
    severity: str
    minutes_down: int


# 1. The explicit schema. Note `text={"format": ...}`, not `response_format=`.
raw = client.responses.create(
    model=MODEL,
    input=NOTE,
    text={
        "format": {
            "type": "json_schema",
            "name": "incident",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "severity": {"type": "string"},
                    "minutes_down": {"type": "integer"},
                },
                "required": ["service", "severity", "minutes_down"],
                "additionalProperties": False,
            },
        }
    },
    max_output_tokens=300,
)
print("json_schema format, still a string until you parse it:")
print(f"  output_text: {raw.output_text}")
print(f"  json.loads:  {json.loads(raw.output_text)}")

# 2. The same result as a validated Python object.
parsed = client.responses.parse(
    model=MODEL,
    input=NOTE,
    text_format=Incident,
    max_output_tokens=300,
)
incident = parsed.output_parsed
print("\nresponses.parse, validated instance:")
print(f"  type: {type(incident).__name__}")
print(f"  value: {incident}")
if incident is not None:
    print(f"  minutes_down is a real int: {incident.minutes_down + 1}")

# 3. The same schema, truncated, through create. Partial text, honest status.
truncated = client.responses.create(
    model=MODEL,
    input=NOTE,
    text={
        "format": {
            "type": "json_schema",
            "name": "incident",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "severity": {"type": "string"},
                    "minutes_down": {"type": "integer"},
                },
                "required": ["service", "severity", "minutes_down"],
                "additionalProperties": False,
            },
        }
    },
    max_output_tokens=16,
)
print("\nSame schema, 16-token cap, via create:")
print(f"  status: {truncated.status}")
reason = truncated.incomplete_details.reason if truncated.incomplete_details else None
print(f"  incomplete reason: {reason}")
print(f"  output_text: {truncated.output_text!r}")

# 4. The same truncation through parse. There is no partial object to hand back.
print("\nSame schema, 16-token cap, via parse:")
try:
    client.responses.parse(
        model=MODEL,
        input=NOTE,
        text_format=Incident,
        max_output_tokens=16,
    )
except ValidationError as exc:
    print(f"  raised {type(exc).__name__}: {exc.errors()[0]['type']}")
    print("  the convenience of parsing is also a failure mode you must catch")
else:
    print("  the model finished inside the cap this time; lower it and rerun")

print(
    "\nA schema constrains what the model may emit, not whether it finishes. "
    "Check status when you call create, and catch ValidationError when you call "
    "parse, the same way you check finish_reason on Chat Completions."
)
