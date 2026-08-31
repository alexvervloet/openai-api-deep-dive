"""
Responses 03: consume typed streaming events.

A Responses stream is not a sequence of interchangeable text chunks. It can
carry lifecycle, output-item, content-part, text-delta, tool, and error events.
This example renders text deltas while still recording every event type it saw.

The terminal branch matches on class rather than on a set of type strings. Both
select the same three events at runtime, but only `isinstance` tells a type
checker which member of the event union it is holding, so `event.response` is
checked instead of assumed.

Predict before running
    Will `response.output_text.delta` be the only event type in the stream?
    Watch the event summary after the answer finishes.

Run
    secrun python responses/03_streaming_events.py
"""

import os
import sys
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseUsage,
)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
client = OpenAI()

stream = client.responses.create(
    model=MODEL,
    instructions="Explain the idea in three short sentences.",
    input="Why should a streaming client switch on event type?",
    max_output_tokens=500,
    stream=True,
)

event_counts: Counter[str] = Counter()
terminal_status = "stream ended without a terminal response event"
usage: ResponseUsage | None = None

for event in stream:
    event_counts[event.type] += 1

    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif isinstance(
        event, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)
    ):
        terminal_status = event.response.status
        usage = event.response.usage

print("\n\nEvent types received:")
for event_type, count in sorted(event_counts.items()):
    print(f"  {event_type}: {count}")

print(f"\nTerminal status: {terminal_status}")
if usage is not None:
    print(f"Output tokens: {usage.output_tokens}")

print(
    "\nRender delta events, but keep lifecycle and item events available for "
    "logging, tool handling, and failure reporting."
)
