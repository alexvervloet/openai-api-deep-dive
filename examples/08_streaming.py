"""
Example 08 — streaming responses.
=================================

By default, `chat.completions.create` waits until the *entire* answer is ready,
then hands it to you in one piece. With `stream=True`, the API instead sends the
answer back in small pieces ("chunks") as the model generates them — exactly like
you see text appear word-by-word in ChatGPT.

Why stream?
  - Perceived speed: the user sees the first words almost immediately instead of
    staring at a blank screen.
  - Long answers: you can start processing/displaying before it's finished.

How it works:
  - The call returns an *iterator* instead of a single response object.
  - Each chunk carries a `delta` — the new bit of content since the last chunk.
  - `delta.content` is often an empty string (e.g. the very first chunk just
    opens the message), so we guard for None/empty.
  - To get token `usage` while streaming, you must opt in with
    `stream_options={"include_usage": True}`; it arrives in the final chunk.

Run it:

    python examples/08_streaming.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) and try again.")

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about streaming data."}],
    stream=True,
    stream_options={"include_usage": True},
)

usage = None
for chunk in stream:
    # The last chunk carries usage but no choices, so check before indexing.
    if chunk.usage is not None:
        usage = chunk.usage
    if chunk.choices:
        piece = chunk.choices[0].delta.content
        if piece:
            # end="" + flush so the text appears live instead of line-buffered.
            print(piece, end="", flush=True)

print("\n\n--- usage ---")
print(usage)
