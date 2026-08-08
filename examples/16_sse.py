"""
Example 16: Server-Sent Events (SSE): the protocol under streaming.

Every streaming AI response, in the ChatGPT UI, in examples/08_streaming.py,
and in every production assistant, travels over a protocol called Server-Sent
Events (SSE). The OpenAI SDK parses it for you, but knowing the raw format
pays off: it's exactly what you'll produce when you build a backend that
forwards AI tokens to a browser.

The SSE wire format
-------------------
SSE is an ordinary HTTP response that stays open and drips data until the
server says it's done. The server sets:

    Content-Type: text/event-stream

Then sends events as plain text lines, each event terminated by a blank line:

    data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}]}\n
    \n
    data: {"id":"chatcmpl-...","choices":[{"delta":{"content":" world"}}]}\n
    \n
    data: [DONE]\n
    \n

Rules:
  - Lines starting with `data:` hold the payload.
  - A blank line terminates the current event.
  - `data: [DONE]` signals the end (OpenAI-specific; the SDK silently swallows it).
  - Other SSE fields (`event:`, `id:`, `retry:`) are valid per spec but
    OpenAI only uses `data:`.

This example has three parts:

  Part 1, raw events: print each chunk exactly as it arrives on the wire.
           The SDK gives us the parsed object; we format it back to text so
           you can see what the HTTP layer actually looks like.

  Part 2, token timing: measure time-to-first-token and generation throughput
           to understand the rhythm of incremental delivery.

  Part 3, partial accumulation: show how a server buffers tokens in memory
           to track progress and recover from interruptions.

The capstone that puts all of this into practice is hands_on/streaming_server.py
a FastAPI server that streams tokens over SSE to a real browser.

Run it:

    secrun python examples/16_sse.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

PROMPT = "Give three one-sentence reasons why streaming matters for AI UIs."

# ---------------------------------------------------------------------------
# Parts 1 & 2: stream with the SDK, print each chunk as a raw SSE event,
# and record per-token timing.
# ---------------------------------------------------------------------------

print("=== raw SSE events (as they appear on the wire) ===\n")

stream = client.chat.completions.create(
    model="gpt-5.4-nano",
    messages=[{"role": "user", "content": PROMPT}],
    stream=True,
    stream_options={"include_usage": True},  # usage arrives in the final chunk
)

start = time.perf_counter()
first_token_time: float | None = None
last_token_time: float | None = None
token_count = 0
partial: list[str] = []
usage = None

for chunk in stream:
    now = time.perf_counter()

    # Reconstruct the raw SSE line the SDK received and parsed.
    # In reality, the TCP stream contained exactly: "data: <json>\n\n"
    payload = json.dumps(chunk.model_dump(exclude_none=True))
    line = f"data: {payload}"
    print(line[:120] + ("..." if len(line) > 120 else ""))

    if chunk.choices:
        piece = chunk.choices[0].delta.content
        if piece:
            if first_token_time is None:
                first_token_time = now
                print(f"  ↳ first token in {now - start:.3f}s")
            last_token_time = now
            token_count += 1
            partial.append(piece)

    if chunk.usage:
        usage = chunk.usage

# The SDK silently swallows the final `data: [DONE]` line; shown here for
# completeness.
print("data: [DONE]")

# ---------------------------------------------------------------------------
# Part 3: assembled response and stats.
# ---------------------------------------------------------------------------

total_elapsed = time.perf_counter() - start

print("\n=== assembled response ===\n")
print("".join(partial))

print("\n--- stats ---")
print(f"Total time:          {total_elapsed:.2f}s")
if first_token_time is not None:
    print(f"Time to first token: {first_token_time - start:.3f}s")
if first_token_time and last_token_time and token_count > 1:
    gen_span = last_token_time - first_token_time
    tps = (token_count - 1) / gen_span if gen_span > 0 else 0
    print(f"Generation span:     {gen_span:.2f}s ({tps:.0f} tokens/s)")
print(f"Chunks with text:    {token_count}")
if usage:
    print(f"API usage:           {usage.prompt_tokens} in "
          f"+ {usage.completion_tokens} out")

# ---------------------------------------------------------------------------
# What you do next: building a server.
# ---------------------------------------------------------------------------

print("""
Key takeaway
------------
In a streaming server, each `piece` above becomes one SSE event you yield
to the browser:

    yield f"data: {json.dumps({'type': 'token', 'text': piece})}\\n\\n"

The browser's fetch + ReadableStream parses those lines and calls your handler
for each event, exactly the same loop but running client-side. See
hands_on/streaming_server.py for the full production-ready implementation.
""")
