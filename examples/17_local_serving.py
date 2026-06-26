"""
Example 17 — Local models: the same client, a different base_url.
=================================================================

The big idea of the whole repo — "send messages, get a message" — isn't tied to
OpenAI's servers. Local runtimes like **Ollama** and **llama.cpp** expose an
*OpenAI-compatible* endpoint, so the exact same `openai` SDK talks to a model
running on your own machine. You change two things and nothing else:

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

`base_url` points at the local server; `api_key` is required by the SDK but
ignored by the local server (any non-empty string works). Everything you learned —
roles, the sampling knobs, streaming, token usage — works unchanged.

Why bother? Privacy (data never leaves the machine), cost (no per-token bill),
and offline use. The trade-off is you run and scale the server yourself, and the
small local models are less capable than the hosted frontier ones.

    python examples/17_local_serving.py

This example needs a local server running; with none, it prints how to start one
and exits cleanly (no key, no crash).

Setup (one time):
    1. Install Ollama from https://ollama.com
    2. Pull a small model:   ollama pull llama3.2
    3. Re-run this script.   (Ollama serves on localhost:11434 automatically.)
"""

import os
import sys

from openai import OpenAI, APIConnectionError

# No real key needed — the local server ignores it, but the SDK requires a value.
BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LOCAL_MODEL", "llama3.2")

client = OpenAI(base_url=BASE_URL, api_key="local-no-key-needed")

print(f"Talking to a local model at {BASE_URL} (model: {MODEL})\n")

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "In one sentence, what is an API?"}],
    )
except APIConnectionError:
    print(f"No local server reachable at {BASE_URL}.")
    print("This example is optional — it needs a local runtime. To try it:")
    print("    1. Install Ollama from https://ollama.com")
    print(f"    2. Pull the model:   ollama pull {MODEL}")
    print("    3. Re-run this script.")
    print("\n(Nothing was charged and no key was used — local serving is free and offline.)")
    sys.exit(0)
except Exception as exc:  # model not pulled, etc. — report, don't crash the suite
    print(f"Reached the server but the call failed: {exc}")
    print(f"Most likely the model isn't pulled yet — run:  ollama pull {MODEL}")
    sys.exit(0)

print(response.choices[0].message.content)
print("\n--- usage ---")
print(response.usage)
print("\nNotice: this is example 01's code with one line changed (the base_url).")
print("Provider-agnostic by construction — the API shape is the abstraction.")
