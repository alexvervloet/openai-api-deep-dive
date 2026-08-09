"""
Example 26: the Responses API, OpenAI's other endpoint.

Every example so far has used `client.chat.completions.create(...)`. There is a
second, newer endpoint: `client.responses.create(...)`. This is not a
replacement you must rush to, and Chat Completions is not deprecated. But you
will meet Responses in OpenAI's own docs and samples, so you should know what it
is, what it buys you, and what it costs you.

WHY IT EXISTS
    Chat Completions is stateless and text-in/text-out. Everything else, memory,
    tool execution, file search, is your job. OpenAI's first answer to that was
    the Assistants API, which is **being shut down on 2026-08-26**. The Responses
    API (plus the Conversations API) is what replaced it: the same stateful,
    tool-running ideas, on a cleaner endpoint.

WHAT CHANGES IN THE REQUEST
    | Chat Completions            | Responses                  |
    |-----------------------------|----------------------------|
    | `messages=[...]`            | `input=...` (str or list)  |
    | a `system` role message     | `instructions="..."`       |
    | `max_completion_tokens`     | `max_output_tokens`        |
    | `.choices[0].message.content` | `.output_text`           |

WHAT YOU ACTUALLY GAIN
    Two things you cannot get from Chat Completions at all:

    1. Server-side conversation state. Pass `previous_response_id` and OpenAI
       keeps the history for you, so you stop re-uploading the whole transcript
       every turn. Compare example 12, where *you* carry the list.

    2. Hosted tools. `{"type": "web_search"}` runs on OpenAI's infrastructure.
       No tool loop, no execution on your machine, no round trip back. Compare
       example 10, where you run the function and feed the result back yourself.

WHAT IT COSTS YOU: PORTABILITY
    This is the honest tradeoff, and it is the reason this repo teaches Chat
    Completions first. `/v1/chat/completions` is the closest thing the industry
    has to a lingua franca: Ollama, LM Studio, vLLM, LiteLLM, Together, Groq and
    most others implement it, which is exactly why example 17 can point the same
    client at a local model by changing one `base_url`. The Responses API is
    OpenAI's own shape. Build on it and that swap stops being free.

    So: reach for Responses when you want the hosted tools or the server-side
    state badly enough to accept the lock-in. Stay on Chat Completions when
    portability matters, which for most of what this repo builds, it does.

Run it:

    secrun python examples/26_responses_api.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()
MODEL = "gpt-5.4-nano"

# --- 1. The same request you already know, in the other shape ----------------
# `input` replaces `messages`, `instructions` replaces the system role, and
# `.output_text` saves you walking into `.choices[0].message.content`.
print("--- 1. a plain request ---")
first = client.responses.create(
    model=MODEL,
    instructions="Answer in one short sentence.",
    input="Why is the sky blue?",
    max_output_tokens=200,
)
print(first.output_text)
print(f"(response id: {first.id})")

# --- 2. Server-side state: the part Chat Completions cannot do ---------------
# In example 12 you kept a `messages` list and re-sent it every turn. Here you
# send ONE id instead. OpenAI holds the transcript; your follow-up is tiny no
# matter how long the conversation gets.
print("\n--- 2. a follow-up, without re-sending the history ---")
second = client.responses.create(
    model=MODEL,
    input="Explain that again, but for a six-year-old.",
    previous_response_id=first.id,
    max_output_tokens=200,
)
print(second.output_text)
print(f"(input tokens this turn: {second.usage.input_tokens}, and the model still had the context)")

# --- 3. A hosted tool: OpenAI runs it, not you -------------------------------
# Example 10 taught the tool loop: model asks, YOU execute, you feed the result
# back. A hosted tool skips all of that. There is no loop here because the
# search already happened on OpenAI's side before this returned.
#
# Note the question. The model decides whether a search is worth it, and it will
# happily answer "what is the capital of France" from memory without searching
# at all. To see the tool actually fire, you have to ask something it cannot
# know: anything about *now*.
print("\n--- 3. a hosted tool (web_search), with no tool loop ---")
searched = client.responses.create(
    model=MODEL,
    tools=[{"type": "web_search"}],
    input="What is one technology news story from this week? One line, cite the source URL.",
)
print(searched.output_text)

# The `output` list shows the machinery: one or more web_search_call items the
# API ran for you, then the message it produced from the results. If you only
# see 'message', the model judged it did not need to search.
print("\noutput items the API produced on your behalf:")
for item in searched.output:
    print(f"  - {item.type}")

print(
    "\nTakeaway: Responses buys you hosted tools and server-side state, and\n"
    "charges you portability for them. Chat Completions still runs everywhere;\n"
    "this endpoint runs on OpenAI. Pick per project, not per fashion."
)
