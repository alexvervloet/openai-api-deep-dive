"""
Example 10 — function / tool calling.
=====================================

The model can't run code, browse, or query your database. But it *can* tell you
"I'd like you to call this function with these arguments" — and then you run the
function and hand the result back. This is "tool calling," and it's how chatbots
get the ability to actually *do* things.

The dance has four steps:

  1. You describe your tools (name, what they do, their parameters as a schema)
     and send them alongside the user's message.
  2. The model replies not with prose but with a `tool_calls` request: which
     function, and what arguments (as JSON).
  3. YOU execute the real function with those arguments.
  4. You send the result back (as a `tool` role message) and the model writes the
     final natural-language answer using it.

The model never runs your code — it only *asks*. You stay in control of what
actually executes.

Run it:

    python examples/10_function_calling.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) and try again.")

client = OpenAI()


# --- The actual function the model is allowed to ask us to run. ---
# In real life this might hit a weather API; here we fake it so the example is
# self-contained.
def get_current_weather(city: str) -> dict:
    fake_db = {"Paris": "18°C, light rain", "Tokyo": "27°C, sunny"}
    return {"city": city, "conditions": fake_db.get(city, "unknown")}


# --- Step 1: describe the tool to the model. ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Paris"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]

messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]

# First call: the model decides it needs the tool.
first = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)
reply = first.choices[0].message

# --- Step 2 & 3: the model asked for tool calls; we run them. ---
# Append the model's tool-call message to the history first — the API requires
# every tool result to follow the assistant message that requested it.
messages.append(reply)

for call in reply.tool_calls or []:
    args = json.loads(call.function.arguments)
    print(f"[model requested: {call.function.name}({args})]")
    result = get_current_weather(**args)
    # --- Step 4: return the result, tagged with the call's id. ---
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    })

# Second call: the model now has the data and writes the final answer.
second = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)
print("\n" + second.choices[0].message.content)
