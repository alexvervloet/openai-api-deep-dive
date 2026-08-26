"""
Example 12: multi-turn conversations (the API has no memory).

Here's the single most common surprise for newcomers:

  >> The API is STATELESS. It remembers nothing between requests. If you want the
  >> model to "remember" what was said earlier, *you* have to send the whole
  >> conversation back every single time.

There's no session, no conversation ID, no server-side history. Each call to
`chat.completions.create` is judged entirely on the `messages` list you hand it
right then. The illusion of a chatbot that remembers is built by you, the
caller, by appending each new turn to a growing list:

    [user] -> [user, assistant] -> [user, assistant, user, assistant] -> ...

That's it. Every example so far sent a fixed list; this one *grows* the list as
the conversation goes, which is all a chat app really is.

Run it (type a few messages, then `quit`):

    secrun python examples/12_conversation.py

Try this to feel the statelessness: tell it your name, then ask "what's my
name?". It works, because the earlier turns are still in the list. Now look at
`trim_history()` below: drop the early turns and the model genuinely forgets,
because for the model the conversation *is* whatever list you send.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

# The system message stays at index 0 for the whole conversation; the user and
# assistant turns accumulate after it. This list IS the conversation's memory.
messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a concise, friendly assistant."},
]


def trim_history(
    history: list[ChatCompletionMessageParam], max_turns: int = 10
) -> list[ChatCompletionMessageParam]:
    """Keep the system message + the most recent `max_turns` messages.

    Every turn you keep is re-sent (and re-billed) on the next request, so real
    apps cap the history. Drop the oldest turns and the model forgets them,
    proof that "memory" is just the list you choose to send.
    """
    if len(history) <= max_turns + 1:
        return history
    return history[:1] + history[-max_turns:]


print("Chat with the model. Type 'quit' to exit.\n")

while True:
    try:
        user_input = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    # 1. Append the user's turn to the running history.
    messages.append({"role": "user", "content": user_input})

    # 2. Send the ENTIRE history every time. That's what gives the model context.
    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages,
    )
    reply = response.choices[0].message.content or ""

    # 3. Append the model's turn too, so the next request includes it.
    messages.append({"role": "assistant", "content": reply})
    messages = trim_history(messages)

    print(f"bot> {reply}\n")
