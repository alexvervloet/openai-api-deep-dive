"""
Responses 08: a Conversation is a durable object, not a chain of responses.

Example 02 chained turns with `previous_response_id`. Each response pointed at
the one before it, and the chain lived only as long as your code held the latest
ID. A Conversation inverts that. You create the object first, get an ID back,
and every response you tag with it deposits its input and output items into that
object. The transcript now belongs to the conversation, not to whichever
response happened to run last.

Which to reach for:

  previous_response_id  one process, one session, you already hold the last ID.
  conversation          items must outlive the process, or be read and appended
                        to by more than one job, device, or worker.

Two things the pairing does not do. It is not a token discount: the model still
receives the accumulated context and you are still billed for those input tokens
every turn, exactly as in example 02. And the two mechanisms are mutually
exclusive, so sending both on one request is a 400, which this script provokes
on purpose rather than asserting. The server's own wording for that error is
truncated mid-word at the time of writing. That is upstream, not a bug here,
and it is a fair reminder to key your handling on `code` rather than on the
prose in `message`.

A Conversation also outlives your script. It holds whatever your users said, so
it is storage you are now responsible for. This example deletes the one it made.

Predict before running
    The second turn uploads one short question. Look at its billed input tokens.
    Does storing the transcript server-side change that number?

Run
    secrun python responses/08_conversation_object.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
INSTRUCTIONS = "Answer in one short sentence."
client = OpenAI()

conversation = client.conversations.create(metadata={"purpose": "responses-dive-08"})
print(f"Created conversation: {conversation.id}")

try:
    first = client.responses.create(
        model=MODEL,
        conversation=conversation.id,
        instructions=INSTRUCTIONS,
        input="Remember that the deployment code is saffron-17.",
        max_output_tokens=300,
    )
    follow_up = "What is the deployment code?"
    second = client.responses.create(
        model=MODEL,
        conversation=conversation.id,
        instructions=INSTRUCTIONS,
        input=follow_up,
        max_output_tokens=300,
    )

    print(f"\nFirst reply:  {first.output_text}")
    print(f"Second reply: {second.output_text}")

    # Nothing linked these two responses to each other. The conversation did.
    print("\nWhat carried the state:")
    print(f"  second.previous_response_id: {second.previous_response_id}")
    print(f"  second.conversation:         {second.conversation}")
    print(f"  new input characters:        {len(follow_up)}")
    if second.usage is not None:
        print(f"  billed input tokens:         {second.usage.input_tokens}")

    # The items are server-side and readable. Newest first.
    print("\nItems now stored on the conversation, newest first:")
    for item in client.conversations.items.list(conversation.id):
        parts = getattr(item, "content", None) or []
        text = " ".join(getattr(part, "text", "") for part in parts).strip()
        print(f"  {item.type:8} {getattr(item, 'role', '-'):9} {text[:60]!r}")

    # The two state mechanisms are alternatives, and the API says so.
    print("\nAsking for both mechanisms at once:")
    try:
        client.responses.create(
            model=MODEL,
            conversation=conversation.id,
            previous_response_id=first.id,
            input="Which one wins?",
            max_output_tokens=100,
        )
    except BadRequestError as exc:
        # exc.body is already the inner error object, not the {"error": ...} wrapper.
        body = exc.body if isinstance(exc.body, dict) else {}
        print(f"  {exc.status_code} {exc.code}")
        print(f"  {body.get('message', exc.message)}")
    else:
        print("  the request was accepted; the API changed, check the guide")

finally:
    deleted = client.conversations.delete(conversation.id)
    print(f"\nDeleted conversation: {deleted.id} (deleted={deleted.deleted})")

print(
    "\nA response chain is a pointer your process must keep. A Conversation is an "
    "object you must manage. Neither reduces the context the model reads or the "
    "input tokens you pay for."
)
