"""
Responses 02: continue a conversation by response ID.

`previous_response_id` saves the client from sending the transcript again. It
does not make earlier input tokens free. OpenAI still bills the context from the
chain on later turns. Instructions are also per request, so this example repeats
them on the follow-up instead of assuming they carry forward.

The API stores response objects for 30 days by default. Use a Conversation when
you need a durable server-side conversation rather than a response chain.

Predict before running
    The second HTTP request contains one short input string. Will its billed
    input-token count represent only that string, or the usable chain context?

Run
    secrun python responses/02_conversation_state.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
INSTRUCTIONS = "Answer in one short sentence. Prefix every answer with STATE:"
client = OpenAI()

first = client.responses.create(
    model=MODEL,
    instructions=INSTRUCTIONS,
    input="Remember that the deployment code is saffron-17.",
    max_output_tokens=300,
)

follow_up = "What is the deployment code?"
second = client.responses.create(
    model=MODEL,
    instructions=INSTRUCTIONS,
    input=follow_up,
    previous_response_id=first.id,
    max_output_tokens=300,
)

print(f"First reply:  {first.output_text}")
print(f"Second reply: {second.output_text}")
print("\nState and billing evidence:")
print(f"  previous response sent: {first.id}")
print(f"  response links to it:   {second.previous_response_id}")
print(f"  new input characters:   {len(follow_up)}")
if second.usage is not None:
    print(f"  billed input tokens:    {second.usage.input_tokens}")

print(
    "\nThe ID shrinks what your client uploads. It does not remove earlier "
    "turns from the model context or the token bill."
)
