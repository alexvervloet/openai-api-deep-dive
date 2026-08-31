"""
Responses 05: run a hosted web search and inspect its evidence.

A hosted tool runs inside OpenAI's service. Your process does not receive a
function call to execute. This example sets `tool_choice="required"` with only
one available tool, so the server must use web search. The returned output items
and source URLs provide independent evidence that the search path ran.

Hosted execution removes the client-side tool loop. It does not remove your
responsibility to approve the tool, restrict its scope, and handle its output as
untrusted external data.

Run
    secrun python responses/05_hosted_web_search.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
client = OpenAI()

response = client.responses.create(
    model=MODEL,
    instructions="Answer in two sentences and cite the sources you used.",
    input="What is the latest stable Python release, and when was it released?",
    tools=[{"type": "web_search", "search_context_size": "low"}],
    tool_choice="required",
    max_tool_calls=3,
    include=["web_search_call.action.sources"],
    max_output_tokens=500,
)

search_calls = [item for item in response.output if item.type == "web_search_call"]
if not search_calls:
    sys.exit(f"The required tool path did not run: {[item.type for item in response.output]}")

source_urls: set[str] = set()
for call in search_calls:
    if call.action.type != "search":
        continue
    for source in call.action.sources or []:
        source_urls.add(source.url)

print(response.output_text)
print("\nExecution evidence:")
print(f"  output item types: {[item.type for item in response.output]}")
print(f"  web-search call statuses: {[call.status for call in search_calls]}")
print("  source URLs returned by the tool:")
for url in sorted(source_urls):
    print(f"    {url}")

if not source_urls:
    print("    no source list was returned; inspect the message annotations instead")
