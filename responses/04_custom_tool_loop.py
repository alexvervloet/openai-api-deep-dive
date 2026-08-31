"""
Responses 04: execute a custom function through an explicit tool loop.

Custom tools are requests, not remote procedure calls. The model emits a typed
`function_call` item. Your application validates its name and arguments, runs
the allowed function, and returns a `function_call_output` item with the same
call ID. Nothing executes merely because the model named it.

Note which controls actually bound this loop. `tool_choice` and
`parallel_tool_calls` shape what the model may ask for. The dispatch table and
the argument check decide what runs. `max_tool_calls` looks like it belongs
here and does not: it caps built-in tools such as web search, so on a custom
function it is silently inert. Example 05 is where it has an effect.

Predict before running
    The first response is forced to call one function. Which output item will
    carry the call, and which field connects its result to the second request?

Run
    secrun python responses/04_custom_tool_loop.py
"""

import json
import os
import sys
from collections.abc import Callable

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
TOOL_NAME = "get_current_weather"
client = OpenAI()


def get_current_weather(city: str) -> dict[str, str]:
    """Return deterministic sample weather for one supported city."""
    sample_weather = {
        "Paris": "18 C, light rain",
        "Tokyo": "27 C, sunny",
    }
    return {"city": city, "conditions": sample_weather[city]}


dispatch: dict[str, Callable[[str], dict[str, str]]] = {
    TOOL_NAME: get_current_weather,
}

tools: list[ToolParam] = [
    {
        "type": "function",
        "name": TOOL_NAME,
        "description": "Get sample weather for a supported city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": ["Paris", "Tokyo"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

first = client.responses.create(
    model=MODEL,
    input="What is the weather in Tokyo?",
    tools=tools,
    tool_choice={"type": "function", "name": TOOL_NAME},
    parallel_tool_calls=False,
)

tool_outputs: ResponseInputParam = []
call_ids: list[str] = []
for item in first.output:
    if item.type != "function_call":
        continue

    function = dispatch.get(item.name)
    if function is None:
        sys.exit(f"Refusing unregistered function: {item.name}")

    try:
        arguments = json.loads(item.arguments)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Tool arguments were not valid JSON: {exc}") from exc

    city = arguments.get("city") if isinstance(arguments, dict) else None
    if city not in {"Paris", "Tokyo"}:
        sys.exit(f"Refusing unsupported city: {city!r}")

    result = function(city)
    print(f"Model requested {item.name}({arguments})")
    print(f"Application returned {result}")
    tool_outputs.append(
        {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result),
        }
    )
    call_ids.append(item.call_id)

if not tool_outputs:
    sys.exit(f"Expected a {TOOL_NAME} call, but received {[item.type for item in first.output]}")

second = client.responses.create(
    model=MODEL,
    previous_response_id=first.id,
    input=tool_outputs,
    tools=tools,
    tool_choice="none",
    instructions="Answer in one sentence using only the supplied tool result.",
    max_output_tokens=300,
)

print(f"\nFinal answer: {second.output_text}")
print(f"Call IDs returned: {call_ids}")
