#!/usr/bin/env python3
"""
extract.py: Turn free-form text into validated structured data.

The second hands-on tool of the repo, and a companion to `ask.py`. Where `ask.py`
gets you *prose* about a file, `extract.py` gets you *data*: point it at messy
free-form text (meeting notes, an email, a support ticket) and it pulls out a
clean, typed, validated structure you could drop straight into a database.

It ties together three things from the examples:
  - Structured outputs + Pydantic validation (examples 09 & 14): the model is
    constrained to a schema and the reply comes back as a validated object.
  - Rich output (example 15): the result is shown as a Markdown summary and a
    real table, not a wall of JSON.
  - Token/cost awareness (utils/tokens.py, utils/pricing.py): same --dry-run
    discipline as ask.py: see the price before you spend.

Examples
--------
  # Extract action items from the sample meeting notes
  secrun python hands_on/extract.py snippets/meeting_notes.txt

  # See tokens + estimated cost without calling the API
  secrun python hands_on/extract.py snippets/meeting_notes.txt --dry-run

  # Use a more capable model for messier text
  secrun python hands_on/extract.py snippets/meeting_notes.txt --model gpt-4o

  # Get the raw validated JSON instead of the pretty tables
  secrun python hands_on/extract.py snippets/meeting_notes.txt --json
"""

import argparse
import os
import sys
from enum import Enum

# Make the repo root importable so `utils.*` is resolvable when running from
# any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from utils.pricing import estimate_cost, format_cost
from utils.tokens import count_message_tokens


# --- The shape we want out of the text -----------------------------------------
# These Pydantic models ARE the contract. The SDK turns them into a strict schema
# the model must follow, and validates the reply back into these types. Field
# descriptions are sent along as guidance, so they double as extraction hints.
class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ActionItem(BaseModel):
    task: str = Field(description="The concrete thing that needs doing.")
    owner: str = Field(description="Who is responsible; 'unassigned' if unclear.")
    due: str | None = Field(
        default=None,
        description="When it's due if stated (e.g. 'Friday', 'end of month'); else null.",
    )
    priority: Priority = Field(description="Inferred urgency from the text.")


class Extraction(BaseModel):
    summary: str = Field(description="A one-sentence summary of the text.")
    action_items: list[ActionItem]


SYSTEM_PROMPT = (
    "You extract structured action items from free-form notes. Capture every task "
    "that someone needs to do. Infer the owner and priority from context; if an "
    "owner truly isn't implied, use 'unassigned'. Only include a due date if the "
    "text actually mentions one."
)


def build_messages(text: str) -> list[dict]:
    """A two-message request: standing instructions + the raw text to mine."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract action items from these notes:\n\n{text}"},
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract validated structured data from a free-form text file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to the free-form text to extract from.")
    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="Model to use (default: gpt-5.4-nano, the cheap workhorse).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validated data as raw JSON instead of formatted tables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show token/cost estimates but DON'T call the API. No money spent.",
    )
    return parser.parse_args(argv)


def render(console: Console, data: Extraction) -> None:
    """Show the extraction as a Markdown summary + a table of action items."""
    console.print(Markdown(f"**Summary:** {data.summary}"))

    if not data.action_items:
        console.print("\n[italic]No action items found.[/italic]")
        return

    table = Table(title="Action items", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Task", style="white")
    table.add_column("Owner", style="cyan")
    table.add_column("Due", style="yellow")
    table.add_column("Priority", justify="center")

    # Color the priority so high-urgency rows jump out.
    colors = {Priority.high: "red", Priority.medium: "yellow", Priority.low: "green"}
    for i, item in enumerate(data.action_items, 1):
        color = colors[item.priority]
        table.add_row(
            str(i),
            item.task,
            item.owner,
            item.due or "n/a",
            f"[{color}]{item.priority.value}[/{color}]",
        )

    console.print(table)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    console = Console()

    # 1. Read the text file.
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"Could not read {args.file!r}: {e}", file=sys.stderr)
        return 1

    # 2. Count input tokens and estimate cost BEFORE spending anything.
    messages = build_messages(text)
    input_tokens = count_message_tokens(messages, model=args.model)

    print(f"Model:         {args.model}")
    print(f"Input tokens:  {input_tokens:,} (estimated locally with tiktoken)")
    try:
        assumed_output = 600  # structured extractions are usually compact
        est = estimate_cost(args.model, input_tokens, assumed_output)
        print(f"Est. cost:     {format_cost(est)} "
              f"(assuming ~{assumed_output:,} output tokens)")
    except KeyError as e:
        print(f"Est. cost:     unknown ({e})")

    if args.dry_run:
        print("\n[--dry-run] Stopping before the API call. No money spent.")
        return 0

    # 3. Make the real call.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\nOPENAI_API_KEY is not set. Store it in your keychain and run under `secrun` "
            "(see SECRETS.md), or run with --dry-run to skip the API call.",
            file=sys.stderr,
        )
        return 1

    from openai import OpenAI  # imported lazily so --dry-run needs no key
    client = OpenAI()

    print("\nCalling the API...\n")
    # response_format=Extraction → the model is constrained to our schema and the
    # reply is parsed + validated back into an Extraction instance for us.
    response = client.chat.completions.parse(
        model=args.model,
        messages=messages,  # type: ignore[arg-type]
        response_format=Extraction,
    )
    message = response.choices[0].message

    if message.refusal:
        print(f"Model refused: {message.refusal}", file=sys.stderr)
        return 1

    data = message.parsed
    assert data is not None  # not a refusal, so parsing succeeded

    # 4. Output: either raw JSON or the formatted view.
    if args.json:
        print(data.model_dump_json(indent=2))
    else:
        render(console, data)

    # 5. Report authoritative usage and real cost.
    usage = response.usage
    if usage is not None:
        print(f"\nTokens used:   {usage.prompt_tokens:,} in + "
              f"{usage.completion_tokens:,} out = {usage.total_tokens:,} total")
        try:
            actual = estimate_cost(
                args.model, usage.prompt_tokens, usage.completion_tokens
            )
            print(f"Actual cost:   {format_cost(actual)}")
        except KeyError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
