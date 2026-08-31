#!/usr/bin/env python3
"""
ask.py: Ask a question about a code snippet.

This is the main hands-on tool of the repo. You point it at a file and ask a
question about it; it builds a chat-completions request, shows you the token
count and an estimated cost, calls the model, and prints the answer plus the
*actual* usage and cost.

It also doubles as a guided tour of the request parameters, every one of which
is exposed as a command-line flag with a short explanation in --help.

Examples
--------
  # Simplest form: explain a file
  secrun python hands_on/ask.py snippets/buggy.py "What does this code do?"

  # Find a bug, with a more capable model
  secrun python hands_on/ask.py snippets/buggy.py "Is there a bug here?" --model gpt-4o

  # See the cost *before* spending anything (no API call is made)
  secrun python hands_on/ask.py snippets/buggy.py "Explain this" --dry-run

  # Turn creativity down to 0 for deterministic, focused answers
  secrun python hands_on/ask.py snippets/buggy.py "Rewrite this cleanly" --temperature 0

  # Cap the answer length and stop at a marker
  secrun python hands_on/ask.py snippets/buggy.py "List 3 issues" --max-tokens 200 --stop "4."
"""

import argparse
import os
import sys

# Make the repo root importable so `utils.*` is resolvable when running from
# any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from utils.pricing import estimate_cost, format_cost
from utils.tokens import count_message_tokens


# A default "system" message. The system role sets the assistant's behavior and
# persona for the whole conversation: see examples/02_roles.py for more.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, friendly senior software engineer helping someone "
    "understand code. Be concrete, point to specific lines, and keep answers "
    "tight. If you spot a bug, explain why it's a bug before suggesting a fix."
)


def build_messages(system_prompt: str, code: str, question: str) -> list[dict]:
    """Assemble the chat-format message list.

    A chat request is a *list of messages*, each with a `role` and `content`.
    Here we use two messages:
      - system:    the standing instructions / persona.
      - user:      the code, clearly fenced so the model knows where it starts
                   and ends, followed by the question.
    The model will reply with an `assistant` message, the part you don't
    write; the API generates it.
    """
    user_content = (
        f"Here is a code snippet:\n\n```\n{code}\n```\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask an OpenAI chat model a question about a code file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to the code snippet to ask about.")
    parser.add_argument("question", help="Your question about the code.")

    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="Model to use (default: gpt-5.4-nano, the cheap workhorse).",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the system prompt (the assistant's standing instructions).",
    )

    # ---- The "knobs". Each one shapes the response. ----
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=(
            "Randomness, 0.0–2.0 (default 0.7). 0 = focused and near-"
            "deterministic; higher = more varied/creative. For factual code "
            "questions, low is usually better."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=(
            "Nucleus sampling, 0.0–1.0. The model samples only from the most "
            "likely tokens whose probabilities sum to top_p. It's an alternative "
            "to temperature. OpenAI recommends changing one, not both."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Hard cap on how many tokens the model may *generate*. Protects "
            "against runaway (expensive) answers. Note: this limits the output, "
            "not the input."
        ),
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        metavar="SEQUENCE",
        help=(
            "A string that, if generated, makes the model stop immediately "
            "(the stop text itself is not included). Repeat the flag for up to "
            "4 stop sequences."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the request and show token/cost estimates, but DON'T call "
             "the API. Great for learning without spending money.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # 1. Read the code file.
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except OSError as e:
        print(f"Could not read {args.file!r}: {e}", file=sys.stderr)
        return 1

    # 2. Build the messages and count input tokens BEFORE sending anything.
    messages = build_messages(args.system, code, args.question)
    input_tokens = count_message_tokens(messages, model=args.model)

    print(f"Model:         {args.model}")
    print(f"Input tokens:  {input_tokens:,} (estimated locally with tiktoken)")
    try:
        # We don't yet know the output size, so estimate against the cap (or a
        # nominal 500 tokens if uncapped) just to give a ballpark.
        assumed_output = args.max_tokens or 500
        est = estimate_cost(args.model, input_tokens, assumed_output)
        print(f"Est. cost:     {format_cost(est)} "
              f"(assuming ~{assumed_output:,} output tokens)")
    except KeyError as e:
        print(f"Est. cost:     unknown ({e})")

    if args.dry_run:
        print("\n[--dry-run] Stopping before the API call. No money spent.")
        return 0

    # 3. Make the real call. We import + construct the client here so --dry-run
    #    works even without a key set.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\nOPENAI_API_KEY is not set. Store it in your keychain and run under `secrun` "
            "(see ../docs/SECRETS.md), or run with --dry-run to skip the API call.",
            file=sys.stderr,
        )
        return 1

    from openai import OpenAI  # imported lazily so --dry-run has no dependency
    client = OpenAI()  # reads OPENAI_API_KEY from the environment automatically

    # Only pass optional knobs if the user actually set them, so we use the
    # API's defaults otherwise.
    request: dict = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        request["top_p"] = args.top_p
    if args.max_tokens is not None:
        # The CLI flag stays --max-tokens because that is what people call it,
        # but the wire parameter is max_completion_tokens: gpt-5.x rejects the
        # old max_tokens name outright.
        request["max_completion_tokens"] = args.max_tokens
    if args.stop is not None:
        # `stop` is not supported on the gpt-5 line (see examples/06). Rather
        # than let the API 400 on a flag the user deliberately passed, say so
        # and carry on without it.
        if args.model.startswith("gpt-5"):
            print(
                f"note: --stop is not supported on {args.model} and will be ignored.\n"
                f"      Use --max-tokens for a length cap, or a JSON schema for a shape.",
                file=sys.stderr,
            )
        else:
            request["stop"] = args.stop

    print("\nCalling the API...\n")
    response = client.chat.completions.create(**request)

    # 4. Print the answer.
    choice = response.choices[0]
    print("=" * 70)
    print(choice.message.content)
    print("=" * 70)

    # `finish_reason` tells you WHY the model stopped:
    #   "stop"   -> it finished naturally (or hit a stop sequence)
    #   "length" -> it ran into your --max-tokens cap (answer is truncated!)
    print(f"\nfinish_reason: {choice.finish_reason}")

    # 5. Report the AUTHORITATIVE usage from the response, and real cost.
    usage = response.usage
    print(f"Tokens used:   {usage.prompt_tokens:,} in + "
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
