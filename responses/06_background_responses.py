"""
Responses 06: manage an asynchronous response by ID.

Background mode returns control before long work finishes. The response then
moves through `queued` or `in_progress` until it reaches a terminal status. This
script keeps start, check, wait, and cancel as separate commands so the response
ID is visibly the handle that survives a dropped client connection.

This example sets `store=False`. OpenAI still keeps background response data
temporarily so it can execute and be polled, for roughly ten minutes according
to the background-mode guide. That makes background work incompatible with a
strict promise that request data never reaches temporary storage.

Run
    secrun python responses/06_background_responses.py start
    secrun python responses/06_background_responses.py check RESP_ID
    secrun python responses/06_background_responses.py wait RESP_ID
    secrun python responses/06_background_responses.py cancel RESP_ID
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
ACTIVE_STATUSES = {"queued", "in_progress"}
client = OpenAI()


def show_response(response: Response) -> None:
    """Print the lifecycle fields needed to decide the next action."""
    print(f"id: {response.id}")
    print(f"status: {response.status}")
    if response.status == "completed":
        print(f"\n{response.output_text}")
    elif response.error is not None:
        print(f"error: {response.error.code}: {response.error.message}")


def start_response() -> None:
    """Start one background job and print commands that can resume it."""
    response = client.responses.create(
        model=MODEL,
        instructions="Write a concise technical note with four titled sections.",
        input="Explain how idempotency, timeouts, retries, and cancellation interact.",
        max_output_tokens=1_200,
        background=True,
        store=False,
    )
    show_response(response)
    print("\nResume from another process:")
    print(f"  secrun python responses/06_background_responses.py wait {response.id}")
    print(f"  secrun python responses/06_background_responses.py cancel {response.id}")


def wait_for_response(response_id: str, interval: float, timeout: float) -> None:
    """Poll while the API reports an active status, bounded by a deadline."""
    deadline = time.monotonic() + timeout
    response = client.responses.retrieve(response_id)
    previous_status = None

    while response.status in ACTIVE_STATUSES:
        if response.status != previous_status:
            print(f"status: {response.status}")
            previous_status = response.status
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Stopped polling after {timeout:g} seconds. The job may still be "
                f"running; retry the wait command with {response_id}."
            )
        time.sleep(interval)
        response = client.responses.retrieve(response_id)

    show_response(response)


def build_parser() -> argparse.ArgumentParser:
    """Build the four-command interface used to expose the response lifecycle."""
    parser = argparse.ArgumentParser(description="Manage a background Response.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start", help="start a background response")

    check = commands.add_parser("check", help="retrieve the current state once")
    check.add_argument("response_id")

    wait = commands.add_parser("wait", help="poll until the response is terminal")
    wait.add_argument("response_id")
    wait.add_argument("--interval", type=float, default=2.0)
    wait.add_argument("--timeout", type=float, default=120.0)

    cancel = commands.add_parser("cancel", help="request cancellation")
    cancel.add_argument("response_id")
    return parser


def main() -> None:
    """Dispatch one lifecycle operation selected by the learner."""
    args = build_parser().parse_args()
    if args.command == "start":
        start_response()
    elif args.command == "check":
        show_response(client.responses.retrieve(args.response_id))
    elif args.command == "wait":
        if args.interval <= 0 or args.timeout <= 0:
            raise SystemExit("--interval and --timeout must be greater than zero.")
        wait_for_response(args.response_id, args.interval, args.timeout)
    elif args.command == "cancel":
        show_response(client.responses.cancel(args.response_id))


if __name__ == "__main__":
    main()
