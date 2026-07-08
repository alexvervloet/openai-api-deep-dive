"""
Example 22 — async & concurrency: many requests at once.
========================================================

Retries (example 13) make a *single* call survive. This is about *throughput*:
when you have 20 independent prompts, sending them one-by-one wastes almost all
your time waiting on the network. Each request is mostly idle time, so you can
have many in flight at once and finish in roughly the time of the slowest one.

The clean way is `AsyncOpenAI` + `asyncio`:
  - `AsyncOpenAI` is the same SDK, but its methods are coroutines you `await`.
  - `asyncio.gather(...)` runs many coroutines concurrently.
  - A `Semaphore` caps how many run at once, so you stay under your rate limit and
    don't open 10,000 sockets. This bounded-concurrency pattern is the one to keep.

This script summarizes several topics, first sequentially then concurrently, and
prints both wall-clock times so you can see the speedup.

Run it:

    secrun python examples/22_async_concurrency.py
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = AsyncOpenAI()

TOPICS = [
    "the water cycle", "compound interest", "how vaccines work",
    "the Doppler effect", "supply and demand", "photosynthesis",
]


async def summarize(topic: str, sem: asyncio.Semaphore) -> str:
    """One request, gated by a semaphore so only N run at a time."""
    async with sem:  # acquire a slot; block here if N are already in flight
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Explain {topic} in one short sentence."}],
            max_tokens=40,
        )
        return (resp.choices[0].message.content or "").strip()


async def run_concurrently(max_in_flight: int = 4) -> list[str]:
    sem = asyncio.Semaphore(max_in_flight)  # at most 4 requests at once
    # gather schedules them all; the semaphore throttles to `max_in_flight`.
    return await asyncio.gather(*(summarize(t, sem) for t in TOPICS))


async def run_sequentially() -> list[str]:
    # A Semaphore(1) is just "one at a time" — the slow baseline.
    sem = asyncio.Semaphore(1)
    return [await summarize(t, sem) for t in TOPICS]


async def main():
    print(f"Summarizing {len(TOPICS)} topics.\n")

    t0 = time.time()
    await run_sequentially()
    seq = time.time() - t0
    print(f"sequential (one at a time): {seq:.1f}s")

    t0 = time.time()
    results = await run_concurrently(max_in_flight=4)
    conc = time.time() - t0
    print(f"concurrent (4 at a time):   {conc:.1f}s   ->  ~{seq / conc:.1f}x faster\n")

    for topic, summary in zip(TOPICS, results):
        print(f"  • {topic}: {summary}")

    print("\nThe work was identical; only the waiting overlapped. Raise max_in_flight")
    print("for more speed — until you hit your account's rate limit, which is exactly")
    print("what the semaphore is there to keep you under.")


if __name__ == "__main__":
    asyncio.run(main())
