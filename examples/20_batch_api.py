"""
Example 20: the Batch API: half price, for work that isn't urgent.

Every example so far made *synchronous* calls: you ask, you wait, you get an
answer. But a lot of real LLM work isn't interactive: classify 10,000 reviews,
summarize a backlog of tickets, generate embeddings for a corpus. For that, the
**Batch API** is the right tool: you upload a file of requests, OpenAI processes
them within 24 hours, and you pay **50% less** per token. No rate-limit juggling,
no `for` loop hammering the endpoint.

The lifecycle (it mirrors a job queue):

  1. Build a JSONL file, one request per line, each with a `custom_id`.
  2. Upload it          (files.create, purpose="batch").
  3. Create the batch   (batches.create), which starts processing.
  4. Poll until done    (batches.retrieve); status goes validating -> in_progress
                        -> completed.
  5. Download results   (files.content), a JSONL keyed by your custom_ids.

This script builds a tiny 3-request batch and submits it. Because a batch can take
minutes to (up to) 24 hours, it does NOT block waiting. It prints the batch id and
shows you how to check on it and fetch results. Re-run with the id to poll.

Run it:

    secrun python examples/20_batch_api.py            # create a batch
    secrun python examples/20_batch_api.py <batch_id> # check status / fetch results
"""

import io
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()

# Three independent jobs. In real life this might be thousands of lines.
PROMPTS = {
    "review-1": "Classify sentiment as positive/negative/neutral: 'Battery lasts forever, love it.'",
    "review-2": "Classify sentiment as positive/negative/neutral: 'Arrived broken and support ignored me.'",
    "review-3": "Classify sentiment as positive/negative/neutral: 'It's a phone. It works.'",
}


def create_batch() -> str:
    # --- Step 1: build the JSONL. Each line is a full /v1/chat/completions request,
    # tagged with a custom_id so you can match answers back to inputs later. ---
    lines = []
    for custom_id, prompt in PROMPTS.items():
        lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
            },
        }))
    jsonl = "\n".join(lines).encode("utf-8")

    # --- Step 2: upload the file (purpose must be "batch"). ---
    upload = client.files.create(file=io.BytesIO(jsonl), purpose="batch")
    print(f"[uploaded input file: {upload.id}]")

    # --- Step 3: create the batch. completion_window is 24h (the only option). ---
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[created batch: {batch.id}  status={batch.status}]")
    print("\nThe batch is now processing (50% cheaper than live calls). Check on it with:")
    print(f"    secrun python examples/20_batch_api.py {batch.id}")
    return batch.id


def check_batch(batch_id: str) -> None:
    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    assert counts is not None
    print(f"status: {batch.status}   "
          f"({counts.completed}/{counts.total} done, {counts.failed} failed)")

    if batch.status != "completed":
        print("Not finished yet; batches run within 24h. Re-run this to check again.")
        return

    # --- Step 5: download and print results, matched by custom_id. ---
    assert batch.output_file_id is not None
    out_text = client.files.content(batch.output_file_id).text
    print("\nResults:")
    for line in out_text.splitlines():
        row = json.loads(line)
        answer = row["response"]["body"]["choices"][0]["message"]["content"].strip()
        print(f"  {row['custom_id']}: {answer}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_batch(sys.argv[1])
    else:
        create_batch()
