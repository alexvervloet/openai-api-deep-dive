"""
Example 13: errors, timeouts, and retries (surviving the real world).

Every example so far assumed the happy path: the request succeeds. In
production it often won't. The network blips, you hit a rate limit, a model
name has a typo, the service is briefly overloaded. Code that ignores this
crashes on the first hiccup.

The good news: the OpenAI SDK already does most of the work for you.

  >> The client AUTOMATICALLY RETRIES transient failures (429 rate limits,
  >> 5xx server errors, connection errors) with exponential backoff.
  >> The default is 2 retries. You usually don't need a retry loop at all.

What you DO need is two things:

  1. Configure the client's `timeout` and `max_retries` for your use case.
  2. Catch the SDK's typed exceptions so you can react differently to a
     "fix your request" error (bad key, bad model, don't retry) versus a
     "try again later" error (rate limit, overload, which the SDK already did).

The exceptions form a hierarchy. Catch the SPECIFIC ones you handle specially,
then a broad `APIError` as a backstop: most specific first, or the broad one
shadows the rest.

Run it:

    secrun python examples/13_error_handling.py

It deliberately requests a nonexistent model to show a NotFoundError being
caught, then makes a normal call with tuned timeout/retry settings.
"""

import os
import sys

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

# Per-client config. `timeout` is in seconds; `max_retries` overrides the
# default of 2. (You can also override per-call with
# `client.with_options(timeout=5).chat.completions.create(...)`.)
client = OpenAI(timeout=20.0, max_retries=3)


def ask(model: str, question: str) -> str | None:
    """Make one request, translating each failure into a clear message.

    Order matters: list the specific subclasses before the broad `APIError`,
    or the broad clause catches everything and the specific ones never run.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content

    except openai.AuthenticationError:
        # 401: bad/missing key. Not retryable; fix the credentials.
        print("Auth failed: check OPENAI_API_KEY.")
    except openai.NotFoundError:
        # 404: usually a typo'd or unavailable model. Not retryable.
        print(f"Model not found: {model!r}. Check the model name.")
    except openai.BadRequestError as e:
        # 400: malformed request (bad params, too many tokens). Not retryable.
        print(f"Bad request: {e}")
    except openai.RateLimitError:
        # 429: the SDK already retried with backoff and still failed.
        # Back off further, queue the work, or slow your request rate.
        print("Rate limited even after retries. Slow down or try later.")
    except openai.APITimeoutError:
        # The request exceeded `timeout` (and exhausted retries).
        print("Request timed out. Consider raising the timeout or streaming.")
    except openai.APIConnectionError:
        # Network failure before a response was received.
        print("Network error: could not reach the API. Check your connection.")
    except openai.APIStatusError as e:
        # Any other non-2xx (e.g. 500/503). These ARE retried by the SDK;
        # reaching here means the retries were used up.
        print(f"API error {e.status_code} after retries: {e.message}")
    except openai.APIError as e:
        # Backstop for anything not caught above.
        print(f"Unexpected API error: {e}")

    return None


# 1. A request that fails predictably: caught and reported, no crash.
print("--- requesting a model that doesn't exist ---")
ask("gpt-4o-mini-does-not-exist", "Hello?")

# 2. A normal request that succeeds, using the tuned client.
print("\n--- a normal request ---")
answer = ask("gpt-5.4-nano", "In one sentence, why is retry logic important?")
if answer:
    print(answer)
