#!/usr/bin/env python3
"""
streaming_server.py — FastAPI streaming server capstone.
=========================================================

This is the capstone project for the SSE section: a production-style FastAPI
server that streams OpenAI responses to a browser using Server-Sent Events.
It shows the three challenges unique to streaming web services:

  1. Token-by-token forwarding — each AI token is wrapped in a JSON SSE event
     and pushed to the browser as it arrives, not buffered until the end.

  2. Disconnect detection — if the browser closes the tab mid-stream, we detect
     it and abort the AI request immediately (stops burning tokens).

  3. Error recovery — transient API errors (rate limits, connection blips)
     trigger automatic retries with exponential backoff before the stream
     starts. Mid-stream errors yield a clean error event so the browser can
     show a message rather than hanging indefinitely.

Run the server
--------------
    # One-time install (if not already in requirements.txt):
    pip install fastapi "uvicorn[standard]"

    # Start (auto-reloads on file saves):
    uvicorn hands_on.streaming_server:app --reload

Then open http://localhost:8000 in a browser. The HTML client in
hands_on/static/index.html is served automatically.

SSE event format
----------------
Each event the server sends is a JSON object on a `data:` line:

    data: {"type": "token",   "text": "Hello"}

    data: {"type": "token",   "text": " world"}

    data: {"type": "done",    "tokens": 42, "elapsed": 1.23}

    data: {"type": "error",   "message": "...", "partial": "...so far..."}

Architecture
------------
  POST /stream  body: {"prompt": "..."}
    → StreamingResponse(text/event-stream)
    → retries the API call up to 3× on transient errors
    → yields token events token-by-token
    → detects client disconnect and aborts early
    → yields done or error event at the end

  GET /
    → serves hands_on/static/index.html

See examples/16_sse.py to understand the underlying SSE protocol.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add repo root to path so `utils.*` is importable from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in .env and try again.")

# Async client — essential for FastAPI so the event loop isn't blocked while
# waiting for the API. Each request gets its own concurrent slot.
async_client = AsyncOpenAI()

app = FastAPI(title="Streaming AI Server")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class StreamRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------

def _token_event(text: str) -> str:
    return f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

def _done_event(tokens: int, elapsed: float) -> str:
    return f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'elapsed': round(elapsed, 2)})}\n\n"

def _error_event(message: str, partial: str = "") -> str:
    payload: dict = {"type": "error", "message": message}
    if partial:
        payload["partial"] = partial
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------

async def _stream_tokens(request: Request, body: StreamRequest):
    """
    Async generator that calls the OpenAI API and yields SSE events.

    Three concerns handled here:

    1. Retry before streaming starts — transient errors (rate limit, network)
       are retried with exponential backoff. Once tokens begin flowing we
       don't retry (partial output would confuse the client).

    2. Disconnect detection — `request.is_disconnected()` is polled each
       iteration. On disconnect we `return` early, which causes the
       StreamingResponse to close its socket and abort the API call.

    3. Partial response — we accumulate tokens into `partial` so that if an
       error occurs mid-stream, the error event includes what was received.
    """
    partial: list[str] = []
    start = time.perf_counter()

    # --- Phase 1: open the stream with retries ---
    stream = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            stream = await async_client.chat.completions.create(
                model=body.model,
                messages=[{"role": "user", "content": body.prompt}],
                max_tokens=body.max_tokens,
                stream=True,
            )
            break
        except (openai.RateLimitError, openai.APIConnectionError) as exc:
            last_exc = exc
            if attempt < 2:
                # Exponential backoff: 1s, 2s before giving up.
                await asyncio.sleep(1.0 * (2 ** attempt))
        except openai.AuthenticationError as exc:
            yield _error_event(f"Authentication failed: {exc}")
            return
        except openai.BadRequestError as exc:
            yield _error_event(f"Bad request: {exc}")
            return

    if stream is None:
        yield _error_event(f"Could not reach the API after 3 attempts: {last_exc}")
        return

    # --- Phase 2: stream tokens to the client ---
    try:
        async for chunk in stream:
            # Check for client disconnect before each yield.
            if await request.is_disconnected():
                return  # generator closes → StreamingResponse cleans up

            if chunk.choices:
                piece = chunk.choices[0].delta.content
                if piece:
                    partial.append(piece)
                    yield _token_event(piece)

    except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
        # Mid-stream transient error — report what we had so far.
        yield _error_event(str(exc), partial="".join(partial))
        return
    except openai.APIError as exc:
        yield _error_event(str(exc), partial="".join(partial))
        return

    # --- Phase 3: final stats event ---
    elapsed = time.perf_counter() - start
    yield _done_event(tokens=len(partial), elapsed=elapsed)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_ui():
    """Serve the browser UI from hands_on/static/index.html."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return {"error": "index.html not found in hands_on/static/"}
    return FileResponse(html_path)


@app.post("/stream")
async def stream_endpoint(request: Request, body: StreamRequest):
    """
    Stream an AI response as SSE.

    The client should read the response body as a stream (fetch + ReadableStream)
    and parse each `data: <json>\\n\\n` event. See hands_on/static/index.html
    for a working browser example.
    """
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # tells Nginx not to buffer this response
    }
    return StreamingResponse(
        _stream_tokens(request, body),
        media_type="text/event-stream",
        headers=headers,
    )
