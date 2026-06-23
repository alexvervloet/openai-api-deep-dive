# OpenAI API — A Guided Deep Dive

A hands-on playground for learning the OpenAI API **from zero**. You'll build a
real CLI tool that answers questions about your code, and along the way you'll
understand every moving part: chat completions, roles, the sampling knobs
(temperature, top_p, max_tokens, stop), token counting, and cost.

This repo is meant to be *walked through*, not just read. Each section ends with
something to run. Do the running — that's where the learning is. And once a
section clicks, [EXERCISES.md](EXERCISES.md) has a quick predict-then-run prompt
for it: committing to an answer *before* you run is what makes it stick.

---

## 0. The one big idea

The OpenAI API is, at its core, astonishingly simple:

> **You send a list of messages. You get back a message.**

That's it. Everything else — the roles, the knobs, the token math — is detail on
top of that single request/response. Hold onto that idea and nothing below will
feel complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
#    ...then open .env and paste your key from
#    https://platform.openai.com/api-keys

# 4. Confirm everything is wired up correctly (makes no API call, costs nothing)
python check_setup.py
```

`check_setup.py` is your first stop if anything goes wrong: it checks your Python
version, your installed packages, and your key, and tells you exactly what to fix.
Green across the board means you're ready for Section 2.

> 💡 **You can learn a lot before spending a cent.** The token-counting and
> cost-estimation parts (Sections 5 & 6) run entirely offline. Skip ahead to
> them if you don't have a key yet.

---

## 2. Your first request

```bash
python examples/01_basic_chat.py
```

Open [examples/01_basic_chat.py](examples/01_basic_chat.py) and read it — it's
tiny. The shape of every call you'll ever make is right there:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "In one sentence, what is an API?"}],
)
print(response.choices[0].message.content)
```

Three things to internalize:

| Thing | What it is |
|-------|-----------|
| `model` | Which model answers. `gpt-4o-mini` is the cheap, fast default. |
| `messages` | A **list** of messages — your half of the conversation. |
| `response.choices[0].message.content` | The model's reply text. |
| `response.usage` | Exactly how many tokens you were billed for. |

---

## 3. Roles: system, user, assistant

A conversation is a transcript, and every line is tagged with a **role**:

- **`system`** — standing instructions. Persona, rules, tone. Set once, steers
  everything. *This is your most powerful lever.*
- **`user`** — what the human says.
- **`assistant`** — what the model said. You re-send past assistant messages to
  give the model **memory** — the API itself is stateless, so the conversation
  only exists in the list you send each time.

```bash
python examples/02_roles.py
```

**Experiment:** open [examples/02_roles.py](examples/02_roles.py), change the
system message to `"You are a grumpy pirate."`, and rerun. Same question, totally
different voice. That's the system role doing its job.

---

## 4. The knobs that shape a response

These four parameters control *how* the model answers. Each has its own runnable
example.

### temperature — how bold the word choices are
`0.0` = focused & repeatable · `0.7` = balanced (default) · `1.5+` = wild.
For code and facts, go **low**. For brainstorming, go high.
```bash
python examples/03_temperature.py
```

### max_tokens — a hard cap on the answer's length
Caps **output** tokens (not input). The model is cut off when the budget runs
out — possibly mid-sentence. Watch `finish_reason`: `"length"` means it was
truncated; `"stop"` means it finished naturally.
```bash
python examples/04_max_tokens.py
```

### top_p — how many options the model may consider
"Nucleus sampling." `0.1` = only the most obvious tokens; `1.0` = everything
(default). **Tune temperature OR top_p, not both** — they interact confusingly.
```bash
python examples/05_top_p.py
```

### stop — make generation halt at a marker
A string (up to 4) that ends generation the moment it would appear. The stop
text itself isn't included. Great for cutting lists or hitting a delimiter.
```bash
python examples/06_stop_sequences.py
```

**Quick reference:**

| Knob | Range | Raise it to... | Default |
|------|-------|----------------|---------|
| `temperature` | 0.0–2.0 | get more variety/creativity | 1.0 |
| `top_p` | 0.0–1.0 | widen the pool of candidate words | 1.0 |
| `max_tokens` | ≥1 | allow a longer answer | model max |
| `stop` | up to 4 strings | end at a specific marker | none |

---

## 5. Tokens — what you actually pay for

Models don't read characters or words. They read **tokens** — chunks of text,
often word-fragments. Rough rule: 1 token ≈ 4 English characters ≈ ¾ of a word.
But "rough" isn't good enough for budgeting, so we count exactly with
**tiktoken**, locally, with no API call.

```bash
python utils/tokens.py          # see a sentence broken into tokens
```

Why counting matters:
1. **Cost** — you're billed per token (next section).
2. **Limits** — every model has a max context window (input + output). Overflow
   it and the request fails.
3. **Intuition** — watching the count change as you edit a prompt teaches you
   how the model "sees" your text.

See [utils/tokens.py](utils/tokens.py) for `count_tokens()` (a raw string) and
`count_message_tokens()` (a full chat list, including the small per-message
overhead the API adds).

---

## 6. Cost estimation

OpenAI charges **separately for input and output tokens**, and output is
typically several times more expensive. [utils/pricing.py](utils/pricing.py) holds a small
price table and an `estimate_cost()` helper.

```bash
python examples/07_token_counting.py   # tokens -> dollars, across models, offline
```

Sample output shows the same request costing wildly different amounts per model
— which is why **choosing the right model is part of prompt engineering.**

> ⚠️ Prices change. The table in `utils/pricing.py` is a snapshot — always confirm at
> <https://platform.openai.com/docs/pricing>.

---

## 7. The first capstone: `ask.py`

Everything above comes together in one tool: ask a question about a code file,
see the token count and estimated cost *before* you spend, get the answer, and
see the *actual* usage and cost after.

```bash
# See the cost first — no API call, no key needed:
python hands_on/ask.py snippets/buggy.py "Is there a bug here?" --dry-run

# For real (needs your key in .env):
python hands_on/ask.py snippets/buggy.py "Is there a bug here?"

# Now turn the knobs you just learned:
python hands_on/ask.py snippets/buggy.py "Rewrite this cleanly" --temperature 0
python hands_on/ask.py snippets/buggy.py "List the issues" --max-tokens 200 --stop "4."
python hands_on/ask.py snippets/buggy.py "Explain this" --model gpt-4o
```

Run `python hands_on/ask.py --help` to see every knob explained inline. Read the source
in [hands_on/ask.py](hands_on/ask.py) — it's commented as a tutorial, especially `build_messages()`
(how the request is assembled) and the usage/cost reporting at the end.

**Suggested exercise:** point `hands_on/ask.py` at your *own* code, try the same question
at `--temperature 0` vs `--temperature 1.2`, and watch both the answers and the
cost change.

---

## 8. Beyond the basics

With the core down, here are the most useful next capabilities — each is a
runnable example in the same numbered style. Every one is still just a variation
on "send messages, get a message."

### Streaming — get the answer as it's typed
With `stream=True` the response arrives in small chunks as it's generated, so the
user sees text appear immediately instead of waiting for the whole thing.
```bash
python examples/08_streaming.py
```

### Structured outputs — make the model return real JSON
Force the reply to be valid JSON, or even match an exact schema you define
(`response_format` with `"strict": True`). The end of fragile "please reply in
JSON" prompting.
```bash
python examples/09_structured_outputs.py
```

### Function / tool calling — let the model use your code
You describe functions; the model decides when to call one and with what
arguments; *you* run it and feed the result back. This is how a model gets to
actually *do* things (query a DB, hit an API).
```bash
python examples/10_function_calling.py
```

### Embeddings — turn text into vectors for search & similarity
A different endpoint (`client.embeddings.create`) that converts text into numbers
capturing meaning. The foundation of semantic search and retrieval (RAG). The
example ranks sentences by similarity to a query — including ones that share *no
words* with it.
```bash
python examples/11_embeddings.py
```

### Multi-turn conversations — the API has no memory
Each request is stateless: the model remembers nothing. A chatbot that "remembers"
is just *you* re-sending the whole `messages` list every turn, appending each new
user and assistant message. The example is a tiny REPL that grows that list.
```bash
python examples/12_conversation.py
```

### Error handling & retries — surviving the real world
The network blips, you hit a rate limit, a model name has a typo. The SDK already
retries transient failures (429/5xx/connection) with backoff; your job is to tune
`timeout`/`max_retries` and catch the *typed* exceptions so "fix your request"
errors are handled differently from "try again later" ones.
```bash
python examples/13_error_handling.py
```

### Pydantic validation — typed, validated responses
Instead of a hand-written JSON Schema + `json.loads` into an untyped dict, define
your shape as a **Pydantic model** and pass it as `response_format`. The SDK sends
the schema, constrains the model, and hands back a *validated instance* — typed
attributes, enforced constraints, editor autocomplete.
```bash
python examples/14_pydantic_validation.py
```

### Formatting output — Markdown, tables & code blocks
Models answer in Markdown; dumped raw to a terminal it's a mess of literal
`**asterisks**`. The `rich` library renders Markdown, syntax-highlighted code, and
real tables in the terminal — the difference between output you skim and output
you squint at.
```bash
python examples/15_rich_output.py
```

### Server-Sent Events (SSE) — the protocol under streaming
Every streaming AI response travels over SSE: a plain HTTP response that stays
open and drips `data: <json>` lines until the server is done. The SDK hides the
parsing, but understanding the wire format is essential when you build a backend
that forwards tokens to a browser. This example shows raw events, per-token
timing, and partial response accumulation.
```bash
python examples/16_sse.py
```

---

## 9. The second capstone: `extract.py`

Where `ask.py` returns *prose*, `extract.py` returns *data*. Point it at messy
free-form text and it pulls out a clean, typed, **validated** structure — then
shows it as a Markdown summary and a real table. It's where examples 14
(Pydantic) and 15 (rich) earn their keep on a realistic task.

```bash
# See tokens + cost first — no API call:
python hands_on/extract.py snippets/meeting_notes.txt --dry-run

# Extract action items (owner, due date, inferred priority) into a table:
python hands_on/extract.py snippets/meeting_notes.txt

# Want the raw validated JSON instead? (e.g. to pipe into another tool)
python hands_on/extract.py snippets/meeting_notes.txt --json
```

Read the source in [hands_on/extract.py](hands_on/extract.py): the `Extraction` / `ActionItem`
Pydantic models *are* the schema the model must follow, and `render()` is the
rich table. **Suggested exercise:** point it at your own meeting notes or an
email, or change the models to extract something else entirely (contacts,
invoice line items) — the prompt barely changes.

---

## 10. The third capstone: `streaming_server.py`

Where `ask.py` and `extract.py` are CLI tools, `streaming_server.py` is a web
service. It's a FastAPI backend that streams AI responses to a browser over SSE,
showing three production concerns: token-by-token forwarding, client disconnect
detection, and error recovery with retries.

```bash
# Start the server (auto-reloads on file saves):
uvicorn hands_on.streaming_server:app --reload

# Then open: http://localhost:8000
```

Open the browser's **Network tab** and click the `/stream` request to see the raw
`text/event-stream` response. Close the tab mid-stream to watch the server log
"client disconnected" and stop the AI call. Read the source in
[hands_on/streaming_server.py](hands_on/streaming_server.py) — the three-phase
generator (`_stream_tokens`) is the core pattern every production streaming
endpoint follows.

**Suggested exercise:** point the server at `gpt-4o` and ask it something long,
then close the browser tab mid-response. Notice in the server logs that generation
stops immediately — no wasted tokens.

---

## Where to go next

You've now covered the essentials, the common extensions, and three capstone
projects. Further on:

- **Retrieval-augmented generation (RAG)** — combine embeddings (Section 8) with
  chat to answer questions over your own documents.
- **The context window** — what happens as conversations get long, and smarter
  ways to manage history than the simple trim in example 12 (summarizing old
  turns, sliding windows).
- **Vision & audio** — passing images to multimodal models, speech-to-text.
- **Streaming + tools together** — the pattern most production assistants use.

Each of these slots neatly on top of the "send messages, get a message" idea you
started with.

---

## Troubleshooting

Hit a snag? Run `python check_setup.py` first — it catches most problems. The
rest, by the error you see:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `ModuleNotFoundError: No module named 'openai'` | Dependencies aren't installed (or your venv isn't active). Run `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `Set OPENAI_API_KEY ...` on every script | No key found. `cp .env.example .env`, paste your real key, save. (The offline token/cost parts in Sections 5–6 still run without a key.) |
| `AuthenticationError` / 401 | The key is present but wrong — expired, revoked, or a typo. Make a fresh one at the [API keys page](https://platform.openai.com/api-keys). |
| `RateLimitError` / 429 | Too many requests, or you're out of credit. Wait a moment, or check your billing/usage in the dashboard. |
| `NotFoundError` / 404 about the model | A model name was mistyped or your account can't access it. The examples use widely-available IDs; if you changed one, check it against the [models list](https://platform.openai.com/docs/models). |
| `SyntaxError` or odd type errors on startup | You're likely on Python 3.9 or older. This repo needs 3.10+ — `check_setup.py` will confirm your version. |
| It "hangs" with no output | Some examples stream, others wait for the full reply before printing. Give it a few seconds; for streaming examples you'll see text appear word by word. |

Still stuck? Every example is small and self-contained — open the file, read the
docstring at the top, and run it directly. The error message almost always points
at the line.

---

## File map

```
check_setup.py              ← run first: verifies Python, packages, and your key
EXERCISES.md                ← active-recall prompts, one per README section
hands_on/
  ask.py                    ← capstone CLI: ask a question about a code file
  extract.py                ← capstone CLI: extract validated data from free text
  streaming_server.py       ← capstone server: stream AI responses over SSE
  static/index.html         ← browser UI for the streaming server
utils/
  tokens.py                 ← tiktoken-based token counting
  pricing.py                ← price table + cost estimation
snippets/buggy.py           ← a sample file to ask questions about
snippets/meeting_notes.txt  ← sample free-form text for extract.py
examples/
  01_basic_chat.py          ← the minimal request
  02_roles.py               ← system / user / assistant
  03_temperature.py         ← randomness
  04_max_tokens.py          ← length cap + finish_reason
  05_top_p.py               ← nucleus sampling
  06_stop_sequences.py      ← halting generation
  07_token_counting.py      ← tokens & cost, fully offline
  08_streaming.py           ← stream the answer as it's generated
  09_structured_outputs.py  ← guaranteed JSON / schema-conformant output
  10_function_calling.py    ← let the model call your functions
  11_embeddings.py          ← vectors & semantic similarity
  12_conversation.py        ← multi-turn chat & the stateless API
  13_error_handling.py      ← timeouts, retries & typed exceptions
  14_pydantic_validation.py ← typed, validated responses via Pydantic
  15_rich_output.py         ← Markdown, tables & code blocks in the terminal
  16_sse.py                 ← SSE protocol: raw events, timing, partial accumulation
```

---

### Footnote — quieting Pylance/type-checker noise

Two patterns trip the type checker repeatedly with the OpenAI SDK. Pre-empt them
and new files stay clean:

1. **Assigning `messages` / `tools` to a variable** (rather than passing the
   literal straight into `create()`) makes Pylance infer a too-narrow type like
   `list[dict[str, str]]`. Annotate with the SDK's own param types — you also get
   key autocomplete:
   ```python
   from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
   messages: list[ChatCompletionMessageParam] = [...]
   tools: list[ChatCompletionToolParam] = [...]
   ```
2. **`.content` is typed `str | None`** (it's `None` when the model returns only
   tool calls). `print()` and f-strings accept it, but `json.loads()` and `+`
   concatenation don't — guard with `... or ""` / `... or "{}"`.

The repo's [.vscode/settings.json](.vscode/settings.json) also sets
`python.analysis.typeCheckingMode` to `basic`, which keeps the useful checks
(undefined names, bad attrs/args) while dropping the strict dict-vs-TypedDict
complaints.
