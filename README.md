# OpenAI API: A Guided Deep Dive

A hands-on playground for learning the OpenAI API **from zero**. You'll build a
real CLI tool that answers questions about your code, and along the way you'll
understand every moving part: chat completions, roles, the sampling knobs
(temperature, top_p, max_completion_tokens, stop), token counting, and cost.

This repo is meant to be *walked through*, not just read. Each section ends with
something to run. Do the running; that's where the learning is. And once a
section clicks, [EXERCISES.md](EXERCISES.md) has a quick predict-then-run prompt
for it: committing to an answer *before* you run is what makes it stick.

---

## 0. The one big idea

The OpenAI API is, at its core, astonishingly simple:

> **You send a list of messages. You get back a message.**

That's it. Everything else (the roles, the knobs, the token math) is detail on
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

# 3. Set up your API key (it does NOT go in .env)
cp .env.example .env               # optional; holds no secrets
#    Store your key in your OS keychain and run scripts with `secrun`: 2-minute
#    setup in ../SECRETS.md. Get a key: https://platform.openai.com/api-keys

# 4. Confirm everything is wired up correctly (makes no API call, costs nothing)
secrun python check_setup.py
```

`check_setup.py` is your first stop if anything goes wrong: it checks your Python
version, your installed packages, and your key, and tells you exactly what to fix.
Green across the board means you're ready for Section 2.

> **You can learn a lot before spending a cent.** The token-counting and
> cost-estimation parts (Sections 5 & 6) run entirely offline. Skip ahead to
> them if you don't have a key yet.

---

## 2. Your first request

```bash
secrun python examples/01_basic_chat.py
```

Open [examples/01_basic_chat.py](examples/01_basic_chat.py) and read it; it's
tiny. The shape of every call you'll ever make is right there:

```python
response = client.chat.completions.create(
    model="gpt-5.4-nano",
    messages=[{"role": "user", "content": "In one sentence, what is an API?"}],
)
print(response.choices[0].message.content)
```

Three things to internalize:

| Thing | What it is |
|-------|-----------|
| `model` | Which model answers. `gpt-5.4-nano` is the cheap, fast default. |
| `messages` | A **list** of messages: your half of the conversation. |
| `response.choices[0].message.content` | The model's reply text. |
| `response.usage` | Exactly how many tokens you were billed for. |

---

## 3. Roles: system, user, assistant

A conversation is a transcript, and every line is tagged with a **role**:

- **`system`**: standing instructions. Persona, rules, tone. Set once, steers
  everything. *This is your most powerful lever.*
- **`user`**: what the human says.
- **`assistant`**: what the model said. You re-send past assistant messages to
  give the model **memory**. The API itself is stateless, so the conversation
  only exists in the list you send each time.

```bash
secrun python examples/02_roles.py
```

**Experiment:** open [examples/02_roles.py](examples/02_roles.py), change the
system message to `"You are a grumpy pirate."`, and rerun. Same question, totally
different voice. That's the system role doing its job.

---

## 4. The knobs that shape a response

These four parameters control *how* the model answers. Each has its own runnable
example.

### temperature: how bold the word choices are
`0.0` = focused & repeatable · `0.7` = balanced (default) · `1.5+` = wild.
For code and facts, go **low**. For brainstorming, go high.
```bash
secrun python examples/03_temperature.py
```

### max_completion_tokens: a hard cap on the answer's length
Caps **output** tokens (not input). The model is cut off when the budget runs
out: possibly mid-sentence. Watch `finish_reason`: `"length"` means it was
truncated; `"stop"` means it finished naturally.
```bash
secrun python examples/04_max_tokens.py
```

### top_p: how many options the model may consider
"Nucleus sampling." `0.1` = only the most obvious tokens; `1.0` = everything
(default). **Tune temperature OR top_p, not both**; they interact confusingly.
```bash
secrun python examples/05_top_p.py
```

### stop: make generation halt at a marker
A string (up to 4) that ends generation the moment it would appear. The stop
text itself isn't included. Great for cutting lists or hitting a delimiter.
```bash
secrun python examples/06_stop_sequences.py
```

**Quick reference:**

| Knob | Range | Raise it to... | Default |
|------|-------|----------------|---------|
| `temperature` | 0.0–2.0 | get more variety/creativity | 1.0 |
| `top_p` | 0.0–1.0 | widen the pool of candidate words | 1.0 |
| `max_completion_tokens` | ≥1 | allow a longer answer | model max |
| `stop` | up to 4 strings | end at a specific marker | none |

---

## 5. Tokens: what you actually pay for

Models don't read characters or words. They read **tokens**: chunks of text,
often word-fragments. Rough rule: 1 token ≈ 4 English characters ≈ ¾ of a word.
But "rough" isn't good enough for budgeting, so we count exactly with
**tiktoken**, locally, with no API call.

```bash
python utils/tokens.py          # see a sentence broken into tokens
```

Why counting matters:
1. **Cost**: you're billed per token (next section).
2. **Limits**: every model has a max context window (input + output). Overflow
   it and the request fails.
3. **Intuition**: watching the count change as you edit a prompt teaches you
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

Sample output shows the same request costing wildly different amounts per model,
which is why **choosing the right model is part of prompt engineering.**

> Prices change. The table in `utils/pricing.py` is a snapshot, so always confirm at
> <https://platform.openai.com/docs/pricing>.

---

## 7. The first capstone: `ask.py`

Everything above comes together in one tool: ask a question about a code file,
see the token count and estimated cost *before* you spend, get the answer, and
see the *actual* usage and cost after.

```bash
# See the cost first: no API call, no key needed:
secrun python hands_on/ask.py snippets/buggy.py "Is there a bug here?" --dry-run

# For real (needs your key; run under secrun):
secrun python hands_on/ask.py snippets/buggy.py "Is there a bug here?"

# Now turn the knobs you just learned:
secrun python hands_on/ask.py snippets/buggy.py "Rewrite this cleanly" --temperature 0
secrun python hands_on/ask.py snippets/buggy.py "List the issues" --max-tokens 200 --stop "4."
secrun python hands_on/ask.py snippets/buggy.py "Explain this" --model gpt-4o
```

Run `secrun python hands_on/ask.py --help` to see every knob explained inline. Read the source
in [hands_on/ask.py](hands_on/ask.py); it's commented as a tutorial, especially `build_messages()`
(how the request is assembled) and the usage/cost reporting at the end.

**Suggested exercise:** point `hands_on/ask.py` at your *own* code, try the same question
at `--temperature 0` vs `--temperature 1.2`, and watch both the answers and the
cost change.

---

## 8. Beyond the basics

With the core down, here are the most useful next capabilities. Each is a
runnable example in the same numbered style. Every one is still just a variation
on "send messages, get a message."

### Streaming: get the answer as it's typed
With `stream=True` the response arrives in small chunks as it's generated, so the
user sees text appear immediately instead of waiting for the whole thing.
```bash
secrun python examples/08_streaming.py
```

### Structured outputs: make the model return real JSON
Force the reply to be valid JSON, or even match an exact schema you define
(`response_format` with `"strict": True`). The end of fragile "please reply in
JSON" prompting.
```bash
secrun python examples/09_structured_outputs.py
```

### Function / tool calling: let the model use your code
You describe functions; the model decides when to call one and with what
arguments; *you* run it and feed the result back. This is how a model gets to
actually *do* things (query a DB, hit an API).
```bash
secrun python examples/10_function_calling.py
```

### Embeddings: turn text into vectors for search & similarity
A different endpoint (`client.embeddings.create`) that converts text into numbers
capturing meaning. The foundation of semantic search and retrieval (RAG). The
example ranks sentences by similarity to a query, including ones that share *no
words* with it.
```bash
secrun python examples/11_embeddings.py
```

### Multi-turn conversations: the API has no memory
Each request is stateless: the model remembers nothing. A chatbot that "remembers"
is just *you* re-sending the whole `messages` list every turn, appending each new
user and assistant message. The example is a tiny REPL that grows that list.
```bash
secrun python examples/12_conversation.py
```

### Error handling & retries: surviving the real world
The network blips, you hit a rate limit, a model name has a typo. The SDK already
retries transient failures (429/5xx/connection) with backoff; your job is to tune
`timeout`/`max_retries` and catch the *typed* exceptions so "fix your request"
errors are handled differently from "try again later" ones.
```bash
secrun python examples/13_error_handling.py
```

### Pydantic validation: typed, validated responses
Instead of a hand-written JSON Schema + `json.loads` into an untyped dict, define
your shape as a **Pydantic model** and pass it as `response_format`. The SDK sends
the schema, constrains the model, and hands back a *validated instance*: typed
attributes, enforced constraints, editor autocomplete.
```bash
secrun python examples/14_pydantic_validation.py
```

### Formatting output: Markdown, tables & code blocks
Models answer in Markdown; dumped raw to a terminal it's a mess of literal
`**asterisks**`. The `rich` library renders Markdown, syntax-highlighted code, and
real tables in the terminal, the difference between output you skim and output
you squint at.
```bash
secrun python examples/15_rich_output.py
```

### Server-Sent Events (SSE): the protocol under streaming
Every streaming AI response travels over SSE: a plain HTTP response that stays
open and drips `data: <json>` lines until the server is done. The SDK hides the
parsing, but understanding the wire format is essential when you build a backend
that forwards tokens to a browser. This example shows raw events, per-token
timing, and partial response accumulation.
```bash
secrun python examples/16_sse.py
```

### Local models: the same client, a different `base_url`
Nothing here is tied to OpenAI's servers. Local runtimes (**Ollama**,
**llama.cpp**) expose an *OpenAI-compatible* endpoint, so the same `openai` SDK
talks to a model on your own machine. You change `base_url` and pass any
non-empty `api_key`, and everything else (roles, knobs, streaming, usage) works
unchanged. Privacy, no per-token bill, and offline use, in exchange for running
the server yourself. The example degrades gracefully if no local server is up.
```bash
secrun python examples/17_local_serving.py    # needs a local runtime; prints how to start one
```

### Vision: send an image, not just text
Multimodal models accept images in the same message: the user `content` becomes a
*list of parts* (text + `image_url`), where the image is either a URL or a local
file inlined as a base64 `data:` URI. Images are billed as tokens, scaled by pixel
size. The example reads a public sample image (or your own local file).
```bash
secrun python examples/18_vision.py            # or: secrun python examples/18_vision.py my_image.png
```

### Reasoning models: think first, answer second
The o-series (and GPT-5 reasoning tiers) generate hidden **reasoning tokens** before
answering, and far better on math/logic/coding. You drop `temperature` and steer with
`reasoning_effort` instead; `usage` reports the hidden thinking you still pay for.
```bash
secrun python examples/19_reasoning.py
```

### The Batch API: half price for non-urgent work
For work that isn't interactive (classify 10k reviews, summarize a backlog), upload
a JSONL of requests and get results within 24h at **50% off**. The example builds a
tiny batch, submits it, and shows how to poll for and fetch results.
```bash
secrun python examples/20_batch_api.py
```

### Prompt caching: don't re-pay for a repeated prefix
On OpenAI this is **automatic**: a prompt's long, identical prefix is cached and
re-billed at a discount on later calls. The one rule is structural: put the
*constant* part (big system prompt, tool catalog, a document) first, the *variable*
question last. The example shows `cached_tokens` kicking in.
```bash
secrun python examples/21_prompt_caching.py
```

### Async & concurrency: many requests at once
A single call is mostly idle waiting on the network, so independent prompts should
run concurrently. `AsyncOpenAI` + `asyncio.gather` + a `Semaphore` (bounded
concurrency) finishes a batch in roughly the time of the slowest call, while staying
under your rate limit. The example times sequential vs. concurrent.
```bash
secrun python examples/22_async_concurrency.py
```

### Moderation: a free safety filter
A dedicated, **free** classifier that flags hateful/violent/sexual/self-harm content
with per-category scores. The pattern: moderate user input on the way in and model
output on the way out, and refuse/redact when flagged.
```bash
secrun python examples/23_moderation.py
```

### Logprobs: how confident was the model?
With `logprobs=True` / `top_logprobs=k` the API returns the probability of each
chosen token and the alternatives it weighed. Turn that into a 0-1 confidence, for
calibrated classification, flagging shaky answers, or debugging.
```bash
secrun python examples/24_logprobs.py
```

### Seed & reproducibility: pinning down a random model
A fixed `seed` makes the same inputs reproduce the same output, for tests, caching,
and reproducible evals, even with real randomness in play. The example runs at
`temperature=0.9` specifically so you can see the seed do the work: the same seed
twice gives identical output, no seed twice gives different output. (At
`temperature=0` the model is already deterministic, so the seed wouldn't visibly be
doing anything; in production you'd combine both for the strongest guarantee.)
Best-effort either way, not a guarantee: watch `system_fingerprint`, which signals a
backend change that can break determinism.
```bash
secrun python examples/25_seed_determinism.py
```

### The Responses API: OpenAI's other endpoint
Everything above uses `chat.completions.create`. There is a second endpoint,
`responses.create`, and you will meet it in OpenAI's own docs. It buys you two
things Chat Completions genuinely cannot do: **server-side conversation state**
(pass `previous_response_id` instead of re-sending the transcript) and **hosted
tools** like `web_search` that run on OpenAI's side with no tool loop of yours.
It is also what replaced the Assistants API, which shuts down **2026-08-26**.

The catch is portability. `/v1/chat/completions` is the industry's common
dialect: Ollama, LM Studio, vLLM, LiteLLM and most hosts implement it, which is
exactly why [example 17](examples/17_local_serving.py) can point the same client
at a local model by changing `base_url`. The Responses API is OpenAI's own
shape, so building on it gives that up. Reach for it when you want the hosted
tools or the state badly enough to accept the lock-in.
```bash
secrun python examples/26_responses_api.py
```

---

## 9. The second capstone: `extract.py`

Where `ask.py` returns *prose*, `extract.py` returns *data*. Point it at messy
free-form text and it pulls out a clean, typed, **validated** structure, then
shows it as a Markdown summary and a real table. It's where examples 14
(Pydantic) and 15 (rich) earn their keep on a realistic task.

```bash
# See tokens + cost first: no API call:
secrun python hands_on/extract.py snippets/meeting_notes.txt --dry-run

# Extract action items (owner, due date, inferred priority) into a table:
secrun python hands_on/extract.py snippets/meeting_notes.txt

# Want the raw validated JSON instead? (e.g. to pipe into another tool)
secrun python hands_on/extract.py snippets/meeting_notes.txt --json
```

Read the source in [hands_on/extract.py](hands_on/extract.py): the `Extraction` / `ActionItem`
Pydantic models *are* the schema the model must follow, and `render()` is the
rich table. **Suggested exercise:** point it at your own meeting notes or an
email, or change the models to extract something else entirely (contacts,
invoice line items), the prompt barely changes.

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
[hands_on/streaming_server.py](hands_on/streaming_server.py), the three-phase
generator (`_stream_tokens`) is the core pattern every production streaming
endpoint follows.

**Suggested exercise:** point the server at `gpt-4o` and ask it something long,
then close the browser tab mid-response. Notice in the server logs that generation
stops immediately, with no wasted tokens.

---

## 11. The fourth capstone: `rag.py`

The embeddings example (Section 8) ranked sentences by similarity. `rag.py` puts
that to work: it answers questions over a small knowledge base by *retrieving*
the most relevant facts and pasting them into the prompt, the smallest thing
that is recognizably **retrieval-augmented generation (RAG)**. No vector
database, no framework; just the embeddings and chat calls you already know,
wired together from scratch.

The one idea to hold onto: **a model can only answer from what's in its context
window. RAG just decides what to put there.**

```bash
# Answer the built-in demo question from the knowledge base:
secrun python hands_on/rag.py

# Ask your own:
secrun python hands_on/rag.py "Can I get a refund?"

# The killer contrast: the same question with NO retrieved context:
secrun python hands_on/rag.py "How long are deleted notes kept?" --no-rag

# See exactly what gets retrieved and what prompt gets sent:
secrun python hands_on/rag.py "What plans are there?" -k 5 --show-prompt
```

The knowledge base is about a *made-up* app, so the model can't fall back on
training, so a correct answer can only come from retrieval. Run it with `--no-rag`
and watch the model guess or refuse; that contrast *is* the lesson. The
embeddings call and the chat call use the same `OPENAI_API_KEY`.

Read the source in [hands_on/rag.py](hands_on/rag.py): `retrieve()` is the whole
embed → score → rank loop, and `build_user_message()` is the entire "augment"
step; RAG is mostly good string assembly. **Suggested exercise:** add a fact to
`KNOWLEDGE_BASE`, then ask a question only that fact can answer.

---

## Where to go next

You've now covered the essentials, the common extensions, and four capstone
projects. Further on:

- **Retrieval-augmented generation (RAG) at scale**: you just built a minimal
  version in `rag.py`, which re-embeds a handful of facts on every run. Real
  systems embed once into a **vector database**, **chunk** long documents, and add
  **reranking** and **evaluation**, enough moving parts to be a deep dive of its
  own.
- **The context window**: what happens as conversations get long, and smarter
  ways to manage history than the simple trim in example 12 (summarizing old turns,
  sliding windows). It's a whole dive: [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive).
- **Vision & audio**: passing images to multimodal models, speech-to-text.
- **Streaming + tools together**: the pattern most production assistants use; built
  hands-on in the [Agents dive](https://github.com/alexvervloet/agents-deep-dive) (streaming
  inside the tool loop).

Each of these slots neatly on top of the "send messages, get a message" idea you
started with.

> 📌 **A note on the Responses API.** This dive uses **Chat Completions**
> (`client.chat.completions.create`), the long-stable, universal interface that
> every other provider and local server also implements, which is exactly why it's
> the right thing to learn first. OpenAI now also offers a newer **Responses API**
> (`client.responses.create`), which folds tools, state, and multi-step runs into one
> endpoint and is their recommended default for *new* OpenAI-only apps. The
> primitives are identical. You still send messages and get a message back, with the
> same models, streaming, structured outputs, and token accounting, so everything
> here transfers directly. Reach for Responses when you want its built-in
> conversation state and server-side tools and don't need provider portability; reach
> for Chat Completions (this dive) when you want the lowest-common-denominator
> interface that runs everywhere.

---

## Troubleshooting

Hit a snag? Run `secrun python check_setup.py` first; it catches most problems. The
rest, by the error you see:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `ModuleNotFoundError: No module named 'openai'` | Dependencies aren't installed (or your venv isn't active). Run `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `Set OPENAI_API_KEY ...` on every script | No key found. Store it in your keychain and run the script under `secrun`. See [SECRETS.md](../SECRETS.md). (The offline token/cost parts in Sections 5–6 still run without a key.) |
| `AuthenticationError` / 401 | The key is present but wrong: expired, revoked, or a typo. Make a fresh one at the [API keys page](https://platform.openai.com/api-keys). |
| `RateLimitError` / 429 | Too many requests, or you're out of credit. Wait a moment, or check your billing/usage in the dashboard. |
| `NotFoundError` / 404 about the model | A model name was mistyped or your account can't access it. The examples use widely-available IDs; if you changed one, check it against the [models list](https://platform.openai.com/docs/models). |
| `SyntaxError` or odd type errors on startup | You're likely on Python 3.9 or older. This repo needs 3.10+; `check_setup.py` will confirm your version. |
| It "hangs" with no output | Some examples stream, others wait for the full reply before printing. Give it a few seconds; for streaming examples you'll see text appear word by word. |

Still stuck? Every example is small and self-contained. Open the file, read the
docstring at the top, and run it directly. The error message almost always points
at the line.

---

## From teaching code to production

Every example here takes shortcuts that are perfect for learning and wrong for a
real deployment. Here's the map from each shortcut to what production uses:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| The answer goes to `print()` | One **structured trace** per request (id, timing, tokens) you can search after the fact |
| `estimate_cost()` just prints a number | An enforced **budget** that refuses the call before it overspends |
| A bare `client.chat.completions.create(...)` | The call wrapped in **retries + backoff** and a **circuit breaker** for 429s/503s/timeouts |
| Every call hits the API | A **response cache** so repeat questions cost nothing |
| Model id and system prompt are string literals in the script | **Versioned prompts/models** behind config, promoted only past an **eval gate** |
| You trust whatever the model returns | **Input/output guardrails** on the request path |

These shortcuts are right for learning and wrong for production. All seven
concerns (observability, cost, reliability, caching, guardrails, prompt
versioning, and eval gates) are built from scratch and wired into one running
app in **[Production](https://github.com/alexvervloet/ai-in-production-deep-dive)** (#8 in the
series). It runs **offline on a mock provider**, so you can see the whole ops
machinery with no key and no cost.

---

## File map

```
check_setup.py              ← run first: verifies Python, packages, and your key
EXERCISES.md                ← active-recall prompts, one per README section
hands_on/
  ask.py                    ← capstone CLI: ask a question about a code file
  extract.py                ← capstone CLI: extract validated data from free text
  streaming_server.py       ← capstone server: stream AI responses over SSE
  rag.py                    ← capstone CLI: answer questions over a knowledge base (RAG)
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
  17_local_serving.py       ← same client, local model via base_url (Ollama/llama.cpp)
  18_vision.py              ← send an image (URL or local base64) alongside text
  19_reasoning.py           ← o-series reasoning models: reasoning_effort, hidden tokens
  20_batch_api.py           ← submit many requests at 50% off, results within 24h
  21_prompt_caching.py      ← automatic prefix caching; structure prompts to hit it
  22_async_concurrency.py   ← AsyncOpenAI + asyncio.gather + a Semaphore (throughput)
  23_moderation.py          ← the free safety classifier (flags + per-category scores)
  24_logprobs.py            ← token probabilities -> confidence & calibrated classification
  25_seed_determinism.py    ← seed pins down randomness (best-effort reproducibility)
  26_responses_api.py       ← the other endpoint: hosted tools + server-side state
```

---

### Footnote: quieting Pylance/type-checker noise

Two patterns trip the type checker repeatedly with the OpenAI SDK. Pre-empt them
and new files stay clean:

1. **Assigning `messages` / `tools` to a variable** (rather than passing the
   literal straight into `create()`) makes Pylance infer a too-narrow type like
   `list[dict[str, str]]`. Annotate with the SDK's own param types; you also get
   key autocomplete:
   ```python
   from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
   messages: list[ChatCompletionMessageParam] = [...]
   tools: list[ChatCompletionToolParam] = [...]
   ```
2. **`.content` is typed `str | None`** (it's `None` when the model returns only
   tool calls). `print()` and f-strings accept it, but `json.loads()` and `+`
   concatenation don't, so guard with `... or ""` / `... or "{}"`.

The repo's [.vscode/settings.json](.vscode/settings.json) also sets
`python.analysis.typeCheckingMode` to `basic`, which keeps the useful checks
(undefined names, bad attrs/args) while dropping the strict dict-vs-TypedDict
complaints.

---

## The series

This is one of sixteen standalone, hands-on deep dives into building with LLM APIs: eight core, plus eight bonus dives.
Each one stands on its own, with its own setup, examples, and capstone, and they
all share the same house style: provider-agnostic, built from scratch (no
frameworks), offline-first examples, and a real capstone. Do them in any order;
this sequence builds naturally:

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts (zero/few-shot, chain-of-thought, roles)
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end: observability, cost, reliability, caching, guardrails, prompt versioning, eval gates

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window: memory, compaction, assembly
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images & audio, not just text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time: drift, quality, alerting, the flywheel

**You are here: #1, OpenAI API.**
