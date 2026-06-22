# OpenAI API — A Guided Deep Dive

A hands-on playground for learning the OpenAI API **from zero**. You'll build a
real CLI tool that answers questions about your code, and along the way you'll
understand every moving part: chat completions, roles, the sampling knobs
(temperature, top_p, max_tokens, stop), token counting, and cost.

This repo is meant to be *walked through*, not just read. Each section ends with
something to run. Do the running — that's where the learning is.

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
```

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
python tokens.py          # see a sentence broken into tokens
```

Why counting matters:
1. **Cost** — you're billed per token (next section).
2. **Limits** — every model has a max context window (input + output). Overflow
   it and the request fails.
3. **Intuition** — watching the count change as you edit a prompt teaches you
   how the model "sees" your text.

See [tokens.py](tokens.py) for `count_tokens()` (a raw string) and
`count_message_tokens()` (a full chat list, including the small per-message
overhead the API adds).

---

## 6. Cost estimation

OpenAI charges **separately for input and output tokens**, and output is
typically several times more expensive. [pricing.py](pricing.py) holds a small
price table and an `estimate_cost()` helper.

```bash
python examples/07_token_counting.py   # tokens -> dollars, across models, offline
```

Sample output shows the same request costing wildly different amounts per model
— which is why **choosing the right model is part of prompt engineering.**

> ⚠️ Prices change. The table in `pricing.py` is a snapshot — always confirm at
> <https://platform.openai.com/docs/pricing>.

---

## 7. The capstone: `ask.py`

Everything above comes together in one tool: ask a question about a code file,
see the token count and estimated cost *before* you spend, get the answer, and
see the *actual* usage and cost after.

```bash
# See the cost first — no API call, no key needed:
python ask.py snippets/buggy.py "Is there a bug here?" --dry-run

# For real (needs your key in .env):
python ask.py snippets/buggy.py "Is there a bug here?"

# Now turn the knobs you just learned:
python ask.py snippets/buggy.py "Rewrite this cleanly" --temperature 0
python ask.py snippets/buggy.py "List the issues" --max-tokens 200 --stop "4."
python ask.py snippets/buggy.py "Explain this" --model gpt-4o
```

Run `python ask.py --help` to see every knob explained inline. Read the source
in [ask.py](ask.py) — it's commented as a tutorial, especially `build_messages()`
(how the request is assembled) and the usage/cost reporting at the end.

**Suggested exercise:** point `ask.py` at your *own* code, try the same question
at `--temperature 0` vs `--temperature 1.2`, and watch both the answers and the
cost change.

---

## 8. Beyond the basics

With the core down, here are four of the most useful next capabilities — each is
a runnable example in the same numbered style. Every one is still just a variation
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

---

## Where to go next

You've now covered the essentials and the most common extensions. Further on:

- **Retrieval-augmented generation (RAG)** — combine embeddings (Section 8) with
  chat to answer questions over your own documents.
- **The context window** — what happens as conversations get long, and how to
  manage / trim history.
- **Vision & audio** — passing images to multimodal models, speech-to-text.
- **Streaming + tools together** — the pattern most production assistants use.

Each of these slots neatly on top of the "send messages, get a message" idea you
started with.

---

## File map

```
ask.py                      ← the capstone CLI (start here after the examples)
tokens.py                   ← tiktoken-based token counting
pricing.py                  ← price table + cost estimation
snippets/buggy.py           ← a sample file to ask questions about
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
```
