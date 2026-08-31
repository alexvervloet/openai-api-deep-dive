# Exercises: make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts: a thing to predict, a thing to change, and a question to
answer from memory. None take more than a couple of minutes.

How to use it: work the section in the README first, then come back here. For each
exercise, **commit to an answer before you run or reveal.** The prediction is
where the learning happens, even (especially) when you're wrong. Answers are
hidden behind ▸ toggles.

> Most of these cost a fraction of a cent. The ones marked **(offline)** make no
> API call at all.

---

## Section 2: Your first request

**Predict.** Before running `examples/01_basic_chat.py`, what *type* is
`response.choices`: a string, a dict, or a list? Why does the code say
`choices[0]`?

<details><summary>▸ Answer</summary>

A **list**. A single request can return several alternative completions (via the
`n` parameter), so the replies live in a list and `[0]` takes the first. With the
default `n=1` there's exactly one.
</details>

**Do.** Change the question in the script to something with a clearly long answer
("Explain how TCP works"). Run it and look at `response.usage`. Which is larger,
`prompt_tokens` or `completion_tokens`? Did that match your guess?

---

## Section 3: Roles

**Do.** Open `examples/02_roles.py` and set the system message to
`"You answer only in haiku."` Rerun. Then move that same instruction into the
*user* message instead of the system message. Does it still obey? Which placement
felt more reliable?

**Recall.** The API is *stateless*. If that's true, how does a chatbot "remember"
your name from three messages ago?

<details><summary>▸ Answer</summary>

It doesn't. *You* do. Every turn you resend the entire `messages` list,
including the earlier user and assistant turns. The "memory" is just that growing
list being sent each time. (You'll build exactly this in `examples/12_conversation.py`.)
</details>

---

## Section 4: The knobs

**Predict, then run.** You run `examples/03_temperature.py` twice at
`temperature=0`. How similar will the two answers be? Now twice at
`temperature=1.5`?

<details><summary>▸ Answer</summary>

At `0` they'll be nearly identical every time (focused, near-deterministic, though
never a 100% guarantee). At `1.5` they'll diverge, sometimes wildly. This is the
whole point of the knob: low for facts/code, high for variety.
</details>

**Do.** In `examples/04_max_tokens.py`, set `max_completion_tokens` to something tiny like
`10` and ask for a paragraph. Inspect `finish_reason`. What value do you get, and
what does it tell you?

<details><summary>▸ Answer</summary>

`"length"`: the model was cut off by your cap, not because it was done. A natural
finish shows `"stop"`. Watching `finish_reason` is how you detect truncated
answers in real code.
</details>

**Recall.** Why does the README warn against tuning `temperature` *and* `top_p` at
the same time?

<details><summary>▸ Answer</summary>

They both reshape the same probability distribution the model samples from, so
their effects interact in ways that are hard to reason about. Pick one lever and
leave the other at its default.
</details>

---

## Section 5: Tokens **(offline)**

**Predict, then run.** Run `python utils/tokens.py`. Before you do: will
`"unbelievable"` be 1 token or several? Will `"    "` (four spaces) cost anything?
Edit the `sample` string to test both.

<details><summary>▸ Answer</summary>

`"unbelievable"` splits into multiple sub-word tokens; common short words are
often one. Whitespace is *not* free: runs of spaces and newlines are tokens too.
Seeing the `Pieces:` list is the fastest way to build intuition for how the model
"sees" text.
</details>

**Do.** Take a chunk of your own code and a chunk of plain English of roughly the
same character count. Count both with `count_tokens()`. Which is denser in tokens,
and why might code cost more than prose?

---

## Section 6: Cost **(offline)**

**Predict.** A request is 2,000 input tokens and 500 output tokens. Using the
prices in `utils/pricing.py`, will it cost more on `gpt-5.4-nano` or `gpt-4o`?
Roughly how many times more?

<details><summary>▸ Answer</summary>

Far more on `gpt-4o`. Do the arithmetic with `estimate_cost()` to see the exact
multiple: then notice that **output** tokens dominate, since output is priced
several times higher than input. Choosing the cheaper model for a task is real
money saved.
</details>

**Do.** Add a fictional model `"gpt-4o-ultra"` at 4x the `gpt-4o` price to the
`PRICING` table and rerun your estimate. (Then delete it.) You've just learned how
to keep the table current when prices change.

---

## Section 7: Capstone: `ask.py`

**Do.** Run `ask.py` on `snippets/buggy.py` with `--dry-run`, note the estimated
cost, then run it for real and compare the estimate to the *actual* cost printed
at the end. Were they close? Where would they diverge most?

<details><summary>▸ Answer</summary>

The input estimate should be near-exact; the gap is on the output side, because
the dry run can only *assume* an output length. The real answer might be shorter
or longer than the assumed ~500 tokens, moving the actual cost accordingly.
</details>

**Stretch.** Point `ask.py` at one of your own files and ask the same question at
`--temperature 0` and `--temperature 1.2`. Compare both the answers and the cost.

---

## Section 8: Beyond the basics

**Recall.** In function/tool calling (`examples/10_function_calling.py`), the
model decides to call your function. Does the OpenAI API run your function for
you?

<details><summary>▸ Answer</summary>

No. The model only emits the function *name and arguments*. **You** run the
function and feed the result back in a follow-up message. The model never executes
your code. It only asks for it to be run.
</details>

**Predict, then run.** In `examples/11_embeddings.py`, the demo ranks sentences by
similarity to a query. Can a sentence that shares *no words* with the query still
rank highly?

<details><summary>▸ Answer</summary>

Yes, and that's the entire value of embeddings. They capture *meaning*, not word
overlap, so "the feline napped" can rank near "a cat is sleeping." This is why
embeddings power semantic search and RAG.
</details>

**Do.** Run `examples/08_streaming.py` and then `examples/01_basic_chat.py` back
to back. The total time-to-finish is similar, so what did streaming actually buy
you?

<details><summary>▸ Answer</summary>

Time-to-*first-token*. The user starts reading immediately instead of staring at a
blank screen until the whole answer is ready. Same total time, far better
perceived responsiveness, which is why chat UIs stream.
</details>

**Recall (vision, `18_vision.py`).** A text request sends a string for `content`.
What does an image request send instead, and how does the image itself travel?

<details><summary>▸ Answer</summary>

A **list of parts**, a `text` part and an `image_url` part, in one user message.
The image is either a URL the model fetches, or a local file inlined as a base64
`data:` URI. The image is billed as tokens, scaled by its pixel size.
</details>

**Predict (reasoning, `19_reasoning.py`).** Why does this example set
`reasoning_effort` instead of `temperature`, and what are `reasoning_tokens`?

<details><summary>▸ Answer</summary>

Reasoning models ignore sampling knobs like `temperature`; you steer how hard they
think with `reasoning_effort`. The `reasoning_tokens` are the model's **hidden**
chain of thought, generated before the visible answer and never shown to you, but
still billed.
</details>

**Recall (batch, `20_batch_api.py`).** What do you trade to get the Batch API's 50%
discount, and what ties each answer back to its input?

<details><summary>▸ Answer</summary>

You trade **immediacy**: results land within 24h instead of instantly. Each line's
`custom_id` is echoed in the results file, so you can match every answer back to the
request that produced it.
</details>

**Predict (caching, `21_prompt_caching.py`).** Two requests share a long system
prompt but ask different questions. Why must the *constant* part come first?

<details><summary>▸ Answer</summary>

Caching only helps the **prefix that's byte-for-byte identical**. Put the constant
block (system prompt, tool catalog, document) at the front and the variable question
at the back, and the long prefix is served from cache at a discount on the second
call (`cached_tokens`).
</details>

**Do (async, `22_async_concurrency.py`).** It runs 6 prompts sequentially, then 4-at-
a-time. Why is the concurrent run ~faster, and what is the `Semaphore` protecting?

<details><summary>▸ Answer</summary>

Each request is mostly **idle network waiting**, so overlapping them finishes in
about the time of the slowest call. The `Semaphore` caps how many run at once, so
you get the speedup without blowing past your account's **rate limit**.
</details>

**Recall (moderation, `23_moderation.py`).** Is the moderation endpoint a chat
model? When do you call it?

<details><summary>▸ Answer</summary>

No. It's a separate, **free** classifier returning category flags + scores. Call it
on **user input on the way in** and **model output on the way out**, refusing or
redacting when `flagged` is true.
</details>

**Predict (logprobs, `24_logprobs.py`).** For a confident yes/no answer vs. a
genuinely uncertain one, how do the `top_logprobs` differ?

<details><summary>▸ Answer</summary>

A confident answer puts **almost all probability on one token**; an uncertain one
**spreads** probability across alternatives. That spread is a usable confidence
signal: auto-accept the confident ones, route the shaky ones to review.
</details>

**Recall (seed, `25_seed_determinism.py`).** The example fixes `seed=42` at
`temperature=0.9` rather than `temperature=0`. Why, and why is the result still only
"best-effort," not a guarantee?

<details><summary>▸ Answer</summary>

At `temperature=0` the model is already deterministic (always the most likely token),
so a fixed seed wouldn't visibly be doing anything, and you couldn't tell its effect
apart from temperature=0's own determinism. Running at `temperature=0.9` keeps real
randomness in play, so a matching seed across two calls visibly pins the output down,
while an unset seed visibly doesn't. Either way, OpenAI can change the backend, and
determinism isn't guaranteed across such changes; the `system_fingerprint` field is
the tell: if it changes between calls, the backend shifted and identical inputs can
drift even with the same seed.
</details>

## Responses API mini-track

**Recall (`responses/01_request_and_items.py`).** Why does the example print
`response.output_text` for the answer but still enumerate `response.output`? What bug
can appear if application code reads only `response.output[0]`?

<details><summary>▸ Answer</summary>

`output_text` combines the text from message items and is the convenient path when text
is all you need. `output` is heterogeneous. A response can include reasoning, tool-call,
and message items, and the API does not promise that the first item is a message. Code
that assumes it is will fail as soon as the model or tool configuration adds another
item type.
</details>

**Predict (`responses/02_conversation_state.py`).** The follow-up request uploads one
short string plus `previous_response_id`. Which earlier data is still part of the model
context and token bill? Does the first request's `instructions` field carry forward?

<details><summary>▸ Answer</summary>

The response chain still supplies the earlier input and output to the model, and those
input tokens are billed again. The response ID reduces the transcript your client sends.
It does not reduce model context. Instructions do not carry forward, so the follow-up
must repeat them when they still apply.
</details>

**Debug (`responses/03_streaming_events.py`).** Replace the event loop with code that
prints `event.delta` for every event. Run it. Which event types have no `delta` field,
and which terminal information would the simplified loop lose even if it skipped those
errors?

<details><summary>▸ Answer</summary>

On a plain text run, exactly one of the nine event types carries `delta`:
`response.output_text.delta`. The other eight raise `AttributeError`.
`response.created`, `response.in_progress` and `response.completed` are lifecycle
events, `response.output_item.added/done` and `response.content_part.added/done`
announce structure, and `response.output_text.done` carries the finished string as
`text`, not `delta`.

Guarding with `getattr(event, "delta", "")` stops the crash and still loses the
answer. Only the terminal event carries `response`, and with it the final `status`
and `usage`. A loop that renders deltas and ignores everything else cannot tell a
completed answer from one truncated at `max_output_tokens`, and has no token counts
to log. It prints something that looks finished either way.
</details>

**Predict (`responses/04_custom_tool_loop.py`).** The tool schema uses strict mode and
an enum. Why does the application still keep a dispatch allowlist and validate the
decoded city before calling Python code?

<details><summary>▸ Answer</summary>

The schema constrains model output at the API boundary. It does not grant authority to
execute a function. The application owns that decision and must reject unknown names or
arguments before they reach code with database, network, or filesystem access. The
checks also protect the boundary if a response is replayed, forged, or produced under a
different tool definition.
</details>

**Do (`responses/05_hosted_web_search.py`).** Change `tool_choice="required"` to
`tool_choice="auto"` and ask a timeless factual question. Run it several times. Record
the output item types and explain why merely listing a tool is not proof that it ran.
Then restore `required` and confirm the call item appears.

<details><summary>▸ Answer</summary>

Asking "why is the sky blue?" under `auto` returned `['message']` on three runs out
of three. The tool was offered every time and used none of them, because the model
already had the answer and searching is a cost it avoids when it sees no need. Under
`required` the same request returns `['web_search_call', 'message']`.

So a tool in the request is a permission, not an event. `auto` says the model may
search; only a `web_search_call` item in `output` says it did. That gap is why the
example checks for the item rather than trusting the configuration, and it is the
same reasoning as example 04's dispatch allowlist seen from the other side: there,
naming a function does not run it; here, offering a tool does not use it. If your
logging records the request instead of the output items, you will believe you have
citations you never received.
</details>

**Do (`responses/06_background_responses.py`).** Start a response and save its ID. Use
`check`, then `wait`. Start another response and call `cancel` twice with the same ID.
Record every status you observe. Why must a production poller stop on any status outside
`queued` and `in_progress`, rather than waiting only for `completed`?

<details><summary>▸ Answer</summary>

`failed`, `cancelled`, and `incomplete` are terminal too. A loop that waits only for
`completed` can poll forever after work has already stopped. Cancellation is idempotent,
so the second cancellation returns the final response rather than starting a second
state transition.
</details>

**Predict (`responses/07_structured_outputs.py`).** The script sends the same schema
three ways: through `create`, through `parse`, and through `parse` with a 16-token cap.
Which of the three can hand your code a Python object it should not trust, and which
one never returns at all?

<details><summary>▸ Answer</summary>

`create` is the one to watch. Under the cap it returns normally with
`status="incomplete"` and `output_text` holding a truncated fragment such as
`'{"service":"checkout","severity":"sev2",'`. Nothing about that return says failure.
Call `json.loads` on it and you get a decode error at best, and if the truncation
happened to land on a syntactically complete object, a silently wrong record at worst.
The status field is the only thing that tells you.

`parse` under the cap never returns: it raises `pydantic.ValidationError` from inside
the SDK, because half an object cannot be validated into a whole one. That is the
friendlier failure of the two, and the reason to prefer `parse` when you can accept an
exception. Either way the schema did its job and the request still failed, which is
the distinction worth keeping: strict mode constrains the shape of what is emitted,
not whether emission finishes.
</details>

**Predict (`responses/08_conversation_object.py`).** The second turn uploads a
28-character question, and the transcript now lives on OpenAI's servers rather than in
your request. Predict the billed input tokens, then compare with the number example 02
printed for the same exchange over `previous_response_id`.

<details><summary>▸ Answer</summary>

They are the same order of magnitude and for the same reason: 60 tokens here against 66
for the response chain, both far more than the 28 characters uploaded. Where the
transcript is stored changes what your client transmits. It does not change what the
model reads, and you are billed for what the model reads.

This is the point both state mechanisms are most often misread on. Neither is a cache
and neither is a discount. `previous_response_id` and `conversation` are answers to
"who holds the transcript", one process versus one durable object, and the honest
reason to pick the Conversation is that items must outlive the process or be shared
across jobs. If you want the token bill to stop growing, that is a context-engineering
problem, and it is the subject of a later dive rather than a parameter on this request.
</details>

---

## Capstones 9, 10 & 11

**Do (`extract.py`).** Run it on `snippets/meeting_notes.txt`, then open the file
and add a line like `"Nobody owns the budget review."` Rerun. How did the model
handle an action item with no clear owner?

**Do (`streaming_server.py`).** Start the server, open the browser, and ask for a
long answer. Open the Network tab, find the `/stream` request, and watch the
`data:` lines arrive. Then close the tab mid-answer and check the server logs.
What did the server do the instant you disconnected, and why does that save money?

<details><summary>▸ Answer</summary>

It detected the client disconnect and stopped the model call immediately, with no
further tokens generated, nothing billed for output you'd never see. Detecting
disconnects is a real production cost lever, not just tidiness.
</details>

**Predict, then run (`rag.py`).** Run `secrun python hands_on/rag.py`, then run it again
with `--no-rag`. Will the answer change? Which one can you trust, and why?

<details><summary>▸ Answer</summary>

With retrieval, the model answers from the fact pasted into the prompt ("30
days"). With `--no-rag` there's no source, since "Nimbus Notes" is made up, so it
guesses or admits it doesn't know. That's the whole idea: a model can only answer
from what's in its context window, and RAG decides what to put there.
</details>

**Do (`rag.py`).** Add a new fact to `KNOWLEDGE_BASE` (say,
`"Nimbus Notes can import notebooks from Evernote and Notion."`) and ask a
question only that fact can answer. Use `--show-prompt` to confirm it actually got
retrieved into the context. If it didn't, what would you try? Reword the
question, or raise `-k`?

---

### Where to take it next

Invent your own. The best exercise is a question *you* genuinely don't know the
answer to. Change one thing, predict the effect, run it, and reconcile the
difference. That loop is the whole skill.
