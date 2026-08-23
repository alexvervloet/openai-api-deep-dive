# Chapter 1: The API Call

*This is the textbook chapter for the OpenAI API deep dive. The [README](README.md) is the lab manual: it tells you what to run and in what order. This chapter is the lecture: what these things are, why they exist, and how they came to work the way they do. Read them in whichever order suits you, but do both. The lab without the theory is cargo culting; the theory without the lab is trivia.*

---

## 1.1 The strange economics of a text-completion machine

In June 2020, OpenAI did something unusual for an AI research lab: instead of publishing a model, it published an API. GPT-3 was, at the time, the largest language model ever trained, and you could not download it. You sent it text over HTTP and it sent text back, metered by the syllable, more or less. Researchers grumbled. Then a strange thing happened. People with no machine learning background at all started building products on it, because the interface asked nothing of them except the ability to make a web request.

That decision, model behind an API rather than model as a download, is the reason this course starts where it does. Almost everything in modern AI engineering happens on the application side of that HTTP boundary. You will probably never train a model. You will send requests to one, thousands or millions of times, and the quality of your product will be determined by how well you construct those requests and handle what comes back.

When ChatGPT arrived in November 2022, it looked like a leap. Under the hood it was mostly packaging: a chat interface, some fine-tuning to make the model behave like an assistant, and a conversation loop. The API you are about to learn is that machinery with the lid off. Once you understand it, ChatGPT stops being magic and becomes something you could sketch on a whiteboard. That is the goal of this chapter.

The one big idea, stated once here and then demonstrated over the following pages:

> **You send a list of messages. You get back a message.**

Every feature in this chapter, and honestly most features in this entire course, is a refinement of that sentence.

## 1.2 What actually crosses the wire

Strip away the SDK and a chat completion request is a small piece of JSON sent to an HTTPS endpoint. It names a model, carries a list of messages, and optionally tweaks a handful of settings. The response is another piece of JSON containing the model's reply and an accounting of what you were billed.

It is worth pausing on how little is there. There is no session ID. There is no login for the conversation. There is no "context" object living on OpenAI's servers, waiting for your next message. The request you send is the entire universe the model sees, and when the response comes back, that universe is discarded.

This property has a name: the API is **stateless**. It is the single most common thing newcomers get wrong, so let's make it concrete. When you chat with ChatGPT and it remembers your name from ten messages ago, the model is not remembering anything. The application is re-sending the whole transcript, your name included, with every single turn. The model reads the entire conversation from the top, every time, and produces one more message. Then it forgets everything.

Why build it this way? Statelessness is an old and deliberate engineering choice, the same one that made the web itself scale. If no server needs to remember your conversation, then any server can handle your next request. OpenAI can route your call to whichever machine in whichever data center has capacity, and nothing breaks. The cost of that scalability is pushed onto you, the application developer: memory is now your job. You will spend a surprising amount of this course, and an entire later dive (Context Engineering), on the consequences.

## 1.3 Roles, or why the transcript is tagged

The messages in your list are not plain strings. Each one carries a **role**, and the roles are how the model knows who said what.

The **user** role is what the human typed. The **assistant** role is what the model said previously; you replay these to give the conversation its memory. The interesting one is **system**.

A system message is standing instructions: who the model should be, what rules it should follow, what tone to take. It is set once, usually first, and it colors everything that follows. If you change nothing else about a request except the system message, swapping "You are a helpful assistant" for "You are a grumpy pirate," the same question produces an entirely different answer. The lab has you do exactly this, and it lands better as an experiment than as a sentence in a book.

Why does a role tag carry so much power? Because of how these models were trained. After the base model learned to predict text from a huge slice of the internet, it went through a second phase where humans rated its answers and the model was tuned toward the highly rated ones. During that phase, the training data was formatted as tagged transcripts, and instructions in the system slot were consistently the ones the model was supposed to obey. The model learned the convention the way it learned everything else, by pattern. The system role is not a security boundary or an enforcement mechanism. It is a very strong habit. That distinction sounds academic now; it becomes the entire subject of the Prompt Injection dive, where you will watch people talk models out of their instructions.

For now, the practical takeaway: the system message is your most powerful single lever, and it is where the "programming" of an AI product mostly lives. Companies building on these APIs treat their system prompts as core intellectual property, version them like code, and test changes to them like releases. That is not paranoia. A one-sentence change in a system prompt can alter product behavior more than a month of feature work.

## 1.4 The knobs: shaping how, not what

A language model does not produce text. It produces, at each step, a probability for every token it could emit next, and then something has to pick one. The sampling parameters control that picker, and understanding them means understanding one idea: the model is always choosing from a weighted menu.

**Temperature** rescales the weights. At temperature 0, the picker always takes the single most probable token, so the same request produces (nearly) the same answer every time. As temperature rises, the picker gets more adventurous, giving real chances to tokens further down the menu. At 1.5 and above, the prose starts to wobble; push far enough and it dissolves into word salad. The name comes from physics, where temperature governs how much particles jitter around their lowest-energy state, and the analogy is honest: a cold model settles into the groove, a hot one bounces out of it.

There is no universally correct setting, only fit to task. Extracting data from a document, answering questions about code, anything where there is a right answer: go low. Brainstorming taglines, generating variations, creative work: go high. The default of around 1.0 (some SDKs and models present 0.7) is a compromise you should feel free to override, and in this course you will usually override it downward.

**top_p** attacks the same problem from a different angle. Instead of rescaling the menu, it truncates it: consider only the smallest set of tokens whose probabilities add up to p, and choose among those. At top_p 0.1 the model may only pick from the obvious candidates; at 1.0 the whole menu is available. This is called nucleus sampling, and the standard advice, repeated here because it is good, is to tune temperature or top_p but not both. They interact in ways that are hard to reason about, and a setting that behaves one way alone behaves differently in combination.

**max_completion_tokens** is not a creativity knob at all; it is a budget cap on the length of the answer. The model does not know the cap exists and does not write shorter to fit it. It just gets cut off, sometimes mid-sentence, when the budget runs out. The response tells you which happened: a `finish_reason` of "stop" means the model chose to end, "length" means the guillotine came down. Checking that field is the difference between an app that truncates answers and never says so, and one that knows it did.

**stop** sequences end generation the moment a given string would appear. They sound like a niche tool and mostly are, but when you want exactly three list items, or output up to a delimiter, they are cleaner than asking politely and hoping.

A pattern worth noticing: none of these knobs change what the model knows or what it can do. They change how it selects among things it was already inclined to say. Later rungs of the course (retrieval, tools, fine-tuning) are for changing the other things. Keeping "which lever changes what" straight is half of AI engineering judgment, and there is a whole guide in the root of this series ([CHOOSING.md](../docs/CHOOSING.md)) devoted to it.

## 1.5 Tokens: the unit of everything

Here is a fact that surprises almost everyone: the model never sees your words. Before your text reaches the network, it is chopped into **tokens**, integer IDs drawn from a fixed vocabulary of roughly a hundred thousand pieces. Common words are single tokens. Rarer words get split: "unbelievable" might become "un", "belie", "vable". Whitespace and punctuation get folded in. The model reads a sequence of these integers and emits more of them, and only at the very end are they decoded back into text for you.

Why bother? A model needs a fixed, finite vocabulary to compute probabilities over, and you cannot enumerate all possible words (people invent new ones constantly, and that is before you consider typos, code, and other languages). Characters would work but make every text enormously long, and these models pay a steep computational price for length. Tokens are the compromise: a vocabulary small enough to be tractable, with pieces large enough that ordinary text stays short. The specific method, byte pair encoding, builds the vocabulary by repeatedly merging the most frequent adjacent pairs in a training corpus, which is why the pieces align eerily well with English morphemes and common code fragments.

Tokens matter to you for three concrete reasons.

First, **money**. You are billed per token, with separate rates for input (what you send) and output (what the model generates), and output typically costs several times more. This asymmetry is not arbitrary: generating a token requires a full pass through the model, one token at a time, while input tokens can be processed in parallel. A rough rule for English is that a token is about four characters, or three-quarters of a word, but "rough" is not a budget, which is why the lab has you count exactly, offline, using the same tokenizer the API uses.

Second, **limits**. Every model has a maximum context window, the total tokens of input plus output it can handle in one request. Exceed it and the request fails. Context windows have grown from 4,000 tokens to hundreds of thousands in a few years, and applications have grown to fill them just as fast.

Third, **intuition**. Odd model behaviors become explicable once you think in tokens. Models are famously bad at counting the letters in a word, and how could they not be: they never see letters. Arithmetic on long numbers is hard partly because numbers tokenize into irregular chunks. When you can look at a prompt and estimate its token count within ten percent, you have acquired an actual professional skill, unglamorous as it sounds.

The billing asymmetry also has a design consequence worth internalizing early: a cheap application is one that sends fat, well-prepared input and asks for lean output. That single sentence explains a lot of production prompt design.

## 1.6 Streaming, and why the answer arrives as a drip

Ask a model for a long answer and it might take twenty seconds to finish. Twenty seconds of blank screen kills a product; the same twenty seconds with text flowing feels fine. This is not a trick you bolt on afterward. The model genuinely produces tokens one at a time, so the most natural thing in the world is to forward each one to the user as it appears. That is streaming, and it is why every AI chat interface you have used shows the answer being "typed."

The wire protocol underneath is Server-Sent Events, a piece of web plumbing that predates the AI boom by more than a decade. An SSE response is just an HTTP response that never closes, dripping lines prefixed with `data:` until the server is done. It was designed for stock tickers and notification feeds, sat in relative obscurity for years, and then turned out to be exactly the right shape for language model output. There is a lesson in that about boring technology.

The SDK hides the parsing, and for a while you can let it. But the moment you build a backend that sits between the model and a browser (which is to say, the moment you build a real product), you are implementing the middle of a relay: receive SSE from the provider, forward events to the client, notice when the client disconnects, and stop the upstream generation so you do not pay for tokens nobody will read. The lab's streaming server capstone walks through exactly that, including the disconnect handling, which everyone forgets and which costs money.

## 1.7 From prose to data: structured output and tool calling

Everything so far treats the model's output as text for a human to read. The turning point in the API's history, and honestly in the industry's history, was making the output reliable enough for a *program* to read.

The early workaround was begging. Prompts ended with "Respond only with valid JSON, no other text," and then your code wrapped the parse in a try/except and prayed. It worked most of the time, and "most of the time" is a miserable foundation for software. **Structured outputs** fixed this properly: you supply a schema, and the API constrains generation so that the reply cannot violate it. Not "is asked nicely not to." Cannot: at each step, tokens that would break the schema are masked out of the menu before sampling. In the Python SDK you can hand over a Pydantic model and receive a validated, typed instance back, which turns the language model into something your codebase can treat like any other function that returns data.

**Tool calling** (originally "function calling," introduced by OpenAI in June 2023) is the same idea pointed in the other direction. You describe functions the model may use: their names, what they do, what arguments they take. The model, when it decides a function would help, replies not with prose but with a structured request: call this function, with these arguments. You run the function (the model cannot execute anything itself; it can only ask), append the result to the conversation, and call the API again so the model can incorporate it.

Read that loop again, because it is the seed of everything in the second half of this course. The model chooses an action, you perform it, the result goes back in, repeat until done. When people say "agent," that loop is what they mean. In this dive it appears in its smallest form, one tool, one round trip. By the Agents dive it will have grown into the whole show.

It is also worth saying plainly why tool calling matters beyond mechanics: it is the model's escape hatch from its own limitations. A language model cannot do reliable arithmetic, cannot know today's weather, cannot query your database. A language model that can ask *your code* to do those things inherits all of those abilities without pretending to have them. The pattern of "let the model decide, let deterministic code execute" turns out to be the sturdiest division of labor in the field.

## 1.8 Embeddings: a different endpoint entirely

Tucked among the examples is one that does not fit the messages-in, message-out mold, and it may be the most consequential of the lot. The embeddings endpoint takes text and returns a long list of numbers, a vector, positioned so that texts with similar meaning end up near each other in space. "How do I get my money back?" and "What is your refund policy?" share almost no words, but their embeddings sit close together, because the model that produces them was trained to place meaning, not spelling.

That property makes search work by meaning instead of by keyword, and search-by-meaning is the foundation of retrieval-augmented generation, the technique for giving models knowledge they were never trained on. It gets a full dive of its own (RAG, chapter 4), and this repo's final capstone builds a miniature version from scratch: embed a small knowledge base, embed the user's question, find the nearest facts, paste them into the prompt. No database, no framework, maybe sixty lines. When you later meet vector databases and rerankers and chunking strategies, you will know they are optimizations of those sixty lines, not something categorically new.

## 1.9 The supporting cast

A production application uses a handful of other capabilities that deserve a paragraph each, less because they are deep than because knowing they exist saves you from reinventing them.

**Error handling** is where the real world intrudes. Networks blip, rate limits trigger, servers have bad minutes. The SDK already retries transient failures with exponential backoff, jittered so that a thousand clients recovering from the same outage do not all retry in the same instant and knock the server over again (this stampede has a name, the thundering herd, and it has taken down real systems). Your job is narrower than people assume: set sensible timeouts, and treat "your request is malformed" differently from "try again in a moment." One is a bug to fix; the other is weather.

**The Batch API** is for work with no human waiting on it. Submit a file of requests, get results within a day, pay half price. Classifying ten thousand support tickets overnight at 50% off is the kind of decision that shows up on invoices, and it requires no cleverness at all, just knowing the option exists.

**Prompt caching** exploits a structural fact: most applications send the same long prefix (system prompt, instructions, reference document) over and over with only the final question changing. Providers cache the processed prefix and re-bill it at a discount. On OpenAI this is automatic, and the entire skill is putting the constant part first and the variable part last. A one-line reordering of your prompt can cut input costs by half or more at scale.

**Moderation** is a free classifier that flags harmful content, run on input before it reaches your model and on output before it reaches your user. Free safety layers are rare; use this one.

**Logprobs** expose the probabilities behind each chosen token, which turns the model's private uncertainty into a number you can act on: route low-confidence classifications to a human, flag shaky answers, debug why the model keeps picking the wrong label.

**Seeds** make sampling reproducible on a best-effort basis, which matters for tests and evaluations. Best-effort is the honest phrase: the API returns a fingerprint of the backend configuration, and when the fingerprint changes, determinism breaks with nothing to announce it. Treat reproducibility as a strong hint, not a contract.

**Reasoning models** (the o-series and their successors) spend hidden tokens thinking before they answer, which buys real gains on math, logic, and code at the price of latency and cost, including paying for thinking you never see. They also change what you steer with: effort, rather than temperature. They get fuller treatment in the Claude chapter, where the equivalent feature exposes its reasoning for inspection.

## 1.10 What you paid for the abstraction

A textbook should tell you not just what a design gives you but what it costs. The chat completions abstraction is clean, and three of its edges will cut you if you do not know where they are.

The transcript is quadratic in disguise. Because the API is stateless and you re-send the whole conversation each turn, a chat of n turns transmits each early message about n times. Long-running conversations get slower and more expensive per turn, invisibly, until suddenly it is a problem. Every serious chat product eventually confronts this; the Context Engineering dive is about the confrontation.

The model's fluency is not tied to its accuracy. The API will return confident, grammatical, well-structured wrongness with exactly the same tone as truth, because producing plausible text is the only thing the machine does. Nothing in the response schema marks an answer as ungrounded. The disciplines for coping (retrieval, evaluation, guardrails) are chapters 4, 5, and 7, and the reason they exist is this paragraph.

The interface is friendlier than the system. A request that types-checks and returns 200 can still be a bad request: too expensive, too slow, poorly cached, unmoderated. The gap between "works" and "works in production" is wide enough that it gets two dives (Production and Observability) at the end of the course.

None of these are reasons to avoid the API. They are the reasons the rest of this course exists.

## 1.11 Where this chapter leaves you

The lab for this dive ends with four small capstones: a CLI that answers questions about code with the cost shown before you spend, an extractor that turns messy notes into validated data, a streaming web server, and a from-scratch RAG pipeline. Between them they touch every idea in this chapter, which is the point. You should leave this dive with the reflexes, not just the vocabulary: check `finish_reason`, count tokens before sending, put the constant part of the prompt first, treat the model's confidence as a style rather than a signal.

One closing note on portability, because it explains a choice this course makes. This dive teaches the Chat Completions interface even though OpenAI now recommends its newer Responses API for new OpenAI-only projects. The reason is that Chat Completions escaped. Nearly every other provider, and every local model runtime you will meet in the Local Models dive, implements it as a de facto standard, the way SQL outgrew IBM. Learn it once and you can talk to almost anything that generates text, including a model running on your own laptop with no API key at all. The next chapter looks at the most important dialect: the same ideas, as Anthropic builds them.

---

*Lab manual: [README.md](README.md) · Exercises: [EXERCISES.md](EXERCISES.md) · Next chapter: [Claude API](../claude-api-deep-dive/TEXTBOOK.md)*
