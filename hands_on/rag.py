#!/usr/bin/env python3
"""
rag.py — Retrieval-Augmented Generation, from scratch.
======================================================

This capstone ties two earlier pieces together: embeddings (Section 8 /
examples/11_embeddings.py) and a chat call (Section 2). It is the smallest thing
that is recognizably **RAG** — no vector database, no framework, ~one screen of
real logic.

The one big idea:

  >> A model can only answer from what's in its context window. RAG is just
  >> deciding what to put there — you *retrieve* the most relevant text and
  >> *stuff it into the prompt* before asking.

The pipeline, end to end:

  1. EMBED the knowledge base — turn each document into a vector
     (via `client.embeddings.create`, the dedicated embeddings endpoint).
  2. EMBED the question the same way.
  3. RETRIEVE — score every document against the question with cosine
     similarity and keep the top-k closest in meaning.
  4. AUGMENT — paste those documents into the prompt as "context."
  5. GENERATE — ask the chat model to answer using ONLY that context.

Why a *made-up* knowledge base below? So you can prove retrieval is doing the
work. The model has never seen "Nimbus Notes" in training, so it can only answer
correctly when the right document is retrieved and pasted in. Try `--no-rag` to
watch it fail (guess or refuse) without the context.

Everything here uses a single OPENAI_API_KEY — the embeddings call and the chat
call hit the same account.

A real app embeds the knowledge base ONCE and stores the vectors (that's what a
vector database is for); here we re-embed every run to keep the moving parts
visible. That storage-and-scaling layer is exactly where a dedicated RAG deep
dive picks up.

Examples
--------
  # Answer the built-in demo question from the knowledge base
  python hands_on/rag.py

  # Ask your own question
  python hands_on/rag.py "Can I get a refund?"

  # See the contrast: same question, but with NO retrieved context
  python hands_on/rag.py "How long are deleted notes kept?" --no-rag

  # Retrieve more documents, and print the exact prompt that gets sent
  python hands_on/rag.py "What are the plans?" -k 5 --show-prompt
"""

import argparse
import math
import os
import sys

# Make the repo root importable so `utils.*` resolves from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from utils.pricing import estimate_cost, estimate_embedding_cost, format_cost

# --- The "knowledge base": a few short, made-up facts about a fictional app. ---
# Each string is one "document" (or chunk). These are things the model cannot
# know from training, so a correct answer can only come from retrieval.
KNOWLEDGE_BASE = [
    "Nimbus Notes has three plans: Free, Plus ($4/month), and Team ($10/user/month).",
    "The Free plan is limited to 1 GB of storage and syncing across 2 devices.",
    "Deleted notes are moved to Trash and kept for 30 days before being permanently removed.",
    "You can export any notebook to Markdown, PDF, or HTML from Settings -> Export.",
    "All Nimbus Notes customer data is stored in data centers located in Frankfurt, Germany.",
    "Annual subscriptions can be refunded in full within 14 days of purchase.",
    "Offline editing is available on the Plus and Team plans, but not on Free.",
    "Two-factor authentication can be enabled under Settings -> Security.",
]

DEMO_QUESTION = "How long do I have to recover a note I deleted?"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# The grounding instruction. This is what keeps a RAG system honest: answer from
# the supplied context, and admit ignorance rather than inventing facts.
GROUNDED_SYSTEM = (
    "You are a support assistant for an app called Nimbus Notes. Answer the "
    "user's question using ONLY the context provided in their message. If the "
    "context does not contain the answer, say you don't know — do not guess or "
    "use outside knowledge."
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors: 1.0 = identical meaning, 0 =
    unrelated. The same by-hand formula from examples/11_embeddings.py."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def retrieve(client, question: str, corpus: list[str], k: int):
    """Embed the question + corpus, score by cosine similarity, return the
    top-k (score, text) pairs and the number of tokens the embeddings billed.

    One embeddings call handles everything: we send the question first, then the
    documents, and split the returned vectors back apart.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=[question] + corpus)
    vectors = [item.embedding for item in response.data]
    query_vec, doc_vecs = vectors[0], vectors[1:]
    scored = [
        (cosine_similarity(query_vec, vec), text)
        for text, vec in zip(corpus, doc_vecs)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k], response.usage.prompt_tokens


def build_user_message(question: str, chunks: list[str]) -> str:
    """Paste the retrieved documents into the prompt as context, then ask. This
    is the whole "augment" step — RAG is mostly good string assembly."""
    context = "\n".join(f"- {c}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a question over a small knowledge base, RAG-style.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEMO_QUESTION,
        help=f"Question to answer (default: a built-in demo: {DEMO_QUESTION!r}).",
    )
    parser.add_argument(
        "--model",
        default=CHAT_MODEL,
        help=f"Model that writes the answer (default: {CHAT_MODEL}).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="How many documents to retrieve and stuff into the prompt (default 3).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip retrieval and ask the question with NO context — the contrast "
             "that shows why RAG matters.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the exact messages (context + question) sent to the model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    load_dotenv()

    # Both the embeddings call and the chat call need the same key.
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    from openai import OpenAI  # imported lazily, after the key check
    client = OpenAI()

    print(f"Question: {args.question}\n")

    embed_tokens = 0
    embed_cost = 0.0

    if args.no_rag:
        # No retrieval: ask the bare question. Watch it guess or refuse, because
        # the answer lives only in the (unsent) knowledge base.
        print("(--no-rag: asking with NO retrieved context)\n")
        system = "You are a support assistant for an app called Nimbus Notes."
        user_message = args.question
    else:
        k = max(1, min(args.top_k, len(KNOWLEDGE_BASE)))
        scored, embed_tokens = retrieve(client, args.question, KNOWLEDGE_BASE, k)
        embed_cost = estimate_embedding_cost(EMBED_MODEL, embed_tokens)

        print(f"Retrieved the top {k} documents (cosine similarity):")
        for score, text in scored:
            print(f"  {score:.3f}  {text}")
        print()

        system = GROUNDED_SYSTEM
        user_message = build_user_message(args.question, [t for _, t in scored])

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    if args.show_prompt:
        print("--- messages sent to the model ---")
        for m in messages:
            print(f"[{m['role']}] {m['content']}\n")
        print("----------------------------------\n")

    response = client.chat.completions.create(model=args.model, messages=messages)
    choice = response.choices[0]

    print("=" * 70)
    print(choice.message.content)
    print("=" * 70)

    # Report what each step cost: embeddings + the answer.
    usage = response.usage
    gen_cost = 0.0
    try:
        gen_cost = estimate_cost(args.model, usage.prompt_tokens, usage.completion_tokens)
    except KeyError:
        pass
    if not args.no_rag:
        print(f"\nEmbeddings:    {embed_tokens:,} tokens -> {format_cost(embed_cost)}")
    print(f"Generation:    {usage.prompt_tokens:,} in + {usage.completion_tokens:,} out "
          f"-> {format_cost(gen_cost)} ({args.model})")
    print(f"Total:         {format_cost(embed_cost + gen_cost)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
