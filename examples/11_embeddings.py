"""
Example 11: embeddings & semantic similarity.

So far every example used a *chat* model that produces text. Embeddings are
different: an embeddings model turns a piece of text into a list of numbers (a
"vector") that captures its *meaning*. Texts with similar meaning end up with
similar vectors, even if they share no words.

This is the engine behind semantic search, recommendations, clustering, and
"retrieval-augmented generation" (RAG), where you find the most relevant
documents to stuff into a prompt.

How we measure "similar": **cosine similarity**, the cosine of the angle between
two vectors. It ranges from -1 (opposite) to 1 (identical direction). Closer to 1
means more similar in meaning.

This example uses a different endpoint, `client.embeddings.create`, and a
cheap, dedicated model (`text-embedding-3-small`). No third-party math libraries
needed; we compute cosine similarity by hand.

Run it:

    secrun python examples/11_embeddings.py
"""

import math
import os
import sys

# Make the repo-root modules (utils/pricing.py) importable no matter where you run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from utils.pricing import estimate_embedding_cost, format_cost

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


query = "How do I reset my password?"
candidates = [
    "Steps to recover a forgotten login credential.",  # same meaning, no shared words
    "Our office is open from 9am to 5pm.",              # unrelated
    "Click 'Forgot password' to receive a reset link.", # clearly relevant
]

# You can embed many texts in one call: pass a list. We embed the query and all
# candidates together.
model = "text-embedding-3-small"
response = client.embeddings.create(
    model=model,
    input=[query] + candidates,
)
vectors = [item.embedding for item in response.data]
query_vec, candidate_vecs = vectors[0], vectors[1:]

print(f"Query: {query!r}\n")
print("Ranked by semantic similarity:")
scored = sorted(
    zip(candidates, candidate_vecs),
    key=lambda pair: cosine_similarity(query_vec, pair[1]),
    reverse=True,
)
for text, vec in scored:
    print(f"  {cosine_similarity(query_vec, vec):.3f}  {text}")

print(f"\n(Each vector has {len(query_vec)} dimensions.)")

# Embeddings are cheap, but not free. The response reports the tokens billed.
tokens = response.usage.prompt_tokens
print(f"Billed {tokens} input tokens -> "
      f"{format_cost(estimate_embedding_cost(model, tokens))}")
