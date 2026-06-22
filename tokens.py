"""
Token counting with tiktoken.
==============================

Models don't see characters or words — they see *tokens*. A token is a chunk of
text (often a word fragment). Roughly, 1 token ≈ 4 characters of English, or
about ¾ of a word, but it varies. The only way to know for sure is to run the
tokenizer, which is exactly what `tiktoken` does — locally, with no API call.

Why count tokens?
  1. Cost: you pay per token (see pricing.py).
  2. Limits: every model has a maximum *context window* (input + output). If you
     overflow it, the request fails.
  3. Intuition: watching the count change as you edit a prompt teaches you a lot
     about how the model reads your text.

Two functions live here:
  - count_tokens(text):          tokens in a raw string.
  - count_message_tokens(msgs):  tokens in a full chat-format message list,
                                 including the small per-message bookkeeping
                                 overhead the API adds.
"""

import tiktoken


def _encoding_for(model: str) -> tiktoken.Encoding:
    """Get the right tokenizer for a model.

    Different model families use different tokenizers (gpt-4o uses `o200k_base`,
    older gpt-4 / gpt-3.5 use `cl100k_base`). `encoding_for_model` knows the
    mapping; we fall back to o200k_base for brand-new models tiktoken hasn't
    learned about yet.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in a plain string."""
    return len(_encoding_for(model).encode(text))


def count_message_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Count tokens for a chat-completions `messages` list.

    The API doesn't just concatenate your text — it wraps each message in a few
    special tokens (to mark the role, the start/end of the message, etc.). This
    function adds that overhead so the number lines up with what you'll actually
    be billed for on the input side.

    The exact overhead is an implementation detail that has shifted slightly
    between model generations; the values below match OpenAI's published
    "how to count tokens" cookbook for current chat models. Treat the result as
    a close estimate, and trust the `usage` field in the real API response for
    the authoritative number.
    """
    enc = _encoding_for(model)

    tokens_per_message = 3  # every message carries <|start|>role/name\n...<|end|>
    tokens_per_name = 1     # if a message includes a "name" field

    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(enc.encode(str(value)))
            if key == "name":
                total += tokens_per_name
    total += 3  # every reply is primed with <|start|>assistant
    return total


if __name__ == "__main__":
    # Run `python tokens.py` to see tokenization in action.
    sample = "The quick brown fox jumps over the lazy dog."
    enc = _encoding_for("gpt-4o-mini")
    ids = enc.encode(sample)
    print(f"Text:   {sample!r}")
    print(f"Tokens: {len(ids)}")
    print("Pieces:", [enc.decode([i]) for i in ids])
