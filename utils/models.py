"""
utils/models.py: turn off hidden reasoning, but only for models that allow it.

gpt-6-luna (this repo's default) thinks before it answers unless you tell it not
to, and while it's thinking it rejects `temperature`, `top_p`, and function tools
on chat completions. Sending `reasoning_effort="none"` makes it behave like a
plain chat model again.

The catch is that "none" isn't universal. Probed against the live API on
2026-10-03:

  accepts "none"   gpt-5.4-mini, gpt-5.5, gpt-5.6-luna, gpt-6-sol, gpt-6-luna
  rejects "none"   gpt-5-nano (it predates "none"), gpt-6-astra, gpt-6.1-sol
  no such param    gpt-4o, gpt-4o-mini (400: "Unrecognized request argument")

So a script that lets you pick the model with `--model` or `OPENAI_MODEL` can't
just hard-code it. These helpers return the right keyword arguments for the model
you picked, or nothing at all.
"""

_ACCEPTS_NONE = ("gpt-6-luna", "gpt-6-sol")


def accepts_none(model: str) -> bool:
    """True if `model` takes reasoning effort "none"."""
    return model.startswith("gpt-5.") or model in _ACCEPTS_NONE


def reasoning_off(model: str) -> dict:
    """Keyword arguments for `chat.completions.create` that turn reasoning off."""
    return {"reasoning_effort": "none"} if accepts_none(model) else {}


def responses_reasoning_off(model: str) -> dict:
    """The same thing for `responses.create`, where effort lives in an object."""
    return {"reasoning": {"effort": "none"}} if accepts_none(model) else {}


if __name__ == "__main__":
    for m in ["gpt-6-luna", "gpt-5.4-nano", "gpt-5-nano", "gpt-6-astra", "gpt-4o-mini"]:
        print(f"{m:14} -> {reasoning_off(m)}")
