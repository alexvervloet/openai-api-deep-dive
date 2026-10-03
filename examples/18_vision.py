"""
Example 18: vision: sending an image alongside text.

Everything so far sent only text. Multimodal models also accept *images* in the
same message, so you can ask "what's in this picture?", read a screenshot, or
pull data out of a photo of a receipt. The request shape barely changes: instead
of a plain string, the user message's `content` becomes a *list of parts*, where
each part is either `{"type": "text", ...}` or `{"type": "image_url", ...}`.

Two ways to provide the image, both shown here:
  - a public URL the model can fetch, or
  - a local file you read and inline as a base64 `data:` URI (no public hosting
    needed, which is what real apps usually do).

Images are billed as tokens too, and the count scales with the image's pixel
dimensions: a big screenshot can cost more than a page of text. Downscale before
sending if you care about cost.

Run it (uses a public sample image):

    secrun python examples/18_vision.py

    # or point it at your own local image (sent as base64):
    secrun python examples/18_vision.py path/to/image.png
"""

import base64
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see ../docs/SECRETS.md) and try again.")

client = OpenAI()

# A public sample image: a tabby cat on a stone wall, a Wikimedia Commons
# featured picture. The 1,795px original is big on purpose; with detail "high"
# it costs about 3,000 tokens. (The boardwalk photo this used to point at, the
# one in OpenAI's docs, was deleted from Commons, and the URL now returns 400.)
SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"


def image_part_from_path(path: str) -> dict:
    """Read a local image and build a base64 `data:` URI image part.

    This is the form most apps use: the bytes travel inside the request, so the
    image never needs to be publicly hosted. We guess the MIME type from the
    extension; the API accepts png, jpeg, gif, and webp.
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg"}.get(ext, ext)  # .jpg -> image/jpeg
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}


# Build the image part: a local file if one was passed, else the public URL.
if len(sys.argv) > 1:
    path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(f"No such file: {path}")
    image_part = image_part_from_path(path)
    print(f"[sending local image as base64: {path}]\n")
else:
    # The URL form: the model's server fetches the image itself.
    image_part = {"type": "image_url", "image_url": {"url": SAMPLE_URL}}
    print("[sending a public sample image by URL]\n")

# The one new idea: `content` is a LIST of parts (text + image), not a string.
#
# `detail` decides how much of the image the model sees, and what you pay. Luna
# bills about 1.2 tokens per 32x32-pixel patch. Measured on 2026-10-03:
#   "low"   at most 307 tokens, whatever the size (19 for a 100x100 icon).
#   "high"  shrinks big images to fit, so at most 3,000 tokens.
#   "auto"  the default if you leave it out. On luna it does NOT shrink: a
#           3000x3000 photo cost 10,603 tokens, and very large images get a 400.
#           gpt-5.4-nano treated "auto" like "high" and charged 3,000 for the
#           same photo, so the cheaper model is the pricier one for big images.
# Set `detail` on purpose rather than inheriting a default that changed.
response = client.chat.completions.create(
    model="gpt-6-luna",
    reasoning_effort="none",  # see example 01
    messages=[
        {  # type: ignore[arg-type]
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in two sentences. What stands out?"},
                {**image_part, "image_url": {**image_part["image_url"], "detail": "high"}},
            ],
        }
    ],
)

print(response.choices[0].message.content)
assert response.usage is not None
print(f"\n[tokens: prompt: {response.usage.prompt_tokens}, "
      f"completion: {response.usage.completion_tokens}]")
print("Notice the prompt tokens: the image itself is most of them. Bigger image = more tokens.")
