"""
Example 15: formatting output: Markdown, tables, and code blocks.

So far we've `print()`ed raw strings. But models love to answer in **Markdown** 
headings, **bold**, bullet lists, and fenced ```code``` blocks, and dumping that
raw to a terminal shows the literal `**asterisks**` and backticks. Ugly.

The `rich` library renders all of that beautifully in the terminal:
  - `Markdown(...)`  turns a Markdown string into styled text.
  - `Syntax(...)`    syntax-highlights a code block for a given language.
  - `Table(...)`     draws real bordered tables from your data.

This pairs naturally with what you've learned: ask the model for Markdown and
render it (live, even, while streaming), or take *structured* data (example 14)
and lay it out as a table. Nothing here is OpenAI-specific; it's how you make
any model's output pleasant to read.

This example needs `rich` (in requirements.txt):

    pip install rich

Run it:

    secrun python examples/15_rich_output.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY via secrun (see SECRETS.md) and try again.")

client = OpenAI()
console = Console()  # rich's entry point; console.print() understands rich objects

# --- 1. Render a Markdown answer from the model ---------------------------------
# Ask explicitly for Markdown, then hand the string to rich.Markdown so headings,
# bold, and lists render as formatting instead of literal symbols.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "In Markdown, give 3 tips for writing good commit messages. "
                "Use a heading and a bulleted list with some **bold**."
            ),
        }
    ],
)
answer = response.choices[0].message.content or ""

console.rule("[bold]1. Markdown answer")
console.print(Markdown(answer))

# --- 2. Syntax-highlight a code block -------------------------------------------
# When the answer IS code, Syntax highlights it for the given language.
snippet = '''def greet(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"
'''

console.rule("[bold]2. Code block")
console.print(Syntax(snippet, "python", theme="monokai", line_numbers=True))

# --- 3. Lay structured data out as a table --------------------------------------
# Tables shine for the structured data from example 14. Here we hard-code a few
# rows; in a real app these would come from a validated model.
console.rule("[bold]3. Table")
table = Table(title="Model line-up")
table.add_column("Model", style="cyan", no_wrap=True)
table.add_column("Good for", style="white")
table.add_column("Relative cost", justify="right", style="green")

table.add_row("gpt-4o-mini", "Everyday tasks, high volume", "$")
table.add_row("gpt-4o", "Harder reasoning, vision", "$$$")

console.print(table)
