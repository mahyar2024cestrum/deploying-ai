"""Tool definitions (JSON schemas) and the dispatch table.

Each of the three services is exposed to the model as a callable function, using
the OpenAI Responses API function-calling format.
"""

from assignment_chat.service_analytics import list_stories, story_statistics
from assignment_chat.service_books_api import search_book_catalogue
from assignment_chat.service_semantic import search_stories

TOOL_SCHEMAS = [
    # --- Service 1: public API ---------------------------------------------
    {
        "type": "function",
        "name": "search_book_catalogue",
        "description": (
            "Search a public library catalogue for real books by author, title, or "
            "subject. Use when the user asks what else they could read, or about an "
            "author or book outside this collection. Returns a prose description."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "An author, title, or subject, e.g. 'Chesterton' or 'detective'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many books to describe, 1 to 5.",
                },
            },
            "required": ["query", "max_results"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    # --- Service 2: semantic / hybrid search --------------------------------
    {
        "type": "function",
        "name": "search_stories",
        "description": (
            "Search the full text of 'The Wisdom of Father Brown' for passages that "
            "answer a question about plot, characters, clues, or events. Always use "
            "this before describing what happens in a story. Optionally pass a keyword "
            "that must literally appear in the passage (hybrid search)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or idea to search for, in natural language.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return, 1 to 5.",
                },
                "keyword": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional exact word that must appear in the passage, such as a "
                        "character name. Pass null to search purely by meaning."
                    ),
                },
            },
            "required": ["query", "top_k", "keyword"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    # --- Service 3: function calling over structured data -------------------
    {
        "type": "function",
        "name": "list_stories",
        "description": (
            "List all twelve stories in the book with their word count and reading time. "
            "Use when the user asks what stories exist or which is longest/shortest."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "story_statistics",
        "description": (
            "Exact figures for one story, a comparison of two stories, or totals for the "
            "whole book. Use for any question about length, word count, paragraphs, or "
            "reading time. Never estimate these numbers yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Story title. Pass an empty string for whole-book totals.",
                },
                "compare_with": {
                    "type": "string",
                    "description": "Optional second story title to compare against. Empty string if unused.",
                },
            },
            "required": ["title", "compare_with"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# Maps the schema name to the Python callable that implements it.
TOOL_FUNCTIONS = {
    "search_book_catalogue": search_book_catalogue,
    "search_stories": search_stories,
    "list_stories": list_stories,
    "story_statistics": story_statistics,
}


def run_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name, converting any failure into a readable message."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Unknown tool: {name}"
    try:
        return str(function(**arguments))
    except Exception as exc:  # keep the conversation alive if a tool misbehaves
        return f"The tool '{name}' failed: {exc.__class__.__name__}: {exc}"
