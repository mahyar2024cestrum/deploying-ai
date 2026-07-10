"""SERVICE 1 - API calls.

Back end: the Open Library search API (https://openlibrary.org/search.json), a
free, key-less public API covering millions of books.

Why Open Library? The first choice was Gutendex (the Project Gutenberg catalogue),
but it timed out repeatedly during testing. Open Library answers in well under a
second, needs no API key, and covers the same question ("what else can I read?").

The brief requires that the API response is NOT passed back verbatim. Two steps
guarantee that:

1. The raw JSON is reduced to a handful of clean fields (we discard the dozens of
   fields the reader does not care about).
2. Those fields are rewritten into a short prose paragraph by a separate, cheap
   model call. The chat model therefore only ever sees prose, never the payload.
"""

import requests

from assignment_chat.config import CHAT_MODEL, get_client

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
FIELDS = "title,author_name,first_publish_year,subject"
TIMEOUT = 15
RETRIES = 2

REPHRASE_INSTRUCTIONS = """
You turn structured catalogue records into flowing prose.
Write 2-4 sentences describing the books listed. Mention titles, authors and the
year naturally. Do not use bullet points, JSON, field names, or numbered lists.
Write plainly and factually; the caller will add their own style.
Only describe books that appear in the records. Invent nothing.
""".strip()


def _fetch(query: str, max_results: int) -> list[dict]:
    """Call Open Library and reduce each record to the few fields we care about."""
    params = {"q": query, "limit": max_results, "fields": FIELDS}

    last_error: Exception | None = None
    for _ in range(RETRIES):
        try:
            response = requests.get(OPEN_LIBRARY_URL, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
    else:
        raise last_error  # type: ignore[misc]

    books = []
    for item in response.json().get("docs", [])[:max_results]:
        books.append(
            {
                "title": item.get("title", "Unknown"),
                "authors": item.get("author_name") or ["Unknown"],
                "year": item.get("first_publish_year"),
                "subjects": (item.get("subject") or [])[:3],
            }
        )
    return books


def _to_prose(query: str, books: list[dict]) -> str:
    """Rewrite the structured records as prose (the non-verbatim requirement)."""
    lines = []
    for book in books:
        year = book["year"] or "year unknown"
        subjects = ", ".join(book["subjects"]) or "unspecified"
        lines.append(
            f"- {book['title']} by {', '.join(book['authors'])}, first published {year}; "
            f"themes: {subjects}"
        )
    facts = "\n".join(lines)

    client = get_client()
    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=REPHRASE_INSTRUCTIONS,
        input=f"Search topic: {query}\n\nRecords:\n{facts}",
        max_output_tokens=250,
        temperature=0.4,
    )
    prose = response.output_text.strip()

    # If the rewrite comes back empty, fall back to a plain sentence rather than
    # dumping the structured records on the user.
    if not prose:
        titles = "; ".join(f"{b['title']} by {', '.join(b['authors'])}" for b in books)
        return f"The catalogue lists: {titles}."
    return prose


def search_book_catalogue(query: str, max_results: int = 3) -> str:
    """Search the Open Library catalogue and describe the matches in prose.

    Args:
        query: an author, title, or subject to search for.
        max_results: how many books to describe (1-5).

    Returns:
        A short prose paragraph. Never raw JSON.
    """
    max_results = max(1, min(int(max_results), 5))

    try:
        books = _fetch(query, max_results)
    except requests.RequestException:
        return (
            "The library catalogue cannot be reached at the moment. "
            "Say so plainly and offer to search the stories instead."
        )

    if not books:
        return f"The catalogue holds nothing matching '{query}'."

    return _to_prose(query, books)
