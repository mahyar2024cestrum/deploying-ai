"""SERVICE 2 - Semantic (and hybrid) query over the story corpus.

Back end: a ChromaDB *persistent* (file-backed) store at data/chroma, built once
by build_index.py and committed to the repository. No Docker, no SQL server.

Retrieval is hybrid:

* optional lexical stage - `where_document={"$contains": keyword}` restricts the
  candidate pool to chunks that literally contain a word (an artist, a name, an
  object such as "revolver");
* semantic stage - the query is embedded and Chroma ranks the remaining chunks by
  cosine distance.

Passing only the keyword gives pure lexical filtering; passing only the query
gives pure semantic search; passing both gives the hybrid behaviour.
"""

from functools import lru_cache

import chromadb

from assignment_chat.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_client,
)


@lru_cache(maxsize=1)
def _collection():
    """Open the persistent collection once and reuse it."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No ChromaDB store at {CHROMA_DIR}. Run: python -m assignment_chat.build_index"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def _embed(text: str) -> list[float]:
    """Embed the query with the same model used to build the index."""
    client = get_client()
    return client.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding


def search_stories(query: str, top_k: int = 3, keyword: str | None = None) -> str:
    """Search the text of 'The Wisdom of Father Brown'.

    Args:
        query: what to look for, in natural language.
        top_k: how many passages to return (1-5).
        keyword: optional exact word that must appear in the passage.

    Returns:
        The matching passages, labelled with their story, as plain text.
    """
    top_k = max(1, min(int(top_k), 5))

    try:
        collection = _collection()
    except FileNotFoundError as exc:
        return str(exc)

    kwargs = {"query_embeddings": [_embed(query)], "n_results": top_k}
    if keyword:
        kwargs["where_document"] = {"$contains": keyword}

    results = collection.query(**kwargs)

    documents = results.get("documents", [[]])[0]
    if not documents:
        if keyword:
            return (
                f"No passage contains the word '{keyword}'. "
                "Try the same question without the keyword."
            )
        return "No passage in the book matches that question."

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    blocks = []
    for text, meta, distance in zip(documents, metadatas, distances):
        story = meta.get("story", "Unknown story")
        similarity = round(1 - distance, 3)
        passage = " ".join(text.split())  # collapse newlines for readability
        blocks.append(f"[From '{story}' | similarity {similarity}]\n{passage}")

    return "\n\n".join(blocks)
