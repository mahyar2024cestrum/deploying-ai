"""One-time build script: download the corpus, chunk it, embed it, persist it.

The graders do NOT need to run this. Its outputs are committed to the repo:

    data/father_brown.txt   the cleaned corpus
    data/stories.csv        per-story metadata (used by the analytics service)
    data/chroma/            a ChromaDB persistent store holding the embeddings

Run it only if you want to rebuild the index from scratch:

    cd 05_src
    python -m assignment_chat.build_index
"""

import csv
import re
import shutil

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from assignment_chat.config import (
    BOOK_AUTHOR,
    BOOK_TITLE,
    CHROMA_DIR,
    COLLECTION_NAME,
    CORPUS_TXT,
    DATA_DIR,
    EMBEDDING_MODEL,
    GUTENBERG_URL,
    STORIES_CSV,
    get_client,
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_BATCH = 100

# Story headings look like "ONE -- The Absence of Mr Glass" or "TWO. -- The Paradise of Thieves".
HEADING = re.compile(
    r"^(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE)\.?\s+--\s+(.+)$",
    re.MULTILINE,
)
WORD_NUMBERS = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
                "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE"]


def download_corpus() -> str:
    """Fetch the book and strip the Project Gutenberg header/footer."""
    if CORPUS_TXT.exists():
        print(f"Using cached corpus: {CORPUS_TXT}")
        return CORPUS_TXT.read_text(encoding="utf-8")

    print(f"Downloading {BOOK_TITLE}...")
    text = requests.get(GUTENBERG_URL, timeout=60).text

    start = text.index("*** START")
    start = text.index("\n", start) + 1
    end = text.index("*** END")
    body = text[start:end].replace("\r\n", "\n").strip()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_TXT.write_text(body, encoding="utf-8")
    print(f"Saved corpus: {CORPUS_TXT} ({len(body):,} chars)")
    return body


def split_into_stories(body: str) -> list[dict]:
    """Split the book body into its twelve stories."""
    matches = list(HEADING.finditer(body))
    if len(matches) != 12:
        raise RuntimeError(f"Expected 12 story headings, found {len(matches)}")

    stories = []
    for i, match in enumerate(matches):
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        story_text = body[text_start:text_end].strip()
        stories.append(
            {
                "number": WORD_NUMBERS.index(match.group(1)) + 1,
                "title": match.group(2).strip(),
                "text": story_text,
            }
        )
    return stories


def write_stories_csv(stories: list[dict]) -> None:
    """Structured metadata used by the analytics service (pandas reads this)."""
    rows = []
    for story in stories:
        words = len(story["text"].split())
        rows.append(
            {
                "number": story["number"],
                "title": story["title"],
                "book": BOOK_TITLE,
                "author": BOOK_AUTHOR,
                "word_count": words,
                "char_count": len(story["text"]),
                "paragraph_count": len([p for p in story["text"].split("\n\n") if p.strip()]),
                # 238 words per minute is a common adult silent-reading estimate.
                "reading_minutes": round(words / 238, 1),
            }
        )

    with STORIES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {STORIES_CSV} ({len(rows)} stories)")


def chunk_stories(stories: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    chunks = []
    for story in stories:
        for idx, piece in enumerate(splitter.split_text(story["text"])):
            chunks.append(
                {
                    "id": f"s{story['number']:02d}_c{idx:03d}",
                    "text": piece,
                    "metadata": {
                        "story": story["title"],
                        "story_number": story["number"],
                        "chunk_index": idx,
                    },
                }
            )
    print(f"Created {len(chunks)} chunks from {len(stories)} stories")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Embed every chunk, in batches, using the embeddings API."""
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        response = client.embeddings.create(
            input=[c["text"] for c in batch],
            model=EMBEDDING_MODEL,
        )
        vectors.extend(item.embedding for item in response.data)
        print(f"  embedded {min(start + EMBED_BATCH, len(chunks))}/{len(chunks)}")
    return vectors


def persist_to_chroma(chunks: list[dict], vectors: list[list[float]]) -> None:
    """Write the embeddings into a ChromaDB persistent (file-backed) store."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Cosine space: OpenAI embeddings are direction-based, and it makes the
    # returned distance fall in [0, 2] so that similarity = 1 - distance.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma has a per-call batch limit, so add in slices.
    for start in range(0, len(chunks), 500):
        batch = chunks[start:start + 500]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=vectors[start:start + 500],
        )
    print(f"Persisted {collection.count()} chunks to {CHROMA_DIR}")


def main() -> None:
    body = download_corpus()
    stories = split_into_stories(body)
    write_stories_csv(stories)
    chunks = chunk_stories(stories)
    vectors = embed_chunks(chunks)
    persist_to_chroma(chunks, vectors)
    print("Index build complete.")


if __name__ == "__main__":
    main()
