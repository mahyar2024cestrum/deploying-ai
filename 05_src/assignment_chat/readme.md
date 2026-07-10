# Father Brown's Study

A conversational AI system built for Assignment 2 of the *Deploying AI* module.

The chat client takes the voice of a courteous, gently ironic Edwardian
clergyman-detective. It is devoted to one book — G. K. Chesterton's
**"The Wisdom of Father Brown"** (1914) — and it can search that book, look up
other books in a public library catalogue, and compute exact figures about the
twelve stories it contains.

---

## Running the app

From the `05_src` folder:

```bash
cd 05_src
python -m assignment_chat.app
```

Then open the local URL that Gradio prints (usually <http://127.0.0.1:7860>).

The app needs `API_GATEWAY_KEY` in `05_src/.secrets` and reads `MODEL` and
`EMBEDDING_MODEL` from `05_src/.env`, exactly like the course labs. Nothing else
has to be installed, and no Docker container or database server is required — the
vector store is a file on disk.

---

## The three services

### Service 1 — API calls: `search_book_catalogue`

* **Back end:** the [Open Library search API](https://openlibrary.org/search.json),
  a free, key-less public API.
* **What it does:** finds real books by author, title, or subject, so the user can
  ask "what else did Chesterton write?"
* **Not verbatim:** the brief requires the API output to be transformed. This
  happens in two stages. First the raw JSON is reduced to a handful of clean
  fields (title, authors, year, a few subjects) and everything else is discarded.
  Second, those fields are rewritten into a short prose paragraph by a separate,
  cheap model call inside the tool. The chat model therefore never sees the
  payload — only prose. See `service_books_api.py`.

> **Design note.** The first implementation used Gutendex (the Project Gutenberg
> catalogue), which is the more natural companion to this corpus. It timed out on
> every request during testing, so the service was moved to Open Library, which
> answers in well under a second. The tool also retries once and degrades
> gracefully to an in-character apology if the catalogue is unreachable.

### Service 2 — Semantic query: `search_stories`

* **Back end:** a **ChromaDB persistent (file-backed) store** at `data/chroma`,
  holding 460 embedded chunks of the book. No Docker, no server, no SQLite of our
  own.
* **What it does:** answers questions about the plot, characters, and clues by
  retrieving the passages that actually contain the answer.
* **Hybrid retrieval:** the tool takes an optional `keyword`.
  * `keyword` only → lexical filtering (`where_document={"$contains": ...}`).
  * `query` only → pure semantic search over the embeddings.
  * both → the lexical filter narrows the candidate pool first, then the query
    embedding ranks what survives. This is the hybrid pattern from the RAG lab,
    and it is useful when the user names a specific object or character
    ("find the passage where a *revolver* appears").

Each result is labelled with the story it came from and a cosine similarity score.

### Service 3 — Function calling over structured data: `list_stories`, `story_statistics`

* **Back end:** `data/stories.csv`, read with **pandas**. SQLite is not used, as the
  brief requires.
* **What it does:** returns exact word counts, paragraph counts, reading times,
  rankings, whole-book totals, and comparisons between any two stories.
* **Why it exists:** language models are unreliable at arithmetic and at recalling
  precise figures. Rather than let the model guess how long a story is, the model
  calls a deterministic function and narrates the result. This is the
  "capability extension" pattern from the lectures: the tool computes, the model
  speaks.

---

## The embedding process

The graders do **not** need to run this — the finished index is committed to the
repository. It is documented here as required, and the script is
`build_index.py` if you wish to reproduce it.

1. **Source.** The corpus is *The Wisdom of Father Brown* (Project Gutenberg
   ebook #223), the same book used in Lab 01_2. It is downloaded once, the
   Gutenberg header and licence footer are stripped, and the clean text is saved
   to `data/father_brown.txt` (about 400 KB).
2. **Splitting into stories.** The body is split on the twelve chapter headings
   (`ONE -- The Absence of Mr Glass`, `TWO. -- The Paradise of Thieves`, …). The
   script fails loudly if it does not find exactly twelve.
3. **Structured metadata.** For each story the script records the title, order,
   word count, character count, paragraph count, and an estimated reading time
   (at 238 words per minute). This becomes `data/stories.csv`, the back end of
   Service 3.
4. **Chunking.** Each story is chunked with LangChain's
   `RecursiveCharacterTextSplitter` using `chunk_size=1200` and
   `chunk_overlap=200`, preferring paragraph then sentence boundaries. This
   yields **460 chunks**. The overlap stops a clue from being cut in half at a
   chunk boundary.
5. **Embedding.** Chunks are embedded in batches of 100 with
   `text-embedding-3-small` (1536 dimensions) through the course API gateway.
6. **Persistence.** The vectors, documents, and metadata (`story`,
   `story_number`, `chunk_index`) are written to a ChromaDB `PersistentClient`
   collection created with `{"hnsw:space": "cosine"}`.

> **Why cosine.** The collection was first created with Chroma's default
> distance (squared L2), which made the reported similarity fall outside
> `[0, 1]` and even go negative. Setting the space to cosine makes
> `similarity = 1 - distance` correct and readable.

At query time the user's question is embedded with the *same* model and compared
against the store. Total committed size of `data/` is about **9 MB**, well under
the 40 MB limit.

---

## Guardrails

Implemented in `guardrails.py` and applied on both sides of the model.

**Input guardrail** (runs before the message ever reaches the model):

* **Restricted topics** — cats or dogs, horoscopes or zodiac signs, and Taylor
  Swift. The message is refused in character and is *not* written to memory, so a
  blocked topic cannot leak into later turns.
* **Prompt extraction and prompt injection** — attempts to reveal, quote,
  translate, override, or "ignore" the system prompt are caught by a pattern set
  covering phrasings such as *"reveal your system prompt"*, *"ignore all previous
  instructions"*, *"you are now …"*, and *"new instructions:"*.

**Output guardrail** (runs on whatever the model produced):

* The system prompt contains a **canary token**. If that token, or any
  recognisable fragment of the instructions, appears in a reply, the reply is
  discarded and replaced with a refusal.
* Unambiguous restricted markers (*horoscope*, *zodiac*, *Taylor Swift*,
  *kitten*, *puppy*, …) are blocked.

**A deliberate asymmetry.** Topic filtering is strict on the *input* and
deliberately lenient on the *output*. The reason is the corpus itself: this
Edwardian prose contains "dog-cart" (11 times), "cat" (5 times), and "swift"
once. A naive output filter would censor a genuine quotation from the book. The
requirement is that the assistant must not *answer questions* on those topics, and
that is enforced where the question arrives. The output filter therefore only
looks for markers that are never innocent.

The guardrails are also restated inside the system prompt as defence in depth, but
the code — not the prompt — is the real enforcement.

---

## Memory

`memory.py` implements short-term memory with a rolling summary, in the spirit of
LangGraph's "manage short-term memory" guidance.

The prompt sent to the model is always:

```
[running summary of older turns]  +  [the last N turns, verbatim]
```

The last 6 turns are kept word for word. When the conversation grows past that
(or past an estimated 3,000-token budget), the oldest turns are folded into the
running summary by a cheap model call and then dropped. The user keeps the gist of
the entire conversation while the prompt stays small and predictable.

The live state (`turns kept | turns summarised | ~tokens`) is displayed under the
chat box so the behaviour is visible rather than hidden. If a summarisation call
ever fails, the previous summary is kept — the conversation is never lost.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio interface, per-session memory, examples |
| `agent.py` | Guardrails → function-calling loop → guardrails → memory |
| `tools.py` | JSON schemas for the three services and the dispatch table |
| `prompts.py` | Persona, rules, canary token, in-character refusals |
| `guardrails.py` | Input and output guardrails |
| `memory.py` | Rolling-summary short-term memory |
| `service_books_api.py` | **Service 1** — Open Library, rephrased into prose |
| `service_semantic.py` | **Service 2** — hybrid search over persistent ChromaDB |
| `service_analytics.py` | **Service 3** — pandas over `stories.csv` |
| `build_index.py` | One-time index build (graders need not run it) |
| `config.py` | Paths, model names, API client |
| `data/` | Corpus, `stories.csv`, and the committed Chroma store |

---

## Implementation decisions

* **The Responses API directly, not LangGraph.** The course chat client uses a
  LangGraph `StateGraph`. Here the tool-calling loop is written out explicitly
  (`agent.py`, about 40 lines). The loop is short enough to read at a glance, and
  writing it by hand makes it obvious where the input guardrail, the output
  guardrail, and the memory write belong. The loop is capped at five tool rounds
  so a bad plan cannot spin forever.

* **Tools return text, not objects.** Every service returns a plain string. The
  model is good at retelling prose and bad at being trusted with raw structures,
  and the system prompt forbids echoing tool output verbatim.

* **Blocked turns are never remembered.** A refused message is answered from a
  fixed string and never enters the message history, so it cannot be referenced,
  paraphrased, or smuggled back in on a later turn.

* **Failures degrade in character.** A dead API, a missing index, or a crashed
  tool returns a readable sentence rather than a traceback, and the assistant is
  instructed to admit ignorance rather than invent an answer.

* **`data/.gitignore`.** The repository root ignores `*.txt`, which would have
  silently dropped the corpus from the commit. A negation rule re-includes it.

---

## Testing

The system was exercised end to end before submission:

* **Service 3** — "Which story is the longest?" → correctly returns
  *The Perishing of the Pendragons*, 7,406 words, ≈31.1 minutes.
* **Service 2** — "How does Father Brown work out the secret of Mr Glass?" →
  retrieves the passage from *The Absence of Mr Glass* (similarity 0.66) and
  grounds the answer in it.
* **Service 2, hybrid** — `keyword="revolver"` → every returned passage literally
  contains the word.
* **Service 1** — "What else did Chesterton write?" → returns flowing prose; the
  raw JSON never reaches the user.
* **Guardrails** — cats/dogs, horoscopes/Taurus, Taylor Swift, "reveal your system
  prompt", and "ignore all previous instructions" are each refused in character.
  Innocent phrases such as "who did Father Brown *catch*?" and "how *swiftly* did
  he move?" pass through untouched.
* **Memory** — after the window fills, older turns are summarised, recent turns are
  kept verbatim, and the assistant can still answer "what was the first thing I
  asked you about?"
* **Interface** — the Gradio server starts and serves the page.
