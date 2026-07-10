"""Central configuration for the Father Brown's Study chat client.

Everything that depends on the environment (paths, model names, API client)
lives here so the rest of the package stays free of setup code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --- Paths -----------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
SRC_DIR = PKG_DIR.parent               # ./05_src
DATA_DIR = PKG_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"       # ChromaDB persistent store (committed)
CORPUS_TXT = DATA_DIR / "father_brown.txt"
STORIES_CSV = DATA_DIR / "stories.csv"

# --- Environment -----------------------------------------------------------
# Load from ./05_src so the app works no matter which folder it is launched from.
load_dotenv(SRC_DIR / ".env")
load_dotenv(SRC_DIR / ".secrets")

CHAT_MODEL = os.getenv("MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_NAME = "father_brown"

USE_GATEWAY = os.getenv("USE_GATEWAY", "false").lower() == "true"
GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"

# Source text used to build the corpus (Project Gutenberg #223).
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/223/pg223.txt"
BOOK_TITLE = "The Wisdom of Father Brown"
BOOK_AUTHOR = "G. K. Chesterton"


def get_client() -> OpenAI:
    """Return an OpenAI client, honouring the course API gateway if enabled.

    We build the client here (rather than importing utils.clients) because that
    helper reads USE_GATEWAY at import time, before our .env file is loaded.
    """
    if USE_GATEWAY:
        return OpenAI(
            base_url=GATEWAY_URL,
            api_key="any value",
            default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
        )
    return OpenAI()
