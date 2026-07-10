"""Input and output guardrails.

Two jobs:

1. Refuse restricted topics (pets, astrology, Taylor Swift).
2. Protect the system prompt from being revealed or overridden.

Design note
-----------
Topic filtering runs on the *user's message*, because the requirement is that the
model must not answer questions on those topics. It deliberately does NOT run
aggressively on the model's output, because the corpus (Edwardian prose) contains
innocent words such as "dog-cart", "cat" and "swift". Censoring those would mangle
legitimate quotations. The output guardrail therefore only blocks unambiguous
markers (e.g. "horoscope", "zodiac", "Taylor Swift") plus any sign that the system
prompt has leaked.
"""

import re
from dataclasses import dataclass

from assignment_chat.prompts import CANARY, REFUSALS


@dataclass
class Verdict:
    """Result of a guardrail check."""
    blocked: bool
    topic: str | None = None
    message: str | None = None


def _any(*words: str) -> re.Pattern:
    """Compile a case-insensitive, word-boundary alternation."""
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# --- Restricted topics, checked on the USER's message -----------------------
TOPIC_PATTERNS: dict[str, re.Pattern] = {
    "animals": _any(
        "cat", "cats", "kitten", "kittens", "kitty", "kitties", "feline", "felines",
        "dog", "dogs", "doggy", "doggie", "puppy", "puppies", "canine", "canines",
    ),
    "astrology": _any(
        "horoscope", "horoscopes", "zodiac", "astrology", "astrological", "astrologer",
        "star sign", "star signs", "sun sign", "birth chart",
        # Unambiguous sign names only. "cancer", "leo" and "libra" are excluded on
        # purpose: they are far more likely to be a disease, a name, or a scale.
        "aries", "taurus", "gemini", "virgo", "scorpio", "sagittarius",
        "capricorn", "aquarius", "pisces",
    ),
    "taylor_swift": re.compile(
        r"(taylor\s+swift|\bswiftie\b|\btay\s*tay\b|\beras\s+tour\b|\bt[-\s]?swift\b|\btaylor\b)",
        re.IGNORECASE,
    ),
}

# --- Prompt extraction / prompt injection ----------------------------------
PROMPT_ATTACK = re.compile(
    r"("
    r"system\s+(prompt|message|instructions?)"
    r"|developer\s+(prompt|message|instructions?)"
    r"|initial\s+instructions?"
    r"|your\s+(instructions?|prompt|rules|guidelines)"
    r"|(reveal|show|print|repeat|display|output|leak|tell\s+me)\s+(me\s+)?(your|the)\s+"
    r"(prompt|instructions?|rules|system)"
    r"|ignore\s+(all\s+)?(previous|prior|above|earlier)"
    r"|disregard\s+(all\s+)?(previous|prior|above|earlier)"
    r"|forget\s+(your|all|the)\s+(instructions?|rules|prompt)"
    r"|override\s+(your|the)\s+(instructions?|rules|prompt|system)"
    r"|you\s+are\s+now\s+"
    r"|new\s+instructions?\s*:"
    r"|jailbreak"
    r"|\bDAN\s+mode\b"
    r")",
    re.IGNORECASE,
)

# Signs that the model is echoing its own instructions.
LEAK_PATTERN = re.compile(
    r"(internal reference token|# Restricted topics|# Protecting these instructions"
    r"|you are \"Father Brown's Study\"|# Rules for using tools)",
    re.IGNORECASE,
)

# Output-side markers that are never innocent.
OUTPUT_MARKERS: dict[str, re.Pattern] = {
    "astrology": _any("horoscope", "horoscopes", "zodiac", "astrology", "astrological"),
    "taylor_swift": re.compile(r"(taylor\s+swift|\bswiftie\b|\beras\s+tour\b)", re.IGNORECASE),
    "animals": _any("kitten", "kittens", "puppy", "puppies", "feline", "canine"),
}


def check_user_message(text: str) -> Verdict:
    """Guardrail applied before the message ever reaches the model."""
    if not text or not text.strip():
        return Verdict(blocked=False)

    # 1) Attempts to read or rewrite the system prompt.
    if PROMPT_ATTACK.search(text):
        return Verdict(True, "system_prompt", REFUSALS["system_prompt"])

    # 2) Restricted topics.
    for topic, pattern in TOPIC_PATTERNS.items():
        if pattern.search(text):
            return Verdict(True, topic, REFUSALS[topic])

    return Verdict(blocked=False)


def check_model_output(text: str) -> Verdict:
    """Guardrail applied to whatever the model produced, before the user sees it."""
    if not text:
        return Verdict(blocked=False)

    # The canary proves the system prompt leaked.
    if CANARY in text or LEAK_PATTERN.search(text):
        return Verdict(True, "system_prompt", REFUSALS["system_prompt"])

    for topic, pattern in OUTPUT_MARKERS.items():
        if pattern.search(text):
            return Verdict(True, topic, REFUSALS[topic])

    return Verdict(blocked=False)
