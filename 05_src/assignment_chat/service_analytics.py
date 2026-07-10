"""SERVICE 3 - Function calling over structured data.

Back end: data/stories.csv, read with pandas. No SQLite, as the brief requires.

This service exists because language models are unreliable at exact arithmetic and
at recalling precise figures. Instead of letting the model guess how long a story
is, we expose deterministic functions it can call. This is the "capability
extension" pattern from the course: the tool computes, the model narrates.
"""

from functools import lru_cache

import pandas as pd

from assignment_chat.config import STORIES_CSV


@lru_cache(maxsize=1)
def _stories() -> pd.DataFrame:
    if not STORIES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {STORIES_CSV}. Run: python -m assignment_chat.build_index"
        )
    return pd.read_csv(STORIES_CSV)


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _find(frame: pd.DataFrame, title: str) -> pd.Series | None:
    """Case-insensitive, partial title match."""
    needle = title.strip().lower()
    exact = frame[frame["title"].str.lower() == needle]
    if not exact.empty:
        return exact.iloc[0]
    partial = frame[frame["title"].str.lower().str.contains(needle, regex=False)]
    if not partial.empty:
        return partial.iloc[0]
    return None


def list_stories() -> str:
    """List the twelve stories with their length and reading time."""
    try:
        frame = _stories()
    except FileNotFoundError as exc:
        return str(exc)

    lines = [
        f"{row.number}. {row.title} - {row.word_count:,} words, "
        f"about {row.reading_minutes} minutes to read"
        for row in frame.itertuples()
    ]
    return "\n".join(lines)


def story_statistics(title: str = "", compare_with: str = "") -> str:
    """Exact figures for one story, or a comparison between two.

    Args:
        title: the story to describe. Leave empty for figures about the whole book.
        compare_with: an optional second story to compare against the first.
    """
    try:
        frame = _stories()
    except FileNotFoundError as exc:
        return str(exc)

    # Whole-book summary.
    if not title.strip():
        total_words = int(frame["word_count"].sum())
        longest = frame.loc[frame["word_count"].idxmax()]
        shortest = frame.loc[frame["word_count"].idxmin()]
        return (
            f"The book holds {len(frame)} stories and {total_words:,} words in total, "
            f"about {round(frame['reading_minutes'].sum())} minutes of reading. "
            f"The longest is '{longest.title}' ({longest.word_count:,} words); "
            f"the shortest is '{shortest.title}' ({shortest.word_count:,} words). "
            f"The average story runs {int(frame['word_count'].mean()):,} words."
        )

    first = _find(frame, title)
    if first is None:
        return f"There is no story called '{title}' in this book. Use list_stories to see them all."

    # Single-story figures.
    if not compare_with.strip():
        rank = int((frame["word_count"] > first.word_count).sum()) + 1
        return (
            f"'{first.title}' is story number {first.number} of {len(frame)}. "
            f"It runs {first.word_count:,} words across {first.paragraph_count} paragraphs "
            f"({first.char_count:,} characters), and takes about {first.reading_minutes} "
            f"minutes to read. It is the {_ordinal(rank)} longest story in the book."
        )

    # Comparison.
    second = _find(frame, compare_with)
    if second is None:
        return f"There is no story called '{compare_with}' in this book."

    difference = int(abs(first.word_count - second.word_count))
    longer, shorter = (first, second) if first.word_count >= second.word_count else (second, first)
    ratio = round(longer.word_count / max(shorter.word_count, 1), 2)
    return (
        f"'{first.title}' runs {first.word_count:,} words ({first.reading_minutes} minutes); "
        f"'{second.title}' runs {second.word_count:,} words ({second.reading_minutes} minutes). "
        f"'{longer.title}' is longer by {difference:,} words, {ratio} times the length of "
        f"'{shorter.title}'."
    )
