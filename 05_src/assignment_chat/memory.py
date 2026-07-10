"""Short-term conversation memory with automatic summarisation.

The chat must remember the conversation, but a long conversation eventually
exceeds the context window. Rather than truncating and silently losing the early
turns, this module keeps a *rolling summary*:

    [running summary of old turns] + [the last N turns, verbatim]

When the stored turns grow past `keep_last_turns`, the oldest ones are folded into
the summary by a cheap model call and then dropped. The user therefore keeps the
gist of the whole conversation while the prompt stays small and predictable.

This mirrors the "manage short-term memory" pattern referenced in the brief.
"""

from assignment_chat.config import CHAT_MODEL, get_client

# Rough token estimate: English averages ~4 characters per token.
CHARS_PER_TOKEN = 4

SUMMARY_INSTRUCTIONS = """
You compress a conversation into notes.
Rewrite the exchange below as a compact third-person summary, at most 120 words.
Keep names, topics, stories discussed, and any preference the user stated.
Drop pleasantries. Write plain sentences, no bullet points.
""".strip()


class ConversationMemory:
    """Holds the conversation and keeps it inside a token budget."""

    def __init__(
        self,
        keep_last_turns: int = 6,
        max_context_tokens: int = 3000,
        model: str = CHAT_MODEL,
    ) -> None:
        self.keep_last_turns = keep_last_turns
        self.max_context_tokens = max_context_tokens
        self.model = model
        self.summary: str = ""
        self.messages: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]
        self.summarised_turns = 0        # how many turns were folded into the summary

    # --- writing -----------------------------------------------------------
    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._compress_if_needed()

    # --- reading -----------------------------------------------------------
    def build_input(self) -> list[dict]:
        """Return the messages to send to the model."""
        history: list[dict] = []
        if self.summary:
            history.append(
                {
                    "role": "developer",
                    "content": (
                        "Summary of the earlier part of this conversation "
                        f"(for your memory only): {self.summary}"
                    ),
                }
            )
        history.extend(self.messages)
        return history

    def estimated_tokens(self) -> int:
        chars = len(self.summary) + sum(len(m["content"]) for m in self.messages)
        return chars // CHARS_PER_TOKEN

    def stats(self) -> str:
        """Human-readable state, shown in the UI so the behaviour is visible."""
        return (
            f"turns kept: {len(self.messages) // 2} | "
            f"turns summarised: {self.summarised_turns} | "
            f"~{self.estimated_tokens()} tokens"
        )

    # --- compression -------------------------------------------------------
    def _needs_compression(self) -> bool:
        too_many_turns = len(self.messages) > self.keep_last_turns * 2
        too_many_tokens = self.estimated_tokens() > self.max_context_tokens
        return too_many_turns or too_many_tokens

    def _compress_if_needed(self) -> None:
        if not self._needs_compression():
            return

        keep = self.keep_last_turns * 2
        old, recent = self.messages[:-keep], self.messages[-keep:]
        if not old:
            return

        self.summary = self._summarise(old)
        self.summarised_turns += len(old) // 2
        self.messages = recent

    def _summarise(self, old: list[dict]) -> str:
        """Fold `old` messages into the existing summary."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        previous = f"Earlier summary: {self.summary}\n\n" if self.summary else ""

        try:
            client = get_client()
            response = client.responses.create(
                model=self.model,
                instructions=SUMMARY_INSTRUCTIONS,
                input=f"{previous}New exchange to fold in:\n{transcript}",
                max_output_tokens=200,
                temperature=0.2,
            )
            return response.output_text.strip()
        except Exception:
            # Never lose the conversation because summarisation failed;
            # fall back to keeping the previous summary.
            return self.summary or "(earlier conversation could not be summarised)"
