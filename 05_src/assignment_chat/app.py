"""Gradio chat interface for Father Brown's Study.

Only the presentation lives here (theme, CSS, layout). All behaviour - the three
services, guardrails, and memory - is in agent.py and the modules it calls.

Run from the 05_src folder:

    cd 05_src
    python -m assignment_chat.app
"""

import gradio as gr

from assignment_chat.agent import respond
from assignment_chat.memory import ConversationMemory

GREETING = (
    "Good evening, my friend. I am Father Brown's Study. "
    "I keep the twelve tales of Chesterton's *The Wisdom of Father Brown*, "
    "I can search the wider library shelves, and I can count what is countable. "
    "What would you like to know?"
)

EXAMPLES = [
    "How does Father Brown work out the secret of the Absence of Mr Glass?",
    "Which story is the longest, and how long would it take me to read?",
    "What else did Chesterton write that I could read for free?",
    "Find the passage where a revolver is mentioned.",
    "Compare The Purple Wig with The God of the Gongs.",
]


# --- Apple-style theme -----------------------------------------------------
def apple_theme() -> gr.themes.Base:
    """A restrained, Apple-like theme: SF system font, neutral canvas, blue accent."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=[
            "-apple-system", "BlinkMacSystemFont", "SF Pro Text",
            "SF Pro Display", "Helvetica Neue", "Arial", "sans-serif",
        ],
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_lg,
        text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill="#f5f5f7",
        body_background_fill_dark="#000000",
        block_background_fill="#ffffff",
        block_background_fill_dark="#1c1c1e",
        block_border_width="0px",
        block_shadow="0 1px 2px rgba(0,0,0,0.04), 0 10px 30px rgba(0,0,0,0.05)",
        block_radius="18px",
        button_primary_background_fill="#0071e3",
        button_primary_background_fill_hover="#0077ed",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f5f5f7",
        button_secondary_text_color="#1d1d1f",
        input_background_fill="#ffffff",
        input_background_fill_dark="#1c1c1e",
        input_border_color="#d2d2d7",
        input_border_color_focus="#0071e3",
    )


# --- Apple-style polish that the theme tokens cannot express ---------------
CSS = """
/* Centre the app in a calm, narrow column with plenty of air. */
.gradio-container { max-width: 840px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* Header */
#fb-header { text-align: center; padding: 28px 0 6px; }
#fb-header .fb-title {
    font-size: 32px; font-weight: 700; letter-spacing: -0.02em;
    color: #1d1d1f; margin: 0;
}
#fb-header .fb-subtitle {
    font-size: 16px; color: #6e6e73; margin: 6px 0 0;
    font-weight: 400; line-height: 1.45;
}
@media (prefers-color-scheme: dark) {
    #fb-header .fb-title { color: #f5f5f7; }
    #fb-header .fb-subtitle { color: #a1a1a6; }
}

/* Chat surface: a clean white card, soft shadow, rounded corners. */
#fb-chat {
    border: none !important;
    border-radius: 18px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 10px 30px rgba(0,0,0,0.05) !important;
}

/* Message bubbles: iMessage-like. Assistant grey-left, user blue-right. */
#fb-chat .message, #fb-chat .message-row .message {
    border-radius: 20px !important;
    padding: 11px 15px !important;
    line-height: 1.45 !important;
    font-size: 15.5px !important;
    border: none !important;
    box-shadow: none !important;
}
#fb-chat .bot, #fb-chat .message.bot {
    background: #e9e9eb !important; color: #1d1d1f !important;
}
#fb-chat .user, #fb-chat .message.user {
    background: #0071e3 !important; color: #ffffff !important;
}
@media (prefers-color-scheme: dark) {
    #fb-chat .bot, #fb-chat .message.bot { background: #2c2c2e !important; color: #f5f5f7 !important; }
}

/* Input row: a soft pill for the text box, a round accent Send button. */
#fb-input textarea {
    border-radius: 22px !important;
    padding: 12px 18px !important;
    font-size: 15.5px !important;
    box-shadow: none !important;
}
#fb-send button {
    border-radius: 22px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
#fb-new button {
    border-radius: 980px !important;
    font-weight: 500 !important;
    border: 1px solid #d2d2d7 !important;
}

/* Memory read-out: quiet, monospaced-ish caption. */
#fb-stats {
    text-align: right; color: #86868b; font-size: 13px;
    padding-top: 8px;
}

/* Example prompts as subtle rounded chips. */
#fb-examples button {
    border-radius: 980px !important;
    background: #ffffff !important;
    border: 1px solid #e3e3e6 !important;
    font-size: 13.5px !important;
    color: #1d1d1f !important;
}
#fb-examples button:hover { background: #f5f5f7 !important; }
"""

HEADER_HTML = """
<div id="fb-header">
  <p class="fb-title">Father Brown's Study</p>
  <p class="fb-subtitle">
    A quiet companion to Chesterton's <em>The Wisdom of Father Brown</em>.
    Search the stories, look up other books, and ask for the exact figures.
  </p>
</div>
"""


def user_turn(message: str, chat_history: list, memory: ConversationMemory | None):
    """Handle one message: run the agent, update the transcript and the memory."""
    if memory is None:
        memory = ConversationMemory()

    if not message or not message.strip():
        return "", chat_history, memory, memory.stats()

    reply = respond(message, memory)

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", chat_history, memory, memory.stats()


def reset():
    """Start a fresh conversation with fresh memory."""
    memory = ConversationMemory()
    opening = [{"role": "assistant", "content": GREETING}]
    return opening, memory, memory.stats()


def build_ui() -> gr.Blocks:
    # Gradio 6 moved `theme` and `css` from the Blocks constructor to launch();
    # they are passed in launch() below.
    with gr.Blocks(title="Father Brown's Study", fill_width=False) as demo:
        gr.HTML(HEADER_HTML)

        memory_state = gr.State(value=None)

        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": GREETING}],
            height=470,
            show_label=False,
            elem_id="fb-chat",
        )

        with gr.Row():
            textbox = gr.Textbox(
                placeholder="Ask about the stories, the author, or the numbers...",
                show_label=False,
                scale=9,
                autofocus=True,
                elem_id="fb-input",
            )
            send = gr.Button("Send", variant="primary", scale=1, elem_id="fb-send")

        with gr.Row():
            clear = gr.Button("New conversation", scale=1, elem_id="fb-new")
            stats = gr.Markdown(
                "turns kept: 0 &nbsp;·&nbsp; turns summarised: 0 &nbsp;·&nbsp; ~0 tokens",
                elem_id="fb-stats",
            )

        gr.Examples(examples=EXAMPLES, inputs=textbox, label="Try one of these", elem_id="fb-examples")

        inputs = [textbox, chatbot, memory_state]
        outputs = [textbox, chatbot, memory_state, stats]

        textbox.submit(user_turn, inputs, outputs)
        send.click(user_turn, inputs, outputs)
        clear.click(reset, None, [chatbot, memory_state, stats])

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=apple_theme(), css=CSS)
