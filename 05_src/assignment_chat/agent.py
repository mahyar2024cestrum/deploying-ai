"""The agent: guardrails, the function-calling loop, and memory.

Flow for every user turn:

    user message
        -> input guardrail        (restricted topics, prompt attacks)
        -> model + tools loop     (the model may call any of the three services)
        -> output guardrail       (prompt leakage, unambiguous restricted markers)
        -> memory                 (store the turn; summarise if it grew too long)

We drive the OpenAI Responses API directly rather than using LangGraph. The loop is
short and explicit, which makes it easy to insert the guardrails on both sides and
to control exactly what enters memory.
"""

import json

from assignment_chat.config import CHAT_MODEL, get_client
from assignment_chat.guardrails import check_model_output, check_user_message
from assignment_chat.memory import ConversationMemory
from assignment_chat.prompts import REFUSALS, return_instructions
from assignment_chat.tools import TOOL_SCHEMAS, run_tool

# A tool may call another tool; cap the loop so a bad plan cannot spin forever.
MAX_TOOL_ROUNDS = 5


def _extract_text(response) -> str:
    """Pull the assistant's text out of a Responses API result."""
    text = (response.output_text or "").strip()
    return text


def respond(user_message: str, memory: ConversationMemory) -> str:
    """Produce one assistant reply, applying guardrails and updating memory."""

    # 1) Input guardrail. A blocked message never reaches the model, and it is not
    #    stored in memory, so it cannot poison later turns.
    verdict = check_user_message(user_message)
    if verdict.blocked:
        return verdict.message or REFUSALS["generic"]

    client = get_client()
    memory.add_user(user_message)

    # 2) Model + tools loop.
    conversation = memory.build_input()
    reply = ""

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.responses.create(
            model=CHAT_MODEL,
            instructions=return_instructions(),
            input=conversation,
            tools=TOOL_SCHEMAS,
            temperature=0.7,
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            reply = _extract_text(response)
            break

        # Feed the model's own tool-call items back in, then append each result.
        conversation = conversation + list(response.output)
        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            # `keyword: null` is valid for our schema but not a Python kwarg default.
            arguments = {k: v for k, v in arguments.items() if v is not None}

            result = run_tool(call.name, arguments)
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
            )
    else:
        reply = "I have turned this over too many times, my friend. Ask me again, more simply?"

    if not reply:
        reply = REFUSALS["generic"]

    # 3) Output guardrail. If the model leaked the prompt or drifted onto a
    #    restricted topic, replace the answer entirely.
    verdict = check_model_output(reply)
    if verdict.blocked:
        reply = verdict.message or REFUSALS["generic"]

    # 4) Remember the turn (this may trigger summarisation).
    memory.add_assistant(reply)
    return reply
