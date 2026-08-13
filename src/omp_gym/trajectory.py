"""Parse an omp session JSONL file into a trajectory.

The session file is the source of truth for what the agent did.
Entry shape, verified against omp v17.2.15 session files:
  {"type": "message", "message": {"role": ..., "content": [...]}}
Assistant content blocks: "thinking", "text", "toolCall".
Tool results are messages with role "toolResult".
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation by the agent."""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AssistantStep:
    """One assistant turn: visible text plus tool calls.

    thinking holds the model's thinking blocks when the session
    recorded them. It is empty for models without thinking output.
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    thinking: str = ""


@dataclass(frozen=True)
class ToolResultStep:
    """The environment's answer to one tool call."""

    call_id: str
    tool_name: str
    text: str
    is_error: bool


@dataclass(frozen=True)
class UserStep:
    """One user turn."""

    text: str


Step = AssistantStep | ToolResultStep | UserStep


@dataclass(frozen=True)
class Trajectory:
    """The ordered steps of one session.

    torn_lines counts lines that were not valid JSON. A live session
    file can end with one torn line while omp still writes to it.
    """

    steps: tuple[Step, ...]
    torn_lines: int


def _block_texts(content: object) -> str:
    """Join the text of all text blocks in a content value."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _parse_assistant(message: dict[str, object]) -> AssistantStep:
    content = message.get("content")
    calls: list[ToolCall] = []
    texts: list[str] = []
    thinking_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                texts.append(str(block.get("text", "")))
            elif block_type == "thinking":
                thinking_parts.append(str(block.get("thinking", "")))
            elif block_type == "toolCall":
                arguments = block.get("arguments")
                calls.append(
                    ToolCall(
                        call_id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=arguments
                        if isinstance(arguments, dict)
                        else {},
                    )
                )
    return AssistantStep(
        text="\n".join(texts),
        tool_calls=tuple(calls),
        thinking="\n".join(thinking_parts),
    )


def parse_session(session_file: Path) -> Trajectory:
    """Read one session file and return its steps in order."""
    steps: list[Step] = []
    torn_lines = 0
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            torn_lines += 1
            continue
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            steps.append(_parse_assistant(message))
        elif role == "toolResult":
            steps.append(
                ToolResultStep(
                    call_id=str(message.get("toolCallId", "")),
                    tool_name=str(message.get("toolName", "")),
                    text=_block_texts(message.get("content")),
                    is_error=bool(message.get("isError", False)),
                )
            )
        elif role == "user":
            steps.append(UserStep(text=_block_texts(message.get("content"))))
    return Trajectory(steps=tuple(steps), torn_lines=torn_lines)
