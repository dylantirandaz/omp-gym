"""Export scored trajectories as chat-format training data.

Output: one JSON document per episode, shape {"messages": [...]}.
Tool calls are rendered as <tool_call> JSON blocks inside assistant
content. Tool results are rendered as <tool_response> blocks inside
user content. This rendering works with every chat template.
Thinking blocks are not exported.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .trajectory import (
    AssistantStep,
    ToolResultStep,
    Trajectory,
    UserStep,
    parse_session,
)

SYSTEM_PROMPT = (
    "You are a coding agent. You work in a repository through tools.\n"
    "To call a tool, write a <tool_call> block that contains one JSON\n"
    'object: {"name": ..., "arguments": {...}}. The environment answers\n'
    "with a <tool_response> block. Available tools: read, bash, edit,\n"
    "write, grep, glob. Work until the task is complete."
)

TOOL_RESULT_LIMIT = 4000


@dataclass(frozen=True)
class ExportStats:
    """What the export run produced."""

    episodes_seen: int
    episodes_exported: int
    train_documents: int
    valid_documents: int


def _render_steps(trajectory: Trajectory, prompt: str) -> list[dict[str, str]]:
    """Render trajectory steps as alternating chat messages."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for step in trajectory.steps:
        match step:
            case AssistantStep():
                parts = [step.text] if step.text else []
                for call in step.tool_calls:
                    payload = json.dumps(
                        {"name": call.name, "arguments": call.arguments}
                    )
                    parts.append(f"<tool_call>\n{payload}\n</tool_call>")
                content = "\n".join(parts)
                if content:
                    messages.append(
                        {"role": "assistant", "content": content}
                    )
            case ToolResultStep():
                body = step.text[:TOOL_RESULT_LIMIT]
                status = "error" if step.is_error else "ok"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response status={status}>\n"
                            f"{body}\n</tool_response>"
                        ),
                    }
                )
            case UserStep():
                if step.text:
                    messages.append({"role": "user", "content": step.text})

    merged: list[dict[str, str]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {
                "role": message["role"],
                "content": merged[-1]["content"] + "\n\n" + message["content"],
            }
        else:
            merged.append(message)
    return merged


def export_dataset(
    runs_dir: Path,
    out_dir: Path,
    min_reward: float,
) -> ExportStats:
    """Collect scored episodes and write train/valid JSONL files."""
    documents: list[str] = []
    episodes_seen = 0
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        episodes_seen += 1
        record = json.loads(episode_file.read_text())
        if float(record["reward"]) < min_reward:
            continue
        trajectory = parse_session(Path(record["session_file"]))
        prompt_file = Path(record["episode_dir"]) / "prompt.txt"
        prompt = (
            prompt_file.read_text().strip()
            if prompt_file.is_file()
            else "Complete the task in this repository."
        )
        rendered = _render_steps(trajectory, prompt)
        documents.append(json.dumps({"messages": rendered}))

    out_dir.mkdir(parents=True, exist_ok=True)
    if not documents:
        (out_dir / "train.jsonl").write_text("")
        (out_dir / "valid.jsonl").write_text("")
        return ExportStats(episodes_seen, 0, 0, 0)

    if len(documents) == 1:
        train, valid = documents, documents
        print("warning: one episode only; valid set repeats the train set")
    else:
        train, valid = documents[:-1], documents[-1:]

    (out_dir / "train.jsonl").write_text("\n".join(train) + "\n")
    (out_dir / "valid.jsonl").write_text("\n".join(valid) + "\n")
    return ExportStats(
        episodes_seen=episodes_seen,
        episodes_exported=len(documents),
        train_documents=len(train),
        valid_documents=len(valid),
    )
