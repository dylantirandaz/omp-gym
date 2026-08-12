"""Export trajectories as chat-format training data.

Two sources feed the dataset:

1. Scored episodes under a runs directory. The test reward filters
   these; episodes below the threshold are dropped.
2. Every omp session below the sessions root, past and current.
   These have no test, so no reward exists and no filter applies.
   Failed work in a session trains the model too.

Output: one JSON document per trajectory, shape {"messages": [...]}.
Tool calls are rendered as <tool_call> JSON blocks inside assistant
content. Tool results are rendered as <tool_response> blocks inside
user content. This rendering works with every chat template.
Thinking blocks are not exported. Trajectories without one
assistant step are skipped because they cannot train anything.
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
    sessions_seen: int
    sessions_exported: int
    torn_lines: int
    train_documents: int
    valid_documents: int


def _render_document(
    trajectory: Trajectory,
    prompt: str | None,
) -> list[dict[str, str]]:
    """Render one trajectory as alternating chat messages.

    Scored episodes pass their task prompt. Harvested sessions pass
    None because their first user step already is the prompt.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
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


def _trainable(trajectory: Trajectory) -> bool:
    """A trajectory trains something only with one assistant step."""
    return any(
        isinstance(step, AssistantStep) for step in trajectory.steps
    )


def _collect_episodes(
    runs_dir: Path,
    min_reward: float,
) -> tuple[list[str], int, int]:
    """Collect documents from scored episodes: (docs, seen, torn)."""
    documents: list[str] = []
    seen = 0
    torn = 0
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        seen += 1
        record = json.loads(episode_file.read_text())
        if float(record["reward"]) < min_reward:
            continue
        trajectory = parse_session(Path(record["session_file"]))
        torn += trajectory.torn_lines
        if not _trainable(trajectory):
            continue
        prompt_file = Path(record["episode_dir"]) / "prompt.txt"
        prompt = (
            prompt_file.read_text().strip()
            if prompt_file.is_file()
            else "Complete the task in this repository."
        )
        rendered = _render_document(trajectory, prompt)
        documents.append(json.dumps({"messages": rendered}))
    return documents, seen, torn


def _collect_sessions(
    sessions_root: Path,
) -> tuple[list[str], int, int]:
    """Collect documents from all omp sessions: (docs, seen, torn)."""
    documents: list[str] = []
    seen = 0
    torn = 0
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        seen += 1
        trajectory = parse_session(session_file)
        torn += trajectory.torn_lines
        if not _trainable(trajectory):
            continue
        rendered = _render_document(trajectory, prompt=None)
        documents.append(json.dumps({"messages": rendered}))
    return documents, seen, torn


def export_dataset(
    runs_dir: Path,
    sessions_root: Path,
    out_dir: Path,
    min_reward: float,
) -> ExportStats:
    """Collect both sources and write train/valid JSONL files."""
    episode_docs, episodes_seen, episode_torn = _collect_episodes(
        runs_dir, min_reward
    )
    if sessions_root.is_dir():
        session_docs, sessions_seen, session_torn = _collect_sessions(
            sessions_root
        )
    else:
        print(f"note: sessions root {sessions_root} does not exist")
        session_docs, sessions_seen, session_torn = [], 0, 0

    documents = episode_docs + session_docs
    out_dir.mkdir(parents=True, exist_ok=True)
    if not documents:
        (out_dir / "train.jsonl").write_text("")
        (out_dir / "valid.jsonl").write_text("")
        return ExportStats(
            episodes_seen=episodes_seen,
            episodes_exported=0,
            sessions_seen=sessions_seen,
            sessions_exported=0,
            torn_lines=episode_torn + session_torn,
            train_documents=0,
            valid_documents=0,
        )

    if len(documents) == 1:
        train, valid = documents, documents
        print("warning: one document only; valid set repeats the train set")
    else:
        valid_size = max(1, len(documents) // 10)
        train = documents[:-valid_size]
        valid = documents[-valid_size:]

    (out_dir / "train.jsonl").write_text("\n".join(train) + "\n")
    (out_dir / "valid.jsonl").write_text("\n".join(valid) + "\n")
    return ExportStats(
        episodes_seen=episodes_seen,
        episodes_exported=len(episode_docs),
        sessions_seen=sessions_seen,
        sessions_exported=len(session_docs),
        torn_lines=episode_torn + session_torn,
        train_documents=len(train),
        valid_documents=len(valid),
    )
