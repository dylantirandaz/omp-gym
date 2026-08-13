"""Export trajectories as per-turn chat training samples.

Two sources feed the dataset:

1. Scored episodes under a runs directory. The test reward filters
   these; episodes below the threshold are dropped.
2. Every omp session below the sessions root, past and current.
   These have no test, so no reward exists and no filter applies.
   Failed work in a session trains the model too.

Each assistant turn becomes one sample: the system prompt, the most
recent context that fits the token budget, then the assistant turn
as the final message. The budget is measured with the trainee's own
tokenizer and chat template, so no sample can lose its completion
to truncation during training. Train with prompt masking so that
only the final assistant message produces loss.

Tool calls are rendered as <tool_call> JSON blocks inside assistant
content. Tool results are rendered as <tool_response> blocks inside
user content. Thinking blocks are not exported.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
TOKEN_SAFETY_MARGIN = 64
MESSAGE_OVERHEAD_TOKENS = 8
VALID_TRAJECTORY_SHARE = 10


class TextTokenCounter(Protocol):
    """Counts plain content tokens for one message body."""

    def __call__(self, text: str) -> int: ...


def load_token_counter(tokenizer_id: str) -> TextTokenCounter:
    """Load the trainee tokenizer and return a content token counter."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count


@dataclass(frozen=True)
class ExportStats:
    """What the export run produced."""

    episodes_seen: int
    sessions_seen: int
    trajectories_exported: int
    turns_exported: int
    turns_skipped_oversize: int
    torn_lines: int
    train_samples: int
    valid_samples: int


def _render_messages(
    trajectory: Trajectory,
    prompt: str | None,
) -> list[dict[str, str]]:
    """Render one trajectory as merged chat messages.

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


@dataclass(frozen=True)
class TurnSplit:
    """Per-turn samples of one trajectory, with skip count."""

    samples: tuple[str, ...]
    skipped_oversize: int


def _split_turns(
    messages: list[dict[str, str]],
    count_tokens: TextTokenCounter,
    token_cap: int,
) -> TurnSplit:
    """Make one sample per assistant turn with tail-window context.

    The sample keeps the system prompt, then as many of the most
    recent context messages as the token budget allows, then the
    assistant turn as the final message. Costs are measured once
    per message: content tokens plus a fixed template overhead that
    overestimates the real chat template, so a sample never loses
    its completion to truncation. A turn whose bare sample (system
    plus turn) does not fit the budget is skipped.
    """
    if not messages or messages[0]["role"] != "system":
        raise AssertionError("rendered messages must start with system")
    costs = [
        count_tokens(message["content"]) + MESSAGE_OVERHEAD_TOKENS
        for message in messages
    ]
    budget = token_cap - TOKEN_SAFETY_MARGIN
    samples: list[str] = []
    skipped = 0
    for index in range(1, len(messages)):
        turn = messages[index]
        if turn["role"] != "assistant":
            continue
        used = costs[0] + costs[index]
        if used > budget:
            skipped += 1
            continue
        start = index
        while start > 1 and used + costs[start - 1] <= budget:
            start -= 1
            used += costs[start]
        sample = [messages[0], *messages[start:index], turn]
        samples.append(json.dumps({"messages": sample}))
    return TurnSplit(samples=tuple(samples), skipped_oversize=skipped)


def _collect_episode_messages(
    runs_dir: Path,
    min_reward: float,
) -> tuple[list[list[dict[str, str]]], int, int]:
    """Render scored episodes: (rendered, seen, torn)."""
    rendered: list[list[dict[str, str]]] = []
    seen = 0
    torn = 0
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        seen += 1
        record = json.loads(episode_file.read_text())
        if float(record["reward"]) < min_reward:
            continue
        trajectory = parse_session(Path(record["session_file"]))
        torn += trajectory.torn_lines
        prompt_file = Path(record["episode_dir"]) / "prompt.txt"
        prompt = (
            prompt_file.read_text().strip()
            if prompt_file.is_file()
            else "Complete the task in this repository."
        )
        rendered.append(_render_messages(trajectory, prompt))
    return rendered, seen, torn


def _collect_session_messages(
    sessions_root: Path,
) -> tuple[list[list[dict[str, str]]], int, int]:
    """Render all omp sessions: (rendered, seen, torn)."""
    rendered: list[list[dict[str, str]]] = []
    seen = 0
    torn = 0
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        seen += 1
        trajectory = parse_session(session_file)
        torn += trajectory.torn_lines
        rendered.append(_render_messages(trajectory, prompt=None))
    return rendered, seen, torn


@dataclass(frozen=True)
class PairStats:
    """What the preference-pair export produced."""

    tasks_seen: int
    tasks_paired: int
    pairs_written: int
    pairs_skipped_oversize: int


def _episode_qualifies_as_loss(session_file: Path) -> bool:
    """A loss episode must be a real attempt, not a provider error.

    Episodes whose assistant stopped with a provider error (bad key,
    unknown model id) say nothing about the policy, so they never
    become the rejected side of a pair.
    """
    trajectory = parse_session(session_file)
    has_assistant = any(
        isinstance(step, AssistantStep) for step in trajectory.steps
    )
    if not has_assistant:
        return False
    for line in session_file.read_text().splitlines():
        if '"stopReason": "error"' in line or '"stopReason":"error"' in line:
            return False
    return True


def _first_assistant_content(
    messages: list[dict[str, str]],
) -> str | None:
    """Return the first assistant message content, if any."""
    for message in messages:
        if message["role"] == "assistant":
            return message["content"]
    return None


def export_pairs(
    runs_dir: Path,
    out_dir: Path,
    max_pairs_per_task: int,
    tokenizer_id: str,
    token_cap: int,
) -> PairStats:
    """Build DPO preference pairs from scored episodes.

    For each task with at least one win and one real loss, each
    (win, loss) combination becomes one pair: the shared prompt,
    the winner's first assistant turn as chosen, the loser's first
    assistant turn as rejected. Later-turn pairing needs aligned
    contexts, which diverging trajectories do not have. Pairs
    longer than the token cap are skipped: one outlier pair once
    padded a whole training run to 8320 tokens and stalled it.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    wins: dict[str, list[tuple[str, str]]] = {}
    losses: dict[str, list[tuple[str, str]]] = {}
    tasks_seen: set[str] = set()
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        record = json.loads(episode_file.read_text())
        task = str(record["task"])
        tasks_seen.add(task)
        session_file = Path(record["session_file"])
        trajectory = parse_session(session_file)
        prompt_file = Path(record["episode_dir"]) / "prompt.txt"
        prompt = (
            prompt_file.read_text().strip()
            if prompt_file.is_file()
            else "Complete the task in this repository."
        )
        first = _first_assistant_content(
            _render_messages(trajectory, prompt)
        )
        if first is None:
            continue
        if float(record["reward"]) >= 1.0:
            wins.setdefault(task, []).append((prompt, first))
        elif _episode_qualifies_as_loss(session_file):
            losses.setdefault(task, []).append((prompt, first))

    documents: list[str] = []
    tasks_paired = 0
    skipped_oversize = 0
    for task in sorted(tasks_seen):
        task_wins = wins.get(task, [])
        task_losses = losses.get(task, [])
        if not task_wins or not task_losses:
            continue
        tasks_paired += 1
        written = 0
        for prompt, chosen in task_wins:
            for _, rejected in task_losses:
                if written >= max_pairs_per_task:
                    break
                total_tokens = (
                    count(SYSTEM_PROMPT + prompt)
                    + max(count(chosen), count(rejected))
                )
                if total_tokens > token_cap:
                    skipped_oversize += 1
                    continue
                documents.append(
                    json.dumps(
                        {
                            "system": SYSTEM_PROMPT,
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                        }
                    )
                )
                written += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    if len(documents) > 1:
        valid_count = max(1, len(documents) // VALID_TRAJECTORY_SHARE)
        train, valid = documents[:-valid_count], documents[-valid_count:]
    else:
        train, valid = documents, documents
    (out_dir / "train.jsonl").write_text(
        "\n".join(train) + ("\n" if train else "")
    )
    (out_dir / "valid.jsonl").write_text(
        "\n".join(valid) + ("\n" if valid else "")
    )
    return PairStats(
        tasks_seen=len(tasks_seen),
        tasks_paired=tasks_paired,
        pairs_written=len(documents),
        pairs_skipped_oversize=skipped_oversize,
    )


def export_dataset(
    runs_dir: Path,
    sessions_root: Path,
    out_dir: Path,
    min_reward: float,
    tokenizer_id: str,
    token_cap: int,
) -> ExportStats:
    """Collect both sources and write per-turn train/valid files.

    The train/valid split separates whole trajectories so that no
    session contributes samples to both files. The valid set takes
    the smallest trajectories until it holds about one tenth of the
    samples.
    """
    count_tokens = load_token_counter(tokenizer_id)
    episode_msgs, episodes_seen, episode_torn = _collect_episode_messages(
        runs_dir, min_reward
    )
    if sessions_root.is_dir():
        session_msgs, sessions_seen, session_torn = (
            _collect_session_messages(sessions_root)
        )
    else:
        print(f"note: sessions root {sessions_root} does not exist")
        session_msgs, sessions_seen, session_torn = [], 0, 0

    splits = [
        _split_turns(messages, count_tokens, token_cap)
        for messages in episode_msgs + session_msgs
    ]
    splits = [split for split in splits if split.samples]
    skipped = sum(split.skipped_oversize for split in splits)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not splits:
        (out_dir / "train.jsonl").write_text("")
        (out_dir / "valid.jsonl").write_text("")
        return ExportStats(
            episodes_seen=episodes_seen,
            sessions_seen=sessions_seen,
            trajectories_exported=0,
            turns_exported=0,
            turns_skipped_oversize=skipped,
            torn_lines=episode_torn + session_torn,
            train_samples=0,
            valid_samples=0,
        )

    if len(splits) == 1:
        train_splits, valid_splits = list(splits), list(splits)
        print(
            "warning: one trajectory only; "
            "valid set repeats the train set"
        )
    else:
        total_samples = sum(len(split.samples) for split in splits)
        valid_budget = max(1, total_samples // VALID_TRAJECTORY_SHARE)
        ordered = sorted(splits, key=lambda split: len(split.samples))
        valid_splits = []
        valid_taken = 0
        for split in ordered:
            if valid_taken >= valid_budget:
                break
            valid_splits.append(split)
            valid_taken += len(split.samples)
        held_out = {id(split) for split in valid_splits}
        train_splits = [
            split for split in splits if id(split) not in held_out
        ]

    train = [sample for split in train_splits for sample in split.samples]
    valid = [sample for split in valid_splits for sample in split.samples]
    (out_dir / "train.jsonl").write_text("\n".join(train) + "\n")
    (out_dir / "valid.jsonl").write_text("\n".join(valid) + "\n")
    return ExportStats(
        episodes_seen=episodes_seen,
        sessions_seen=sessions_seen,
        trajectories_exported=len(splits),
        turns_exported=len(train) + len(valid),
        turns_skipped_oversize=skipped,
        torn_lines=episode_torn + session_torn,
        train_samples=len(train),
        valid_samples=len(valid),
    )
