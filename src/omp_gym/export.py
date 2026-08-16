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

Tool calls are rendered as bare JSON objects on their own lines in
assistant content: the serving shim parses that envelope, and small
models cannot round-trip the <tool_call> special token through the
server decode. Tool results are rendered as <tool_response> blocks
inside user content. Tool-step prose and thinking blocks are not
exported.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .mint import _CORRECTION
from .trajectory import (
    AssistantStep,
    ToolCall,
    ToolResultStep,
    Trajectory,
    UserStep,
    parse_session,
)

SYSTEM_PROMPT = (
    "You are a coding agent. Work in the current repository through tools.\n"
    "Write one JSON object on its own line for each tool call:\n"
    '{"name": "read", "arguments": {"path": "relative path", "i": "purpose"}}\n'
    'bash arguments: {"command": "command", "i": "purpose"}\n'
    'edit arguments: {"input": "hashline patch", "i": "purpose"}\n'
    'write arguments: {"path": "relative path", "content": "full file", '
    '"i": "purpose"}\n'
    'grep arguments: {"pattern": "regex", "path": "relative path", '
    '"i": "purpose"}\n'
    'glob arguments: {"path": "glob pattern", "i": "purpose"}\n'
    "The environment answers with <tool_response>. Use relative paths. "
    "Read source and tests before the change. Run the failing tests. "
    "For a small source file, use write with the full file content. "
    "Inspect each result. Run the tests again. Do not report success until they pass."
)
TASK_PROMPT_PREFIX = "Complete the task in this repository."

_PATCH_FILE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\*\*\* Begin Patch\s*\n)?\[([^#\]\r\n]+)#[0-9A-Fa-f]{4}\]"
)
TOOL_RESULT_LIMIT = 4000
_WORKSPACE_PATH_PATTERN = re.compile(r"/(?:[^/\s'\":]+/)*ws(?=/|[\s'\":]|$)")
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
    sessions_filtered: int
    trajectories_exported: int
    turns_exported: int
    turns_skipped_oversize: int
    torn_lines: int
    train_samples: int
    valid_samples: int


def _elide_middle(text: str, limit: int) -> str:
    """Truncate long text by keeping the head and the tail.

    Tool output puts the setup at the top and the error or result
    at the bottom. Cutting only the tail loses the part that
    matters for failures, so the middle goes instead.
    """
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    marker = f"\n...[elided {len(text) - limit} chars]...\n"
    return text[:head] + marker + text[len(text) - tail :]


def _normalize_workspace_paths(text: str) -> str:
    """Replace one saved episode workspace root with the current root."""
    return _WORKSPACE_PATH_PATTERN.sub(".", text)


def _normalize_tool_arguments(
    arguments: dict[str, object],
) -> dict[str, object]:
    """Replace saved episode workspace paths with relative paths."""
    normalized = dict(arguments)
    for name, value in arguments.items():
        if not isinstance(value, str):
            continue
        relative_value = _normalize_workspace_paths(value)
        if name == "path" and relative_value.startswith("./"):
            relative_value = relative_value[2:]
        normalized[name] = relative_value
    return normalized


def _canonical_training_call(
    call: ToolCall,
    authoring_workspace: Path | None,
) -> tuple[str, dict[str, object]]:
    """Render successful edits as simple full-file writes."""
    arguments = _normalize_tool_arguments(call.arguments)
    if call.name != "edit" or authoring_workspace is None:
        return call.name, arguments
    path_argument = arguments.get("path")
    if isinstance(path_argument, str):
        relative_path = path_argument
    else:
        patch = arguments.get("input")
        if not isinstance(patch, str):
            return call.name, arguments
        found = _PATCH_FILE_PATTERN.search(patch)
        if found is None:
            return call.name, arguments
        relative_path = found.group(1)
    if Path(relative_path).is_absolute():
        return call.name, arguments
    workspace = authoring_workspace.resolve()
    final_file = (workspace / relative_path).resolve()
    if not final_file.is_relative_to(workspace) or not final_file.is_file():
        return call.name, arguments
    intent = arguments.get("i")
    return (
        "write",
        {
            "path": relative_path,
            "content": final_file.read_text(),
            "i": intent
            if isinstance(intent, str) and intent
            else "Write verified file",
        },
    )


_AUTHORING_TOOL_NAMES = frozenset({"edit", "write"})


def _render_messages(
    trajectory: Trajectory,
    prompt: str | None,
    authoring_workspace: Path | None = None,
) -> list[dict[str, str]]:
    """Render one trajectory as merged chat messages.

    Scored episodes pass their task prompt. Harvested sessions pass
    None because their first user step already is the prompt.
    """
    failed_authoring_call_ids = frozenset(
        step.call_id
        for step in trajectory.steps
        if isinstance(step, ToolResultStep)
        and step.is_error
        and step.tool_name in _AUTHORING_TOOL_NAMES
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    for step in trajectory.steps:
        match step:
            case AssistantStep():
                calls = tuple(
                    call
                    for call in step.tool_calls
                    if call.call_id not in failed_authoring_call_ids
                )
                parts = [] if step.tool_calls else ([step.text] if step.text else [])
                for call in calls:
                    name, arguments = _canonical_training_call(
                        call,
                        authoring_workspace,
                    )
                    payload = json.dumps(
                        {
                            "name": name,
                            "arguments": arguments,
                        }
                    )
                    parts.append(payload)
                content = "\n".join(parts)
                if content:
                    messages.append({"role": "assistant", "content": content})
            case ToolResultStep():
                if step.call_id in failed_authoring_call_ids:
                    continue
                body = _elide_middle(
                    _normalize_workspace_paths(step.text),
                    TOOL_RESULT_LIMIT,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_response>\n{body}\n</tool_response>",
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
    plus turn) does not fit the budget is kept with its middle
    elided, and counted as truncated.
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
        turn_cost = costs[index]
        if costs[0] + turn_cost > budget:
            turn_budget = budget - costs[0] - MESSAGE_OVERHEAD_TOKENS
            turn = dict(turn)
            turn["content"] = _elide_middle(turn["content"], max(turn_budget * 3, 64))
            turn_cost = count_tokens(turn["content"]) + MESSAGE_OVERHEAD_TOKENS
            skipped += 1
        used = costs[0] + turn_cost
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
            else TASK_PROMPT_PREFIX
        )
        rendered.append(
            _render_messages(trajectory, prompt, episode_file.parent / "ws")
        )
    return rendered, seen, torn


def _session_failed(session_file: Path, trajectory: Trajectory) -> bool:
    """True when a session ended as a failure, not a success.

    Three signals: a provider error ended the session, the last
    tool result is an error with no successful call after it, or
    the user corrected the agent and no work followed the last
    correction. Sessions without any of these pass the filter.
    """
    raw = session_file.read_text()
    if '"stopReason": "error"' in raw:
        return True
    last_tool_error_at = -1
    last_assistant_at = -1
    corrections_before = 0
    for index, step in enumerate(trajectory.steps):
        if isinstance(step, ToolResultStep) and step.is_error:
            last_tool_error_at = index
        elif isinstance(step, AssistantStep) and step.tool_calls:
            last_assistant_at = index
        elif isinstance(step, UserStep):
            if _CORRECTION.search(step.text):
                corrections_before += 1
    if corrections_before and last_assistant_at == -1:
        return True
    if (
        corrections_before
        and last_tool_error_at > last_assistant_at
        and last_tool_error_at != -1
    ):
        return True
    return False


def _collect_session_messages(
    sessions_root: Path,
    min_quality: bool,
) -> tuple[list[list[dict[str, str]]], int, int, int]:
    """Render all omp sessions: (rendered, seen, filtered, torn).

    With min_quality on, sessions that ended as failures do not
    enter the SFT dataset. They remain available for DPO pairs.
    """
    rendered: list[list[dict[str, str]]] = []
    seen = 0
    filtered = 0
    torn = 0
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        seen += 1
        trajectory = parse_session(session_file)
        torn += trajectory.torn_lines
        if min_quality and _session_failed(session_file, trajectory):
            filtered += 1
            continue
        rendered.append(_render_messages(trajectory, prompt=None))
    return rendered, seen, filtered, torn


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
    has_assistant = any(isinstance(step, AssistantStep) for step in trajectory.steps)
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
        first = _first_assistant_content(_render_messages(trajectory, prompt))
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
                total_tokens = count(SYSTEM_PROMPT + prompt) + max(
                    count(chosen), count(rejected)
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
    (out_dir / "train.jsonl").write_text("\n".join(train) + ("\n" if train else ""))
    (out_dir / "valid.jsonl").write_text("\n".join(valid) + ("\n" if valid else ""))
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
    min_quality: bool = True,
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
        session_msgs, sessions_seen, sessions_filtered, session_torn = (
            _collect_session_messages(sessions_root, min_quality)
        )
    else:
        print(f"note: sessions root {sessions_root} does not exist")
        session_msgs, sessions_seen = [], 0
        sessions_filtered, session_torn = 0, 0

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
            sessions_filtered=sessions_filtered,
            trajectories_exported=0,
            turns_exported=0,
            turns_skipped_oversize=skipped,
            torn_lines=episode_torn + session_torn,
            train_samples=0,
            valid_samples=0,
        )

    if len(splits) == 1:
        train_splits, valid_splits = list(splits), list(splits)
        print("warning: one trajectory only; valid set repeats the train set")
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
        train_splits = [split for split in splits if id(split) not in held_out]

    train = [sample for split in train_splits for sample in split.samples]
    valid = [sample for split in valid_splits for sample in split.samples]
    (out_dir / "train.jsonl").write_text("\n".join(train) + "\n")
    (out_dir / "valid.jsonl").write_text("\n".join(valid) + "\n")
    return ExportStats(
        episodes_seen=episodes_seen,
        sessions_seen=sessions_seen,
        sessions_filtered=sessions_filtered,
        trajectories_exported=len(splits),
        turns_exported=len(train) + len(valid),
        turns_skipped_oversize=skipped,
        torn_lines=episode_torn + session_torn,
        train_samples=len(train),
        valid_samples=len(valid),
    )
