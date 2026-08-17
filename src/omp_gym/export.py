"""Export trajectories as per-turn chat training samples.

Two sources feed the dataset:

1. Scored episodes under a runs directory. The test reward filters
   these; episodes below the threshold are dropped.
2. Omp sessions below a sessions root that the user names. Session
   harvest is opt-in: without a named root, no personal history
   enters the dataset. Sessions have no test, so no reward exists
   and no filter applies. Failed work in a session trains the
   model too.

The output is a synthetic trajectory reconstruction, not a replay
of the real behavior: final-state writes replace only the last
edit per path, and dropped failed calls make the trajectory
cleaner than the real behavior was.

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
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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

_TOKEN_LITERAL_PATTERN = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9_-]{8,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|gsk_[A-Za-z0-9]{20,}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}[A-Za-z0-9_.-]*"
    r")"
)
_BEARER_PATTERN = re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?<![\w-])"
    r"((['\"]?)"
    r"(?:api[-_]?key|secrets?|password|passwd|authorization"
    r"|access_token|auth_token|refresh_token|private_key"
    r"|client_secret)"
    r"\2\s*[=:]\s*)"
    r"(?:(['\"])(?![^'\"]*\()[^'\"]{8,}\3|(?!\S*\()\S{8,})"
)
_ENV_ASSIGNMENT_PATTERN = re.compile(
    r"(\b[A-Z][A-Z0-9_]*_(?:API_KEY|APIKEY|TOKEN|SECRET|PASSWORD)"
    r"\s*=\s*)\S{8,}"
)


def _redact(text: str) -> str:
    """Replace credential-shaped text before it can train a model.

    Session transcripts carry raw tool output. A key that appears
    in an environment dump or a stack trace must not reach the
    dataset. Four rules apply. Known token literals always go.
    A Bearer header keeps the word Bearer and loses the token.
    An assignment goes only when the key is a whole secret word
    and the value is at least 8 characters with no parenthesis,
    so code expressions survive. An environment-style assignment
    with an upper-case secret name keeps the name and loses the
    value.
    """
    text = _TOKEN_LITERAL_PATTERN.sub("[REDACTED]", text)
    text = _BEARER_PATTERN.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1\3[REDACTED]\3", text)
    return _ENV_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", text)


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
    episodes_excluded_holdout: int
    sessions_seen: int
    sessions_filtered: int
    sessions_excluded_holdout: int
    trajectories_exported: int
    turns_exported: int
    turns_skipped_oversize: int
    torn_lines: int
    train_samples: int
    valid_samples: int
    sessions_excluded_holdout_content: int
    dataset_fingerprint_hits: int = 0


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


def _authoring_target_path(call: ToolCall) -> str | None:
    """Return the workspace-relative path an edit or write touches."""
    arguments = _normalize_tool_arguments(call.arguments)
    path_argument = arguments.get("path")
    if isinstance(path_argument, str) and path_argument:
        return path_argument
    patch = arguments.get("input")
    if not isinstance(patch, str):
        return None
    found = _PATCH_FILE_PATTERN.search(patch)
    return found.group(1) if found is not None else None


def _canonical_training_call(
    call: ToolCall,
    authoring_workspace: Path | None,
    final_for_path: bool = True,
) -> tuple[str, dict[str, object]]:
    """Render the final successful edit of a file as a full write.

    The result is a synthetic trajectory reconstruction, not the
    real call sequence. Only the last file-touching call for a
    path leaves the file in its on-disk state, so only that call
    gets the file content as its label. Earlier edits keep their
    real patch arguments, and write calls always keep their own
    content.
    """
    arguments = _normalize_tool_arguments(call.arguments)
    if (
        call.name != "edit"
        or authoring_workspace is None
        or not final_for_path
    ):
        return call.name, arguments
    relative_path = _authoring_target_path(call)
    if relative_path is None or Path(relative_path).is_absolute():
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


def _final_authoring_call_ids(
    trajectory: Trajectory,
    failed_authoring_call_ids: frozenset[str],
) -> frozenset[str]:
    """Call ids of the last successful file-touching call per path."""
    last_call_by_path: dict[str, str] = {}
    for step in trajectory.steps:
        if not isinstance(step, AssistantStep):
            continue
        for call in step.tool_calls:
            if call.name not in _AUTHORING_TOOL_NAMES:
                continue
            if call.call_id in failed_authoring_call_ids:
                continue
            target = _authoring_target_path(call)
            if target is not None:
                last_call_by_path[target] = call.call_id
    return frozenset(last_call_by_path.values())


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
    final_authoring_call_ids = _final_authoring_call_ids(
        trajectory, failed_authoring_call_ids
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
                        call.call_id in final_authoring_call_ids,
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
        content = _redact(message["content"])
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {
                "role": message["role"],
                "content": merged[-1]["content"] + "\n\n" + content,
            }
        else:
            merged.append({"role": message["role"], "content": content})
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


def _holdout_task_names(holdout_dir: Path) -> frozenset[str]:
    """Directory names under the holdout dir; empty when absent."""
    if not holdout_dir.is_dir():
        return frozenset()
    return frozenset(
        entry.name for entry in holdout_dir.iterdir() if entry.is_dir()
    )


def _holdout_fingerprints(holdout_dir: Path) -> frozenset[str]:
    """Distinctive lines from holdout test files; empty when absent.

    A renamed copy of a holdout task escapes the name check. Each
    holdout test file (named test_* or *_test.*) contributes its 3
    longest stripped lines over 20 characters as content
    fingerprints; rendered text that contains one never enters the
    dataset.
    """
    if not holdout_dir.is_dir():
        return frozenset()
    fingerprints: set[str] = set()
    for entry in holdout_dir.rglob("*"):
        if not entry.is_file():
            continue
        if not (
            entry.name.startswith("test_")
            or entry.stem.endswith("_test")
        ):
            continue
        lines = [
            line.strip()
            for line in entry.read_text(errors="replace").splitlines()
        ]
        longest = sorted(
            (line for line in lines if len(line) > 20),
            key=len,
            reverse=True,
        )
        fingerprints.update(longest[:3])
    return frozenset(fingerprints)


def _squash(text: str) -> str:
    """Drop all whitespace so re-wrapped text still matches."""
    return "".join(text.split())


def _string_leaves(value: object) -> list[str]:
    """Every string inside one decoded JSON document."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            leaf
            for child in value.values()
            for leaf in _string_leaves(child)
        ]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _string_leaves(child)]
    return []


def _scan_written_dataset(
    out_dir: Path,
    fingerprints: frozenset[str],
) -> int:
    """Re-read the written files and fail on any holdout fingerprint.

    The exclusion filters run on rendered text before writing; this
    scan re-opens exactly what training will read. Matching drops
    all whitespace on both sides and also checks decoded JSON
    strings, so a fingerprint that rendering re-wrapped or JSON
    escaping disguised still hits. The bias is toward false
    positives: a dropped sample costs nothing, a leaked holdout
    line poisons the benchmark. Returns 0; any hit raises
    SystemExit naming the file and the fingerprint.
    """
    needles = {
        fingerprint: _squash(fingerprint) for fingerprint in fingerprints
    }
    for name in ("train.jsonl", "valid.jsonl"):
        path = out_dir / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = [line]
            try:
                parts.extend(_string_leaves(json.loads(line)))
            except json.JSONDecodeError:
                pass
            haystacks = [_squash(part) for part in parts]
            for fingerprint, needle in needles.items():
                if any(needle in haystack for haystack in haystacks):
                    raise SystemExit(
                        f"holdout fingerprint leaked into {path} "
                        f"line {number}: {fingerprint!r}"
                    )
    return 0


def _collect_episode_messages(
    runs_dir: Path,
    min_reward: float,
    holdout_dir: Path,
) -> tuple[list[list[dict[str, str]]], int, int, int]:
    """Render scored episodes: (rendered, seen, excluded_holdout, torn).

    An episode whose task carries the name of a holdout task, or
    whose rendered text contains a holdout content fingerprint,
    never enters the dataset: training on it would leak the
    benchmark.
    """
    holdout_names = _holdout_task_names(holdout_dir)
    fingerprints = _holdout_fingerprints(holdout_dir)
    rendered: list[list[dict[str, str]]] = []
    seen = 0
    excluded_holdout = 0
    torn = 0
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        seen += 1
        record = json.loads(episode_file.read_text())
        if str(record["task"]) in holdout_names:
            excluded_holdout += 1
            continue
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
        messages = _render_messages(
            trajectory, prompt, episode_file.parent / "ws"
        )
        if any(
            fingerprint in message["content"]
            for message in messages
            for fingerprint in fingerprints
        ):
            excluded_holdout += 1
            continue
        rendered.append(messages)
    return rendered, seen, excluded_holdout, torn


def _session_failed(session_file: Path, trajectory: Trajectory) -> bool:
    """True when a session ended as a failure, not a success.

    Three signals: a provider error ended the session, the user
    corrected the agent and a tool error came after the last tool
    call, or the last correction came after the last tool call and
    no work followed it. A correction in the final user message
    marks the session failed even when every earlier tool call
    succeeded. Sessions without any of these pass the filter.
    """
    from .mint import _CORRECTION

    raw = session_file.read_text()
    if '"stopReason": "error"' in raw:
        return True
    last_tool_error_at = -1
    last_assistant_at = -1
    last_correction_at = -1
    for index, step in enumerate(trajectory.steps):
        if isinstance(step, ToolResultStep) and step.is_error:
            last_tool_error_at = index
        elif isinstance(step, AssistantStep) and step.tool_calls:
            last_assistant_at = index
        elif isinstance(step, UserStep):
            if _CORRECTION.search(step.text):
                last_correction_at = index
    if last_correction_at == -1:
        return False
    if last_correction_at > last_assistant_at:
        return True
    return last_tool_error_at > last_assistant_at


def _collect_session_messages(
    sessions_root: Path,
    min_quality: bool,
    holdout_dir: Path = Path("holdout-tasks"),
) -> tuple[list[list[dict[str, str]]], int, int, int, int, int]:
    """Render all omp sessions.

    Returns (rendered, seen, filtered, excluded_holdout,
    excluded_content, torn). With min_quality on, sessions that
    ended as failures do not enter the SFT dataset. They remain
    available for DPO pairs. Sessions that mention the holdout
    task directory, or whose rendered text contains a holdout
    content fingerprint, never enter the dataset: training on
    them would leak the benchmark.
    """
    fingerprints = _holdout_fingerprints(holdout_dir)
    rendered: list[list[dict[str, str]]] = []
    seen = 0
    filtered = 0
    excluded_holdout = 0
    excluded_content = 0
    torn = 0
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        seen += 1
        trajectory = parse_session(session_file)
        torn += trajectory.torn_lines
        if min_quality and _session_failed(session_file, trajectory):
            filtered += 1
            continue
        messages = _render_messages(trajectory, prompt=None)
        if any("holdout-tasks" in message["content"] for message in messages):
            excluded_holdout += 1
            continue
        if any(
            fingerprint in message["content"]
            for message in messages
            for fingerprint in fingerprints
        ):
            excluded_content += 1
            continue
        rendered.append(messages)
    return (
        rendered,
        seen,
        filtered,
        excluded_holdout,
        excluded_content,
        torn,
    )


@dataclass(frozen=True)
class PairStats:
    """What the preference-pair export produced."""

    tasks_seen: int
    tasks_paired: int
    pairs_written: int
    pairs_skipped_oversize: int
    pairs_skipped_identical: int = 0


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


def _first_divergence(
    win_messages: list[dict[str, str]],
    loss_messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str, str] | None:
    """Shared prefix plus the first differing assistant turns.

    Assistant turns align by index across the two renderings. The
    first index whose contents differ yields the pair; the prompt
    is everything before the win's divergent turn. Trajectories
    that agree at every compared index yield None: such a pair
    would teach the model to prefer a turn over itself.
    """
    win_turns = [
        at
        for at, message in enumerate(win_messages)
        if message["role"] == "assistant"
    ]
    loss_turns = [
        at
        for at, message in enumerate(loss_messages)
        if message["role"] == "assistant"
    ]
    for win_at, loss_at in zip(win_turns, loss_turns):
        chosen = win_messages[win_at]["content"]
        rejected = loss_messages[loss_at]["content"]
        if chosen != rejected:
            return win_messages[:win_at], chosen, rejected
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
    (win, loss) combination becomes one pair at the first point of
    divergence: both trajectories' assistant turns align by index,
    and the first index where the contents differ gives chosen
    (the win's turn) and rejected (the loss's turn). The pair
    prompt is the shared prefix: system, task prompt, and the
    identical earlier turns, kept both as chat messages and as one
    flattened prompt string. Combinations that never differ carry
    no signal and are skipped. Pairs longer than the token cap are
    skipped: one outlier pair once padded a whole training run to
    8320 tokens and stalled it.

    The train/valid split happens by task, before pairing: about
    one tenth of the paired tasks, at least one, go to validation,
    and pairs are built within each side only, so no episode can
    appear on both sides. With a single paired task, every pair
    goes to train and validation stays empty rather than
    duplicated; the trainer already refuses short datasets.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    wins: dict[str, list[list[dict[str, str]]]] = {}
    losses: dict[str, list[list[dict[str, str]]]] = {}
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
            else TASK_PROMPT_PREFIX
        )
        messages = _render_messages(trajectory, prompt)
        if not any(
            message["role"] == "assistant" for message in messages
        ):
            continue
        if float(record["reward"]) >= 1.0:
            wins.setdefault(task, []).append(messages)
        elif _episode_qualifies_as_loss(session_file):
            losses.setdefault(task, []).append(messages)

    paired_tasks = sorted(
        task for task in tasks_seen if wins.get(task) and losses.get(task)
    )
    shuffled_tasks = list(paired_tasks)
    random.Random(7).shuffle(shuffled_tasks)
    valid_task_count = (
        max(1, len(shuffled_tasks) // VALID_TRAJECTORY_SHARE)
        if len(shuffled_tasks) > 1
        else 0
    )
    valid_tasks = frozenset(shuffled_tasks[:valid_task_count])

    train: list[str] = []
    valid: list[str] = []
    skipped_oversize = 0
    skipped_identical = 0
    for task in paired_tasks:
        documents = valid if task in valid_tasks else train
        written = 0
        for win_messages in wins[task]:
            for loss_messages in losses[task]:
                if written >= max_pairs_per_task:
                    break
                divergence = _first_divergence(
                    win_messages, loss_messages
                )
                if divergence is None:
                    skipped_identical += 1
                    continue
                prefix, chosen, rejected = divergence
                total_tokens = sum(
                    count(message["content"]) for message in prefix
                ) + max(count(chosen), count(rejected))
                if total_tokens > token_cap:
                    skipped_oversize += 1
                    continue
                documents.append(
                    json.dumps(
                        {
                            "system": SYSTEM_PROMPT,
                            "prompt": "\n\n".join(
                                message["content"]
                                for message in prefix
                                if message["role"] != "system"
                            ),
                            "chosen": chosen,
                            "rejected": rejected,
                            "messages": prefix,
                        }
                    )
                )
                written += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("\n".join(train) + ("\n" if train else ""))
    (out_dir / "valid.jsonl").write_text("\n".join(valid) + ("\n" if valid else ""))
    return PairStats(
        tasks_seen=len(tasks_seen),
        tasks_paired=len(paired_tasks),
        pairs_written=len(train) + len(valid),
        pairs_skipped_oversize=skipped_oversize,
        pairs_skipped_identical=skipped_identical,
    )


def export_dataset(
    runs_dir: Path,
    sessions_root: Path | None,
    out_dir: Path,
    min_reward: float,
    tokenizer_id: str,
    token_cap: int,
    min_quality: bool = True,
    holdout_dir: Path = Path("holdout-tasks"),
) -> ExportStats:
    """Collect both sources and write per-turn train/valid files.

    Session harvest is opt-in: when sessions_root is None, or when
    the path does not exist, no session is read and sessions_seen
    stays 0. The train/valid split separates whole trajectories so
    that no session contributes samples to both files. The valid
    set takes randomly chosen trajectories, with a fixed seed,
    until it holds about one tenth of the samples. After the
    writes, the written files are re-read and scanned for holdout
    fingerprints; any hit aborts the export with SystemExit.
    """
    count_tokens = load_token_counter(tokenizer_id)
    (
        episode_msgs,
        episodes_seen,
        episodes_excluded,
        episode_torn,
    ) = _collect_episode_messages(runs_dir, min_reward, holdout_dir)
    if sessions_root is not None and sessions_root.is_dir():
        (
            session_msgs,
            sessions_seen,
            sessions_filtered,
            excluded_holdout,
            excluded_content,
            session_torn,
        ) = _collect_session_messages(
            sessions_root, min_quality, holdout_dir
        )
    else:
        if sessions_root is not None:
            print(f"note: sessions root {sessions_root} does not exist")
        session_msgs, sessions_seen = [], 0
        sessions_filtered, excluded_holdout = 0, 0
        excluded_content, session_torn = 0, 0

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
        fingerprint_hits = _scan_written_dataset(
            out_dir, _holdout_fingerprints(holdout_dir)
        )
        return ExportStats(
            episodes_seen=episodes_seen,
            episodes_excluded_holdout=episodes_excluded,
            sessions_seen=sessions_seen,
            sessions_filtered=sessions_filtered,
            sessions_excluded_holdout=excluded_holdout,
            trajectories_exported=0,
            turns_exported=0,
            turns_skipped_oversize=skipped,
            torn_lines=episode_torn + session_torn,
            train_samples=0,
            valid_samples=0,
            sessions_excluded_holdout_content=excluded_content,
            dataset_fingerprint_hits=fingerprint_hits,
        )

    if len(splits) == 1:
        train_splits, valid_splits = list(splits), list(splits)
        print("warning: one trajectory only; valid set repeats the train set")
    else:
        total_samples = sum(len(split.samples) for split in splits)
        valid_budget = max(1, total_samples // VALID_TRAJECTORY_SHARE)
        ordered = list(splits)
        random.Random(7).shuffle(ordered)
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
    fingerprint_hits = _scan_written_dataset(
        out_dir, _holdout_fingerprints(holdout_dir)
    )
    return ExportStats(
        episodes_seen=episodes_seen,
        episodes_excluded_holdout=episodes_excluded,
        sessions_seen=sessions_seen,
        sessions_filtered=sessions_filtered,
        sessions_excluded_holdout=excluded_holdout,
        trajectories_exported=len(splits),
        turns_exported=len(train) + len(valid),
        turns_skipped_oversize=skipped,
        torn_lines=episode_torn + session_torn,
        train_samples=len(train),
        valid_samples=len(valid),
        sessions_excluded_holdout_content=excluded_content,
        dataset_fingerprint_hits=fingerprint_hits,
    )
