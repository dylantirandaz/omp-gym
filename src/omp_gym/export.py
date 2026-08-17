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
as the final message. Train with prompt masking so that only the
final assistant message produces loss.

Tool calls ride inside <tool_call id="..."> envelopes carrying the
call id, so the serving shim can round-trip sequence part of the
message as plain text. Tool results ride inside matching
<tool_response id="..."> envelopes. Prose text the model wrote
together with its tool calls stays in the sample before the calls.

Every written record carries a schema_version and a provenance
object naming the session, the task, the session source, and, for
scored episodes, the reward improvement. A manifest beside the
train/valid files records counts, the tokenizer, the budgeting
mode, and the git sha when available.
"""

import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
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

SCHEMA_VERSION = "2"

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


class SampleCounter(Protocol):
    """Counts the chat-template token cost of one message list."""

    def __call__(self, messages: Sequence[dict[str, str]]) -> int: ...


def load_token_counter(tokenizer_id: str) -> SampleCounter:
    """Load the trainee tokenizer and return a chat-template counter."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def counter(messages: Sequence[dict[str, str]]) -> int:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        return len(tokens)

    return counter


def _heuristic_sample_cost(messages: Sequence[dict[str, str]]) -> int:
    """Count one aligned token per four charged characters."""

    return sum(
        len(message["content"]) // 4 + 1 + MESSAGE_OVERHEAD_TOKENS
        for message in messages
    )


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
    deduped: int = 0
    quarantined: int = 0
    redacted: int = 0
    suspect: int = 0
    binary_skipped: int = 0


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


def _normalize_workspace_paths(text: str, anchor: str | None = None) -> str:
    """Replace the workspace root with the relative root ".".

    When an anchor is available (the exact workspace path recorded
    by the episode), only that exact string goes. Without one, the
    ws-segment regex applies; session harvest has no episode record
    to anchor against.
    """
    if anchor:
        return text.replace(anchor, ".")
    return _WORKSPACE_PATH_PATTERN.sub(".", text)


def _normalize_tool_arguments(
    arguments: dict[str, object],
    anchor: str | None = None,
) -> dict[str, object]:
    """Replace the recorded workspace paths with relative paths."""
    normalized = dict(arguments)
    for name, value in arguments.items():
        if not isinstance(value, str):
            continue
        relative_value = _normalize_workspace_paths(value, anchor)
        if name == "path" and relative_value.startswith("./"):
            relative_value = relative_value[2:]
        normalized[name] = relative_value
    return normalized


def _authoring_target_path(call: ToolCall, anchor: str | None = None) -> str | None:
    """Return the workspace-relative path an edit or write touches."""
    arguments = _normalize_tool_arguments(call.arguments, anchor)
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
    anchor: str | None = None,
    final_for_path: bool = True,
) -> tuple[str, dict[str, object], bool]:
    """Render the final successful edit of a file as a full write.

    The result is a synthetic trajectory reconstruction, not the
    real call sequence. Only the last file-touching call for a
    path leaves the file in its on-disk state, so only that call
    gets the file content as its label. Earlier edits keep their
    real patch arguments, and write calls always keep their own
    content. A binary or non-UTF-8 final file skips the canonical
    reconstruction; the bool flag reports that skip.
    """
    arguments = _normalize_tool_arguments(call.arguments, anchor)
    if call.name != "edit" or authoring_workspace is None or not final_for_path:
        return call.name, arguments, False
    relative_path = _authoring_target_path(call, anchor)
    if relative_path is None or Path(relative_path).is_absolute():
        return call.name, arguments, False
    workspace = authoring_workspace.resolve()
    final_file = (workspace / relative_path).resolve()
    if not final_file.is_relative_to(workspace) or not final_file.is_file():
        return call.name, arguments, False
    intent = arguments.get("i")
    try:
        content = final_file.read_text()
    except UnicodeDecodeError:
        return call.name, arguments, True
    return (
        "write",
        {
            "path": relative_path,
            "content": content,
            "i": intent
            if isinstance(intent, str) and intent
            else "Write verified file",
        },
        False,
    )


_AUTHORING_TOOL_NAMES = frozenset({"edit", "write"})


def _final_authoring_call_ids(
    trajectory: Trajectory,
    failed_authoring_call_ids: frozenset[str],
    anchor: str | None = None,
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
            target = _authoring_target_path(call, anchor)
            if target is not None:
                last_call_by_path[target] = call.call_id
    return frozenset(last_call_by_path.values())


def _envelope(tag: str, call_id: str, body: str) -> str:
    """Render one tagged envelope, carrying the call id when known."""
    attribute = f' id="{call_id}"' if call_id else ""
    return f"<{tag}{attribute}>\n{body}\n</{tag}>"


def _render_detail(
    trajectory: Trajectory,
    prompt: str | None,
    authoring_workspace: Path | None = None,
) -> tuple[list[dict[str, str]], int, int]:
    """Render one trajectory as merged chat messages with counts.

    Scored episodes pass their task prompt. Harvested sessions pass
    None because their first user step already is the prompt.
    Returns (messages, redacted messages, binary-skipped calls).
    """
    anchor = str(authoring_workspace) if authoring_workspace else None
    failed_authoring_call_ids = frozenset(
        step.call_id
        for step in trajectory.steps
        if isinstance(step, ToolResultStep)
        and step.is_error
        and step.tool_name in _AUTHORING_TOOL_NAMES
    )
    final_authoring_call_ids = _final_authoring_call_ids(
        trajectory, failed_authoring_call_ids, anchor
    )
    binary_skipped = 0
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
                parts = [step.text] if step.text else []
                for call in calls:
                    name, arguments, skipped = _canonical_training_call(
                        call,
                        authoring_workspace,
                        anchor,
                        call.call_id in final_authoring_call_ids,
                    )
                    binary_skipped += int(skipped)
                    payload = json.dumps(
                        {
                            "name": name,
                            "arguments": arguments,
                        }
                    )
                    parts.append(_envelope("tool_call", call.call_id, payload))
                content = "\n".join(parts)
                if content:
                    messages.append({"role": "assistant", "content": content})
            case ToolResultStep():
                if step.call_id in failed_authoring_call_ids:
                    continue
                body = _elide_middle(
                    _normalize_workspace_paths(step.text, anchor),
                    TOOL_RESULT_LIMIT,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": _envelope("tool_response", step.call_id, body),
                    }
                )
            case UserStep():
                if step.text:
                    messages.append({"role": "user", "content": step.text})

    merged: list[dict[str, str]] = []
    redacted = 0
    for message in messages:
        content = _redact(message["content"])
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {
                "role": message["role"],
                "content": merged[-1]["content"] + "\n\n" + content,
            }
        else:
            merged.append({"role": message["role"], "content": content})
        if merged[-1]["content"] != message["content"]:
            redacted += 1
    return merged, redacted, binary_skipped


def _render_messages(
    trajectory: Trajectory,
    prompt: str | None,
    authoring_workspace: Path | None = None,
) -> list[dict[str, str]]:
    """Render one trajectory as merged chat messages."""
    messages, _, _ = _render_detail(trajectory, prompt, authoring_workspace)
    return messages


@dataclass(frozen=True)
class CollectedTrajectory:
    """One rendered trajectory with its origin and its counts."""

    messages: list[dict[str, str]]
    provenance: dict[str, object]
    redacted: int
    binary_skipped: int
    torn_lines: int


@dataclass(frozen=True)
class TurnSplit:
    """Per-turn samples of one trajectory, with skip count."""

    samples: tuple[str, ...]
    skipped_oversize: int


def _is_tool_result_envelope(message: dict[str, str]) -> bool:
    """Envelope-tagged user content never opens a sample's context."""
    return message["role"] == "user" and (
        message["content"].startswith("<tool_response")
    )


def _elide_turn_container(content: str, limit: int) -> str:
    """Elide one turn's prose text while preserving tool envelopes.

    A tool_call envelope carries one JSON payload; cutting inside
    it corrupts the structure and rains a worthless parser into
    the model. Only the prose portion takes the cut; the envelopes
    go through whole.
    """
    lines = content.split("\n")
    envelope_lines = [line for line in lines if line.lstrip().startswith("<tool_call")]
    prose_lines = [line for line in lines if not line.lstrip().startswith("<tool_call")]
    prose = "\n".join(prose_lines)
    elided = _elide_middle(prose, max(limit, 0))
    parts = ([elided] if elided else []) + envelope_lines
    return "\n".join(parts)


def _sample_record(
    sample: list[dict[str, str]],
    collected: CollectedTrajectory,
) -> dict[str, object]:
    """One exported sample wrapped in schema and provenance info."""
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "messages": sample,
        "provenance": collected.provenance,
    }
    if collected.torn_lines > 0:
        record["suspect"] = True
    return record


def _split_turns(
    collected: CollectedTrajectory,
    sample_cost: SampleCounter,
    budget: int,
) -> TurnSplit:
    """Make one sample per assistant turn with tail-window context.

    The sample keeps the system prompt, then as many of the most
    recent context messages as the token budget allows, then the
    assistant turn as the final message. The start of the context
    window never opens on a tool-result envelope; it advances to
    the next boundary message instead. A turn whose bare sample
    (system plus turn) exceeds the budget elides only its prose
    text; if the full sample still exceeds the budget the sample
    is dropped and counted as oversize.
    """
    messages = collected.messages
    if not messages or messages[0]["role"] != "system":
        raise AssertionError("rendered messages must start with system")
    samples: list[str] = []
    skipped = 0
    for index in range(1, len(messages)):
        turn = messages[index]
        if turn["role"] != "assistant":
            continue
        if sample_cost([messages[0], turn]) > budget:
            turn = dict(turn)
            char_limit = max(budget, 0) * 3
            turn["content"] = _elide_turn_container(turn["content"], char_limit)
            if sample_cost([messages[0], turn]) > budget:
                skipped += 1
                continue
        start = index
        while start > 1:
            candidate = [messages[0], *messages[start - 1 : index], turn]
            if sample_cost(candidate) > budget:
                break
            start -= 1
        while start < index and _is_tool_result_envelope(messages[start]):
            start += 1
        sample = [messages[0], *messages[start:index], turn]
        if sample_cost(sample) > budget:
            skipped += 1
            continue
        samples.append(json.dumps(_sample_record(sample, collected)))
    return TurnSplit(samples=tuple(samples), skipped_oversize=skipped)


def _holdout_task_names(holdout_dir: Path) -> frozenset[str]:
    """Directory names under the holdout dir; empty when absent."""
    if not holdout_dir.is_dir():
        return frozenset()
    return frozenset(entry.name for entry in holdout_dir.iterdir() if entry.is_dir())


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
        if not (entry.name.startswith("test_") or entry.stem.endswith("_test")):
            continue
        lines = [
            line.strip() for line in entry.read_text(errors="replace").splitlines()
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
        return [leaf for child in value.values() for leaf in _string_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _string_leaves(child)]
    return []


def _holdout_needles(fingerprints: frozenset[str]) -> dict[str, str]:
    """Pre-squashed fingerprint needles for one unified match."""
    return {fingerprint: _squash(fingerprint) for fingerprint in fingerprints}


def _fingerprint_hit(parts: list[str], needles: dict[str, str]) -> str | None:
    """The one holdout hit shared by per-sample and dataset checks.

    Both the per-message filter and the post-write dataset scan
    call this ONE helper, so both apply the same whitespace-free
    normalization.
    """
    haystacks = [_squash(part) for part in parts]
    for fingerprint, needle in needles.items():
        if any(needle in haystack for haystack in haystacks):
            return fingerprint
    return None


def _scan_written_dataset(
    out_dir: Path,
    fingerprints: frozenset[str],
) -> int:
    """Re-read the written files and fail on any holdout fingerprint.

    The exclusion filters run on rendered text before writing; this
    scan re-opens exactly what training will read. The same unified
    _fingerprint_hit helper used on rendered messages decides, so
    the filter and the scan never drift apart. Returns 0; any hit
    raises SystemExit naming the file and the fingerprint.
    """
    needles = _holdout_needles(fingerprints)
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
            hit = _fingerprint_hit(parts, needles)
            if hit is not None:
                raise SystemExit(
                    f"holdout fingerprint leaked into {path} line {number}: {hit!r}"
                )
    return 0


def _load_episode(
    episode_file: Path,
) -> tuple[CollectedTrajectory | None, dict[str, object] | None]:
    """Read one episode, quarantining malformed or dishonest records.

    A malformed record, an unreadable session, or a session_file
    that resolves outside the episode_dir returns (None, None);
    the caller counts the record as quarantined and never follows
    the escape. The episode record's workspace anchor underlies
    the normalization of rendered paths, so only the recorded
    absolute workspace string collapses to ".".
    """
    try:
        record = json.loads(episode_file.read_text())
        episode_dir = Path(str(record["episode_dir"])).resolve()
        session_file = Path(str(record["session_file"])).resolve()
        if not session_file.is_relative_to(episode_dir):
            return None, None
        trajectory = parse_session(session_file)
        prompt_file = episode_dir / "prompt.txt"
        prompt = (
            prompt_file.read_text().strip()
            if prompt_file.is_file()
            else TASK_PROMPT_PREFIX
        )
        workspace = episode_dir / "ws"
        messages, redacted, binary_skipped = _render_detail(
            trajectory, prompt, workspace
        )
        provenance: dict[str, object] = {
            "session": session_file.name,
            "task": str(record.get("task", "")),
            "source": trajectory.source,
        }
        improvement = record.get("reward_improvement")
        if improvement is not None:
            provenance["reward_improvement"] = improvement
        collected = CollectedTrajectory(
            messages=messages,
            provenance=provenance,
            redacted=redacted,
            binary_skipped=binary_skipped,
            torn_lines=trajectory.torn_lines,
        )
        return collected, record
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None, None


def _collect_episode_messages(
    runs_dir: Path,
    min_reward: float,
    holdout_dir: Path,
) -> tuple[
    list[CollectedTrajectory],
    int,
    int,
    int,
    int,
]:
    """Render scored episodes: (rendered, seen, excluded, torn, quarantined).

    An episode whose task carries the name of a holdout task, or
    whose rendered text contains a holdout content fingerprint,
    never enters the dataset: training on it would leak the
    benchmark. A malformed or escaping record counts as
    quarantined and never aborts the export.
    """
    holdout_names = _holdout_task_names(holdout_dir)
    needles = _holdout_needles(_holdout_fingerprints(holdout_dir))
    rendered: list[CollectedTrajectory] = []
    seen = 0
    excluded = 0
    quarantined = 0
    torn = 0
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        seen += 1
        collected, record = _load_episode(episode_file)
        if collected is None or record is None:
            quarantined += 1
            continue
        if str(record.get("task", "")) in holdout_names:
            excluded += 1
            continue
        if float(record.get("reward", 0.0)) < min_reward:
            continue
        torn += collected.torn_lines
        hit = _fingerprint_hit(
            [message["content"] for message in collected.messages],
            needles,
        )
        if hit is not None:
            excluded += 1
            continue
        rendered.append(collected)
    return rendered, seen, excluded, torn, quarantined


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
) -> tuple[
    list[CollectedTrajectory],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Render all omp sessions.

    Returns (rendered, seen, filtered, excluded_holdout,
    excluded_content, torn, quarantined). With min_quality on,
    sessions that ended as failures do not enter the SFT dataset.
    They remain available for DPO pairs. Sessions that mention
    the holdout task directory, or whose rendered text contains
    a holdout content fingerprint, never enter the dataset:
    training on them would leak the benchmark. Unreadable or
    broken sessions count as quarantined, never aborts.
    """
    needles = _holdout_needles(_holdout_fingerprints(holdout_dir))
    rendered: list[CollectedTrajectory] = []
    seen = 0
    filtered = 0
    excluded_holdout = 0
    excluded_content = 0
    quarantined = 0
    torn = 0
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        seen += 1
        try:
            trajectory = parse_session(session_file)
        except OSError:
            quarantined += 1
            continue
        torn += trajectory.torn_lines
        if min_quality and _session_failed(session_file, trajectory):
            filtered += 1
            continue
        messages, redacted, binary_skipped = _render_detail(trajectory, prompt=None)
        if any("holdout-tasks" in message["content"] for message in messages):
            excluded_holdout += 1
            continue
        hit = _fingerprint_hit(
            [message["content"] for message in messages],
            needles,
        )
        if hit is not None:
            excluded_content += 1
            continue
        rendered.append(
            CollectedTrajectory(
                messages=messages,
                provenance={
                    "session": session_file.name,
                    "task": session_file.stem,
                    "source": trajectory.source,
                },
                redacted=redacted,
                binary_skipped=binary_skipped,
                torn_lines=trajectory.torn_lines,
            )
        )
    return (
        rendered,
        seen,
        filtered,
        excluded_holdout,
        excluded_content,
        torn,
        quarantined,
    )


@dataclass(frozen=True)
class PairStats:
    """What the preference-pair export produced."""

    tasks_seen: int
    tasks_paired: int
    pairs_written: int
    pairs_skipped_oversize: int
    pairs_skipped_identical: int = 0
    pairs_skipped_context: int = 0
    quarantined: int = 0
    excluded_holdout: int = 0


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


_CONTEXT_MISMATCH = object()


def _first_divergence(
    win_messages: list[dict[str, str]],
    loss_messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str, str] | object | None:
    """Shared prefix plus the first differing assistant turns.

    Assistant turns align by index across the two renderings. The
    first index whose contents differ yields the pair; the prompt
    is everything before the win's divergent turn. The shared
    prefix must match message-for-message (role and content) up to
    the divergence, otherwise the divergent turn pair comes from
    unrelated contexts and returns the mismatch sentinel.
    Trajectories that agree at every compared index yield None.
    """
    win_turns = [
        at for at, message in enumerate(win_messages) if message["role"] == "assistant"
    ]
    loss_turns = [
        at for at, message in enumerate(loss_messages) if message["role"] == "assistant"
    ]
    for win_at, loss_at in zip(win_turns, loss_turns, strict=False):
        chosen = win_messages[win_at]["content"]
        rejected = loss_messages[loss_at]["content"]
        if chosen == rejected:
            continue
        win_prefix = win_messages[:win_at]
        loss_prefix = loss_messages[:loss_at]
        if len(win_prefix) != len(loss_prefix):
            return _CONTEXT_MISMATCH
        mismatched = any(
            before["role"] != after["role"] or before["content"] != after["content"]
            for before, after in zip(win_prefix, loss_prefix, strict=False)
        )
        if mismatched:
            return _CONTEXT_MISMATCH
        return win_prefix, chosen, rejected
    return None


def _git_sha() -> str | None:
    """The current repository sha; None when unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed static argv
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _write_dataset(
    out_dir: Path,
    train: list[str],
    valid: list[str],
    manifest: dict[str, object],
    fingerprints: frozenset[str],
) -> None:
    """Write the dataset through a temp directory with a leak scan.

    Files are built in a temporary sibling directory, the holdout
    scan runs against them there, and only a clean scan promotes
    the files into place. A failed scan leaves no partial dataset
    in the output directory.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{out_dir.name}-", dir=out_dir.parent
    ) as staging:
        staging_dir = Path(staging)
        (staging_dir / "train.jsonl").write_text(
            "\n".join(train) + ("\n" if train else "")
        )
        (staging_dir / "valid.jsonl").write_text(
            "\n".join(valid) + ("\n" if valid else "")
        )
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        _scan_written_dataset(staging_dir, fingerprints)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
            shutil.move(str(staging_dir / name), str(out_dir / name))


def _canonical_messages_hash(messages: Sequence[dict[str, str]]) -> str:
    """sha256 over the canonical messages JSON of one trajectory."""
    canonical = json.dumps(list(messages), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dedup_trajectories(
    trajectories: list[CollectedTrajectory],
) -> tuple[list[CollectedTrajectory], int]:
    """Drop identical renderings; the first occurrence stays."""
    seen: set[str] = set()
    kept: list[CollectedTrajectory] = []
    deduped = 0
    for trajectory in trajectories:
        digest = _canonical_messages_hash(trajectory.messages)
        if digest in seen:
            deduped += 1
            continue
        seen.add(digest)
        kept.append(trajectory)
    return kept, deduped


def export_pairs(
    runs_dir: Path,
    out_dir: Path,
    max_pairs_per_task: int,
    tokenizer_id: str,
    token_cap: int,
    holdout_dir: Path = Path("holdout-tasks"),
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
    no signal and are skipped. When the two contexts differ before
    the divergence, the pair is skipped as well. Pairs longer than
    the token cap are skipped: one outlier pair once padded a whole
    training run to 8320 tokens and stalled it.

    The train/valid split happens by task, before pairing: about
    one tenth of the paired tasks, at least one, go to validation,
    and pairs are built within each side only, so no episode can
    appear on both sides. With a single paired task, every pair
    goes to train and validation stays empty rather than
    duplicated; the trainer already refuses short datasets.

    Episodes named for a holdout task, or carrying a holdout
    content fingerprint, never enter the pair set, and the final
    files pass through the same pre-write scan as the SFT export.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    holdout_names = _holdout_task_names(holdout_dir)
    fingerprints = _holdout_fingerprints(holdout_dir)
    needles = _holdout_needles(fingerprints)

    wins: dict[str, list[CollectedTrajectory]] = {}
    losses: dict[str, list[CollectedTrajectory]] = {}
    tasks_seen: set[str] = set()
    quarantined = 0
    excluded_holdout = 0
    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        collected, record = _load_episode(episode_file)
        if collected is None or record is None:
            quarantined += 1
            continue
        task = str(record.get("task", ""))
        tasks_seen.add(task)
        if task in holdout_names:
            excluded_holdout += 1
            continue
        hit = _fingerprint_hit(
            [message["content"] for message in collected.messages],
            needles,
        )
        if hit is not None:
            excluded_holdout += 1
            continue
        if not any(message["role"] == "assistant" for message in collected.messages):
            continue
        if float(record.get("reward", 0.0)) >= 1.0:
            wins.setdefault(task, []).append(collected)
        else:
            session_file = Path(str(record["session_file"]))
            try:
                if _episode_qualifies_as_loss(session_file):
                    losses.setdefault(task, []).append(collected)
            except OSError:
                quarantined += 1

    paired_tasks = sorted(
        task for task in tasks_seen if wins.get(task) and losses.get(task)
    )
    shuffled_tasks = list(paired_tasks)
    random.Random(7).shuffle(shuffled_tasks)  # noqa: S311 - seeded split
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
    skipped_context = 0
    for task in paired_tasks:
        documents = valid if task in valid_tasks else train
        written = 0
        for win in wins[task]:
            for loss in losses[task]:
                if written >= max_pairs_per_task:
                    break
                divergence = _first_divergence(win.messages, loss.messages)
                if divergence is _CONTEXT_MISMATCH:
                    skipped_context += 1
                    continue
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
                            "schema_version": SCHEMA_VERSION,
                            "provenance": {
                                "task": task,
                                "chosen_session": win.provenance["session"],
                                "rejected_session": loss.provenance["session"],
                                "source": win.provenance["source"],
                            },
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

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "tokenizer": tokenizer_id,
        "budgeting": "heuristic",
        "train_samples": len(train),
        "valid_samples": len(valid),
        "tasks_seen": len(tasks_seen),
        "tasks_paired": len(paired_tasks),
        "pairs_written": len(train) + len(valid),
        "pairs_skipped_oversize": skipped_oversize,
        "pairs_skipped_identical": skipped_identical,
        "pairs_skipped_context": skipped_context,
        "quarantined": quarantined,
        "excluded_holdout": excluded_holdout,
    }
    _write_dataset(out_dir, train, valid, manifest, fingerprints)
    return PairStats(
        tasks_seen=len(tasks_seen),
        tasks_paired=len(paired_tasks),
        pairs_written=len(train) + len(valid),
        pairs_skipped_oversize=skipped_oversize,
        pairs_skipped_identical=skipped_identical,
        pairs_skipped_context=skipped_context,
        quarantined=quarantined,
        excluded_holdout=excluded_holdout,
    )


def export_dataset(
    runs_dir: Path,
    sessions_root: Path | None,
    out_dir: Path,
    min_reward: float,
    tokenizer: str | None = None,
    token_cap: int = 2048,
    min_quality: bool = True,
    holdout_dir: Path = Path("holdout-tasks"),
    strict_torn: bool = False,
) -> ExportStats:
    """Collect both sources and write per-turn train/valid files.

    Session harvest is opt-in: when sessions_root is None, or when
    the path does not exist, no session is read and sessions_seen
    stays 0. The train/valid split separates whole task families
    so that samples from the same provenance task never span both
    files. The valid set takes randomly chosen task groups, with a
    fixed seed, until it holds about one tenth of the samples. A
    single trajectory duplicates into valid with a warning, and
    the manifest records valid_duplicated: true.

    Budgeting: with a tokenizer id, the trainee tokenizer's chat
    template counts exact tokens; without one, a character
    heuristic estimates them, recorded as budgeting: heuristic in
    the manifest. Samples still over the cap after prose elision
    are dropped and counted. Output files are built in a temp
    directory and scanned for holdout fingerprints before moving
    into place; a failed scan leaves no partial dataset.
    """
    if tokenizer is not None:
        sample_cost = load_token_counter(tokenizer)
        budgeting = "exact"
        budget = token_cap
    else:
        sample_cost = _heuristic_sample_cost
        budgeting = "heuristic"
        budget = token_cap - TOKEN_SAFETY_MARGIN

    (
        episode_msgs,
        episodes_seen,
        episodes_excluded,
        episode_torn,
        episodes_quarantined,
    ) = _collect_episode_messages(runs_dir, min_reward, holdout_dir)
    if sessions_root is not None and sessions_root.is_dir():
        (
            session_msgs,
            sessions_seen,
            sessions_filtered,
            excluded_holdout,
            excluded_content,
            session_torn,
            sessions_quarantined,
        ) = _collect_session_messages(sessions_root, min_quality, holdout_dir)
    else:
        if sessions_root is not None:
            print(f"note: sessions root {sessions_root} does not exist")
        session_msgs, sessions_seen = [], 0
        sessions_filtered, excluded_holdout = 0, 0
        excluded_content, session_torn = 0, 0
        sessions_quarantined = 0

    trajectories = episode_msgs + session_msgs
    torn = episode_torn + session_torn
    if strict_torn and torn > 0:
        raise SystemExit(
            f"torn sessions: {torn} unreadable lines; export aborted (strict_torn)"
        )
    trajectories, deduped = _dedup_trajectories(trajectories)
    quarantined = episodes_quarantined + sessions_quarantined
    redacted = sum(trajectory.redacted for trajectory in trajectories)
    binary_skipped = sum(trajectory.binary_skipped for trajectory in trajectories)

    splits = [
        _split_turns(trajectory, sample_cost, budget) for trajectory in trajectories
    ]
    kept = [
        (trajectory, split)
        for trajectory, split in zip(trajectories, splits, strict=True)
        if split.samples
    ]
    skipped = sum(split.skipped_oversize for _, split in kept)
    suspect = sum(
        len(split.samples) for trajectory, split in kept if trajectory.torn_lines > 0
    )

    fingerprints = _holdout_fingerprints(holdout_dir)
    manifest_base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "tokenizer": tokenizer,
        "budgeting": budgeting,
        "episodes_seen": episodes_seen,
        "episodes_excluded_holdout": episodes_excluded,
        "sessions_seen": sessions_seen,
        "sessions_filtered": sessions_filtered,
        "sessions_excluded_holdout": excluded_holdout,
        "sessions_excluded_holdout_content": excluded_content,
        "deduped": deduped,
        "quarantined": quarantined,
        "redacted": redacted,
        "suspect": suspect,
        "binary_skipped": binary_skipped,
        "torn_lines": torn,
    }

    valid_duplicated = False
    if not kept:
        _write_dataset(out_dir, [], [], manifest_base, fingerprints)
        return ExportStats(
            episodes_seen=episodes_seen,
            episodes_excluded_holdout=episodes_excluded,
            sessions_seen=sessions_seen,
            sessions_filtered=sessions_filtered,
            sessions_excluded_holdout=excluded_holdout,
            trajectories_exported=0,
            turns_exported=0,
            turns_skipped_oversize=skipped,
            torn_lines=torn,
            train_samples=0,
            valid_samples=0,
            sessions_excluded_holdout_content=excluded_content,
            dataset_fingerprint_hits=0,
            deduped=deduped,
            quarantined=quarantined,
            redacted=redacted,
            suspect=suspect,
            binary_skipped=binary_skipped,
        )

    groups: dict[str, list[TurnSplit]] = {}
    for trajectory, split in kept:
        task_name = str(trajectory.provenance.get("task", ""))
        groups.setdefault(task_name, []).append(split)

    if len(groups) == 1:
        valid_duplicated = True
        train_splits = [split for group in groups.values() for split in group]
        valid_splits = list(train_splits)
        print("warning: one trajectory only; valid set repeats the train set")
    else:
        total_samples = sum(len(split.samples) for _, split in kept)
        valid_budget = max(1, total_samples // VALID_TRAJECTORY_SHARE)
        ordered_tasks = sorted(groups)
        random.Random(7).shuffle(ordered_tasks)  # noqa: S311 - seeded split
        valid_tasks: set[str] = set()
        valid_taken = 0
        for task_name in ordered_tasks:
            if valid_taken >= valid_budget:
                break
            valid_tasks.add(task_name)
            valid_taken += sum(len(split.samples) for split in groups[task_name])
        train_splits = [
            split
            for task_name, group in groups.items()
            if task_name not in valid_tasks
            for split in group
        ]
        valid_splits = [
            split
            for task_name, group in groups.items()
            if task_name in valid_tasks
            for split in group
        ]

    train = [sample for split in train_splits for sample in split.samples]
    valid = [sample for split in valid_splits for sample in split.samples]
    manifest: dict[str, object] = {
        **manifest_base,
        "train_samples": len(train),
        "valid_samples": len(valid),
        "trajectories_exported": len(kept),
        "turns_exported": len(train) + len(valid),
        "turns_skipped_oversize": skipped,
        "valid_duplicated": valid_duplicated,
    }
    _write_dataset(out_dir, train, valid, manifest, fingerprints)
    return ExportStats(
        episodes_seen=episodes_seen,
        episodes_excluded_holdout=episodes_excluded,
        sessions_seen=sessions_seen,
        sessions_filtered=sessions_filtered,
        sessions_excluded_holdout=excluded_holdout,
        trajectories_exported=len(kept),
        turns_exported=len(train) + len(valid),
        turns_skipped_oversize=skipped,
        torn_lines=torn,
        train_samples=len(train),
        valid_samples=len(valid),
        sessions_excluded_holdout_content=excluded_content,
        dataset_fingerprint_hits=0,
        deduped=deduped,
        quarantined=quarantined,
        redacted=redacted,
        suspect=suspect,
        binary_skipped=binary_skipped,
    )
