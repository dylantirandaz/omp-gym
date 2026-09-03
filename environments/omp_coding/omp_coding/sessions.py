"""Read OMP session files and split them into episodes.

OMP persists one append-only JSONL file per session under
``~/.omp/agent/sessions/<encoded-cwd>/<timestamp>_<id>.jsonl``. Line one is a
fixed-width title slot, line two is the ``session`` header, and every later
line is one tree entry with ``id`` and ``parentId``. Tool output that spilled
to disk and subagent transcripts live in the sibling directory that shares
the file stem.

Verified against real files: successful ``bash`` results omit
``details.exitCode``; multi-file ``edit`` results carry one entry per file
under ``details.perFileResults``; ``details.snapshotsPruned`` marks edits
whose ``oldText``/``newText`` were dropped; spills live in ``N.<tool>.log``
or ``N.<tool>-original.log``; subagent sidecars share the parent's artifact
directory.

This module is pure standard library so it runs on any host that holds
session files, including Windows.
"""

from __future__ import annotations

import json
import ntpath
import posixpath
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SESSION_VERSION = 3
MAX_SESSION_BYTES = 512 * 1024 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
TRUNCATION_MARK = "[Session persistence truncated large content]"
_TITLE_SLOT = "title"
_ARTIFACT_REF = re.compile(r"artifact://(\d+)")
_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_SYSTEM_REMINDER = re.compile(r"<system-reminder\b[^>]*>.*?</system-reminder>", re.DOTALL)
_WRAPPER = re.compile(r"\A<([A-Za-z][\w-]*)\b[^>]*>\n?(.*)</\1>\Z", re.DOTALL)
_SURPLUS_BLANK_LINES = re.compile(r"\n{3,}")

Attribution = Literal["user", "agent", "system", "unknown"]
MutationKind = Literal["write", "edit", "delete"]
EpisodeEnd = Literal["user", "exit", "end"]

_EDIT_KINDS: Mapping[str, MutationKind] = {
    "update": "edit",
    "create": "write",
    "delete": "delete",
}


@dataclass(frozen=True)
class SessionHeader:
    """Identity of one session file."""

    id: str
    cwd: str
    started_at: datetime
    title: str
    path: Path
    artifact_dir: Path
    parent_session: str | None = None


@dataclass(frozen=True)
class Usage:
    """Token and cost accounting summed over assistant messages."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost,
        )


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the assistant."""

    call_id: str
    name: str
    arguments: Mapping[str, object]
    intent: str = ""


@dataclass(frozen=True)
class ToolResult:
    """The recorded answer to one tool call.

    ``text`` is the text the model saw. ``artifact_text`` is the full spilled
    output when the text references ``artifact://N`` and the log file exists.
    ``exit_code`` is the process exit status for command tools: OMP records
    ``details.exitCode`` only for failures, so a non-error result without one
    is a zero exit and an error result without one (timeout, cancel) is
    ``None``. ``timestamp`` is when the result was appended to the session.
    """

    call_id: str
    tool_name: str
    text: str
    is_error: bool
    details: Mapping[str, object] = field(default_factory=dict)
    artifact_text: str | None = None
    exit_code: int | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class UserTurn:
    """One user-role message on the main branch."""

    entry_id: str
    timestamp: datetime
    text: str
    attribution: Attribution


@dataclass(frozen=True)
class AssistantTurn:
    """One assistant message with its model identity and usage."""

    entry_id: str
    timestamp: datetime
    text: str
    thinking: str
    tool_calls: tuple[ToolCall, ...]
    provider: str
    model: str
    usage: Usage
    stop_reason: str


@dataclass(frozen=True)
class FileMutation:
    """One full-content file change made through the write or edit tool.

    ``path`` is the path exactly as the tool recorded it. ``absolute_path`` is
    that path resolved against the session working directory using the path
    rules of the machine that recorded the session (Windows rules when the
    working directory is a drive path, POSIX rules otherwise). ``content`` is
    the complete file text after the change, or an empty string for a delete.
    ``old_text`` is the complete text before an edit or delete, or ``None``
    for a write.
    """

    order: int
    timestamp: datetime
    kind: MutationKind
    path: str
    absolute_path: str
    content: str
    old_text: str | None
    from_subagent: bool


@dataclass(frozen=True)
class CommandRun:
    """One shell command executed through the bash tool."""

    order: int
    timestamp: datetime
    command: str
    exit_code: int | None
    is_error: bool
    output: str
    from_subagent: bool


Step = UserTurn | AssistantTurn | ToolResult


@dataclass(frozen=True)
class Episode:
    """One user request and everything the agent did until the next one."""

    session: SessionHeader
    index: int
    prompt: str
    started_at: datetime
    ended_at: datetime
    steps: tuple[Step, ...]
    mutations: tuple[FileMutation, ...]
    commands: tuple[CommandRun, ...]
    models: tuple[str, ...]
    usage: Usage
    assistant_turns: int
    tool_calls: int
    ended_by: EpisodeEnd
    unresolved_mutations: int


@dataclass(frozen=True)
class Session:
    """One parsed session: its header and the main-branch steps in order.

    ``mutations`` and ``commands`` share one ``order`` counter over the merged
    parent and subagent timeline. ``exits`` holds the timestamps of
    ``session_exit`` entries on the main branch and ``unresolved`` the
    timestamps of edit results whose file content could not be reconstructed.
    """

    header: SessionHeader
    steps: tuple[Step, ...]
    mutations: tuple[FileMutation, ...]
    commands: tuple[CommandRun, ...]
    torn_lines: int
    subagent_files: tuple[Path, ...]
    exits: tuple[datetime, ...] = ()
    unresolved: tuple[datetime, ...] = ()


@dataclass(frozen=True)
class SessionLoadError:
    """The file is not a readable version 3 session."""

    path: Path
    reason: str


@dataclass(frozen=True)
class _Lines:
    """Parsed JSON objects of one file, title slot removed."""

    header: Mapping[str, object]
    title: str
    entries: tuple[Mapping[str, object], ...]
    torn_lines: int


@dataclass(frozen=True)
class _Timeline:
    """Mutations and commands of one file before merging."""

    mutations: tuple[FileMutation, ...]
    commands: tuple[CommandRun, ...]
    unresolved: tuple[datetime, ...]


@dataclass(frozen=True)
class _Parsed:
    """Main branch of one file before the timeline is derived."""

    header: SessionHeader
    steps: tuple[Step, ...]
    torn_lines: int
    exits: tuple[datetime, ...]


def discover_sessions(root: Path) -> tuple[Path, ...]:
    """Return every top-level session file under ``root``, sorted.

    Subagent transcripts live inside the artifact directory that shares the
    stem of their parent session file and are never returned.
    """
    return tuple(
        sorted(
            path
            for path in root.rglob("*.jsonl")
            if path.is_file() and not path.parent.with_suffix(".jsonl").is_file()
        )
    )


def read_session(path: Path) -> Session | SessionLoadError:
    """Parse one session file and its subagent sidecars."""
    artifact_dir = path.with_suffix("")
    parent = _read_file(path, artifact_dir)
    if isinstance(parent, SessionLoadError):
        return parent
    timelines = [_timeline(parent, from_subagent=False)]
    subagent_files: list[Path] = []
    for sidecar in sorted(artifact_dir.glob("*.jsonl")) if artifact_dir.is_dir() else ():
        child = _read_file(sidecar, artifact_dir)
        if isinstance(child, SessionLoadError):
            continue
        subagent_files.append(sidecar)
        timelines.append(_timeline(child, from_subagent=True))
    merged = _merge_timelines(timelines)
    return Session(
        header=parent.header,
        steps=parent.steps,
        mutations=merged.mutations,
        commands=merged.commands,
        torn_lines=parent.torn_lines,
        subagent_files=tuple(subagent_files),
        exits=parent.exits,
        unresolved=merged.unresolved,
    )


def session_episodes(session: Session) -> tuple[Episode, ...]:
    """Split the main branch at every user-attributed turn."""
    steps = session.steps
    starts = [
        (index, step)
        for index, step in enumerate(steps)
        if isinstance(step, UserTurn) and step.attribution == "user"
    ]
    episodes: list[Episode] = []
    for number, (start, first) in enumerate(starts):
        next_start = starts[number + 1][1].timestamp if number + 1 < len(starts) else None
        ended_at, ended_by = _episode_end(session, first.timestamp, next_start)
        inclusive = ended_by == "end"
        window = [
            step
            for stamp, step in _timed_steps(steps[start:])
            if _in_window(stamp, first.timestamp, ended_at, inclusive)
        ]
        episodes.append(_episode(session, number, first, ended_at, ended_by, window))
    return tuple(episodes)


def clean_prompt(text: str) -> str:
    """Return the user's own words without OMP's injected wrappers."""
    cleaned = _SYSTEM_REMINDER.sub("", text).strip()
    wrapped = _WRAPPER.match(cleaned)
    if wrapped:
        cleaned = wrapped.group(2).strip()
    return _SURPLUS_BLANK_LINES.sub("\n\n", cleaned).strip()


def _episode_end(
    session: Session, started_at: datetime, next_start: datetime | None
) -> tuple[datetime, EpisodeEnd]:
    for exit_at in session.exits:
        if started_at < exit_at and (next_start is None or exit_at < next_start):
            return exit_at, "exit"
    if next_start is not None:
        return next_start, "user"
    last = started_at
    for stamp, _ in _timed_steps(session.steps):
        last = max(last, stamp)
    for item in (*session.mutations, *session.commands):
        last = max(last, item.timestamp)
    return last, "end"


def _episode(
    session: Session,
    index: int,
    first: UserTurn,
    ended_at: datetime,
    ended_by: EpisodeEnd,
    steps: list[Step],
) -> Episode:
    inclusive = ended_by == "end"
    assistant = [step for step in steps if isinstance(step, AssistantTurn)]
    models: list[str] = []
    usage = Usage()
    for turn in assistant:
        name = f"{turn.provider}/{turn.model}"
        if name not in models:
            models.append(name)
        usage = usage + turn.usage
    return Episode(
        session=session.header,
        index=index,
        prompt=clean_prompt(first.text),
        started_at=first.timestamp,
        ended_at=ended_at,
        steps=tuple(steps),
        mutations=tuple(
            m
            for m in session.mutations
            if _in_window(m.timestamp, first.timestamp, ended_at, inclusive)
        ),
        commands=tuple(
            c
            for c in session.commands
            if _in_window(c.timestamp, first.timestamp, ended_at, inclusive)
        ),
        models=tuple(models),
        usage=usage,
        assistant_turns=len(assistant),
        tool_calls=sum(len(turn.tool_calls) for turn in assistant),
        ended_by=ended_by,
        unresolved_mutations=sum(
            _in_window(stamp, first.timestamp, ended_at, inclusive)
            for stamp in session.unresolved
        ),
    )


def _in_window(stamp: datetime, start: datetime, end: datetime, inclusive: bool) -> bool:
    return start <= stamp and (stamp <= end if inclusive else stamp < end)


def _timed_steps(steps: Iterable[Step]) -> list[tuple[datetime, Step]]:
    """Pair each step with its timestamp; a result without one follows its call."""
    timed: list[tuple[datetime, Step]] = []
    last: datetime | None = None
    for step in steps:
        stamp = step.timestamp if step.timestamp is not None else last
        if stamp is None:
            continue
        timed.append((stamp, step))
        last = stamp
    return timed


def _read_file(path: Path, artifact_dir: Path) -> _Parsed | SessionLoadError:
    lines = _load_lines(path)
    if isinstance(lines, SessionLoadError):
        return lines
    header = _header(lines, path, artifact_dir)
    if isinstance(header, SessionLoadError):
        return header
    steps: list[Step] = []
    exits: list[datetime] = []
    previous = header.started_at
    for entry in _main_branch(lines.entries):
        stamp = _entry_time(entry, previous)
        previous = stamp
        kind = entry.get("type")
        if kind == "message":
            step = _step(entry, stamp, artifact_dir)
            if step is not None:
                steps.append(step)
        elif kind == "custom" and entry.get("customType") == "session_exit":
            exits.append(stamp)
    return _Parsed(header, tuple(steps), lines.torn_lines, tuple(exits))


def _load_lines(path: Path) -> _Lines | SessionLoadError:
    try:
        size = path.stat().st_size
    except OSError as error:
        return SessionLoadError(path, f"cannot stat: {error}")
    if size > MAX_SESSION_BYTES:
        return SessionLoadError(path, f"file exceeds {MAX_SESSION_BYTES} bytes")
    objects: list[Mapping[str, object]] = []
    torn = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                torn += 1
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
            else:
                torn += 1
    title = ""
    if objects and objects[0].get("type") == _TITLE_SLOT:
        title = _text(objects[0].get("title"))
        objects = objects[1:]
    if not objects:
        return SessionLoadError(path, "missing session header")
    entries = tuple(e for e in objects[1:] if isinstance(e.get("id"), str))
    return _Lines(objects[0], title, entries, torn)


def _header(lines: _Lines, path: Path, artifact_dir: Path) -> SessionHeader | SessionLoadError:
    raw = lines.header
    if raw.get("type") != "session" or not isinstance(raw.get("id"), str):
        return SessionLoadError(path, "missing session header")
    if raw.get("version") != SESSION_VERSION:
        return SessionLoadError(path, f"unsupported session version {raw.get('version')!r}")
    started = _iso_time(raw.get("timestamp"))
    if started is None:
        return SessionLoadError(path, "header has no timestamp")
    parent = raw.get("parentSession")
    return SessionHeader(
        id=_text(raw.get("id")),
        cwd=_text(raw.get("cwd")),
        started_at=started,
        title=lines.title or _text(raw.get("title")),
        path=path,
        artifact_dir=artifact_dir,
        parent_session=parent if isinstance(parent, str) else None,
    )


def _main_branch(entries: tuple[Mapping[str, object], ...]) -> list[Mapping[str, object]]:
    """Walk ``parentId`` from the last appended entry to its root."""
    by_id = {_text(entry["id"]): entry for entry in entries}
    if not entries:
        return []
    branch: list[Mapping[str, object]] = []
    seen: set[str] = set()
    current: object = entries[-1]["id"]
    while isinstance(current, str) and current in by_id and current not in seen:
        seen.add(current)
        entry = by_id[current]
        branch.append(entry)
        current = entry.get("parentId")
    branch.reverse()
    return branch


def _step(entry: Mapping[str, object], stamp: datetime, artifact_dir: Path) -> Step | None:
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    entry_id = _text(entry.get("id"))
    if role == "user":
        return UserTurn(
            entry_id=entry_id,
            timestamp=stamp,
            text=_content_text(message.get("content")),
            attribution=_attribution(message.get("attribution")),
        )
    if role == "assistant":
        return _assistant_turn(entry_id, stamp, message)
    if role == "toolResult":
        return _tool_result(message, stamp, artifact_dir)
    return None


def _assistant_turn(entry_id: str, stamp: datetime, message: Mapping[str, object]) -> AssistantTurn:
    content = message.get("content")
    blocks = content if isinstance(content, list) else []
    texts: list[str] = []
    thoughts: list[str] = []
    calls: list[ToolCall] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(_text(block.get("text")))
        elif kind == "thinking":
            thoughts.append(_text(block.get("thinking")))
        elif kind == "toolCall":
            arguments = block.get("arguments")
            calls.append(
                ToolCall(
                    call_id=_text(block.get("id")),
                    name=_text(block.get("name")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    intent=_text(block.get("intent")),
                )
            )
    return AssistantTurn(
        entry_id=entry_id,
        timestamp=stamp,
        text="\n".join(texts),
        thinking="\n".join(thoughts),
        tool_calls=tuple(calls),
        provider=_text(message.get("provider")),
        model=_text(message.get("model")),
        usage=_usage(message.get("usage")),
        stop_reason=_text(message.get("stopReason")),
    )


def _tool_result(message: Mapping[str, object], stamp: datetime, artifact_dir: Path) -> ToolResult:
    details = message.get("details")
    details = details if isinstance(details, dict) else {}
    tool_name = _text(message.get("toolName"))
    text = _content_text(message.get("content"))
    is_error = bool(message.get("isError"))
    exit_code = details.get("exitCode")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        exit_code = None if is_error else 0
    return ToolResult(
        call_id=_text(message.get("toolCallId")),
        tool_name=tool_name,
        text=text,
        is_error=is_error,
        details=details,
        artifact_text=_artifact_text(artifact_dir, tool_name, text),
        exit_code=exit_code,
        timestamp=stamp,
    )


def _artifact_text(artifact_dir: Path, tool_name: str, text: str) -> str | None:
    reference = _ARTIFACT_REF.search(text)
    if reference is None:
        return None
    number = reference.group(1)
    # OMP keeps the unfiltered stream as ``-original`` when the shown text was
    # a filtered projection; either file is the full output.
    for name in (f"{number}.{tool_name}.log", f"{number}.{tool_name}-original.log"):
        candidate = artifact_dir / name
        try:
            if not candidate.is_file() or candidate.stat().st_size > MAX_ARTIFACT_BYTES:
                continue
            return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return None


def _timeline(parsed: _Parsed, *, from_subagent: bool) -> _Timeline:
    """Derive file mutations and command runs from the joined tool calls."""
    calls: dict[str, ToolCall] = {}
    mutations: list[FileMutation] = []
    commands: list[CommandRun] = []
    unresolved: list[datetime] = []
    cwd = parsed.header.cwd
    for stamp, step in _timed_steps(parsed.steps):
        if isinstance(step, AssistantTurn):
            calls.update((call.call_id, call) for call in step.tool_calls)
            continue
        if not isinstance(step, ToolResult):
            continue
        call = calls.get(step.call_id)
        if step.tool_name == "bash" and call is not None:
            commands.append(_command_run(step, call, stamp, from_subagent))
            continue
        # A failed write or edit changed nothing on disk.
        if step.is_error:
            continue
        changes: list[FileMutation | None] = []
        if step.tool_name == "write" and call is not None:
            changes.append(_write_mutation(call, stamp, cwd, from_subagent))
        elif step.tool_name == "edit":
            changes.extend(
                _edit_mutation(change, stamp, cwd, from_subagent)
                for change in _edit_changes(step.details)
            )
        for mutation in changes:
            if mutation is None:
                unresolved.append(stamp)
            else:
                mutations.append(mutation)
    return _Timeline(tuple(mutations), tuple(commands), tuple(unresolved))


def _command_run(step: ToolResult, call: ToolCall, stamp: datetime, from_subagent: bool) -> CommandRun:
    return CommandRun(
        order=0,
        timestamp=stamp,
        command=_text(call.arguments.get("command")),
        exit_code=step.exit_code,
        is_error=step.is_error,
        output=step.artifact_text if step.artifact_text is not None else step.text,
        from_subagent=from_subagent,
    )


def _write_mutation(call: ToolCall, stamp: datetime, cwd: str, from_subagent: bool) -> FileMutation | None:
    path = call.arguments.get("path")
    content = call.arguments.get("content")
    if not isinstance(path, str) or not isinstance(content, str) or TRUNCATION_MARK in content:
        return None
    return FileMutation(
        order=0,
        timestamp=stamp,
        kind="write",
        path=path,
        absolute_path=_absolute_path(cwd, path),
        content=content,
        old_text=None,
        from_subagent=from_subagent,
    )


def _edit_changes(details: Mapping[str, object]) -> list[Mapping[str, object]]:
    """One edit result describes one file, or several under ``perFileResults``."""
    per_file = details.get("perFileResults")
    if isinstance(per_file, list):
        return [item for item in per_file if isinstance(item, dict)]
    return [details]


def _edit_mutation(
    change: Mapping[str, object], stamp: datetime, cwd: str, from_subagent: bool
) -> FileMutation | None:
    # Older OMP builds omit ``op`` on plain updates; pruned snapshots
    # (``snapshotsPruned``) omit the texts and cannot be reconstructed.
    op = change.get("op", "update")
    path = change.get("path")
    old_text = change.get("oldText")
    new_text = change.get("newText")
    kind = _EDIT_KINDS.get(op) if isinstance(op, str) else None
    if kind is None or not isinstance(path, str):
        return None
    if kind == "delete":
        new_text = ""
    if kind == "write":
        old_text = None
    if not isinstance(new_text, str) or not isinstance(old_text, str | None):
        return None
    if TRUNCATION_MARK in new_text or (old_text is not None and TRUNCATION_MARK in old_text):
        return None
    return FileMutation(
        order=0,
        timestamp=stamp,
        kind=kind,
        path=path,
        absolute_path=_absolute_path(cwd, path),
        content=new_text,
        old_text=old_text,
        from_subagent=from_subagent,
    )


def _merge_timelines(timelines: list[_Timeline]) -> _Timeline:
    """Interleave parent and subagent items by time and number them."""
    items: list[tuple[datetime, int, int, FileMutation | CommandRun]] = []
    for rank, timeline in enumerate(timelines):
        for position, item in enumerate((*timeline.mutations, *timeline.commands)):
            items.append((item.timestamp, rank, position, item))
    items.sort(key=lambda entry: entry[:3])
    mutations: list[FileMutation] = []
    commands: list[CommandRun] = []
    for order, (_, _, _, item) in enumerate(items):
        if isinstance(item, FileMutation):
            mutations.append(replace(item, order=order))
        else:
            commands.append(replace(item, order=order))
    unresolved = sorted(stamp for timeline in timelines for stamp in timeline.unresolved)
    return _Timeline(tuple(mutations), tuple(commands), tuple(unresolved))


def _absolute_path(cwd: str, path: str) -> str:
    flavour = ntpath if _DRIVE_PATH.match(cwd) else posixpath
    return flavour.normpath(flavour.join(cwd, path))


def _entry_time(entry: Mapping[str, object], fallback: datetime) -> datetime:
    stamp = _iso_time(entry.get("timestamp"))
    if stamp is not None:
        return stamp
    message = entry.get("message")
    epoch = message.get("timestamp") if isinstance(message, dict) else None
    if isinstance(epoch, int | float) and not isinstance(epoch, bool):
        return datetime.fromtimestamp(epoch / 1000, tz=UTC)
    return fallback


def _iso_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _usage(raw: object) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    cost = raw.get("cost")
    total_cost = cost.get("total") if isinstance(cost, dict) else cost
    return Usage(
        input_tokens=_count(raw.get("input")),
        output_tokens=_count(raw.get("output")),
        cache_read_tokens=_count(raw.get("cacheRead")),
        cache_write_tokens=_count(raw.get("cacheWrite")),
        total_tokens=_count(raw.get("totalTokens")),
        cost=float(total_cost) if isinstance(total_cost, int | float) else 0.0,
    )


def _count(value: object) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def _attribution(value: object) -> Attribution:
    if value in ("user", "agent", "system"):
        return value
    return "unknown"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        _text(block.get("text"))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
