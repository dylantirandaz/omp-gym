"""Mint tasks from sessions where the agent failed.

A session shows a failure when the user corrected the agent or a
test run failed late in the session. A candidate becomes a task
when the session contains enough material to rebuild one: the
first user request as the prompt, full file contents from write
tool calls as the workspace, and the failing test command found in
the session's bash history.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .trajectory import (
    AssistantStep,
    ToolResultStep,
    UserStep,
    parse_session,
)

_CORRECTION = re.compile(
    r"\b(no,? that'?s (wrong|not)|wrong|you broke|still (fails|failing|broken)"
    r"|doesn'?t work|not what i|revert|undo that|stop)\b",
    re.IGNORECASE,
)
_TEST_COMMAND = re.compile(
    r"(?:^|\s)(pytest\b[^\n]*|python3?\s+test_\S+|npm\s+test[^\n]*"
    r"|cargo\s+test[^\n]*|go\s+test[^\n]*)"
)
_TEST_FAILED = re.compile(
    r"\b(failed|FAILED|FAILURES|panic:|AssertionError)\b"
)


@dataclass(frozen=True)
class MintedTask:
    """One task minted from a failed session."""

    name: str
    source_session: str
    signals: int
    test_command: str
    fidelity: str
    task_dir: str


def _session_cwd(session_file: Path) -> Path | None:
    """Read the working directory recorded in the session header."""
    for line in session_file.read_text().splitlines()[:20]:
        if '"type": "session"' in line or '"type":"session"' in line:
            import json as _json

            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            cwd = entry.get("cwd")
            if isinstance(cwd, str):
                return Path(cwd)
    return None


def _relative_write_path(path: str, cwd: Path | None) -> str:
    """Resolve a write path to a workspace-relative location.

    Absolute paths under the session's cwd become relative to it.
    Other absolute paths keep only their basename. Relative paths
    pass through unchanged.
    """
    candidate = Path(path)
    if candidate.is_absolute() and cwd is not None:
        try:
            return str(candidate.relative_to(cwd))
        except ValueError:
            return candidate.name
    if candidate.is_absolute():
        return candidate.name
    return path


def _clean_read_payload(text: str) -> str | None:
    """Recover file content from a read tool payload.

    Stored reads carry three wrappers, alone or combined: a CLI
    header block (URL:, Content-Type: ...), line-number prefixes
    ("159: ..."), and a trailing "[Showing lines ...]" marker.
    Returns None when the payload is not file content (directory
    listing, error page, empty result).
    """
    if not text.strip():
        return None
    if text.strip() == "(empty directory)":
        return None
    body = text
    # Drop a CLI header block if present (URL/Content-Type/Method).
    if "\n---\n" in body:
        head, _, rest = body.partition("\n---\n")
        head_lines = [line for line in head.splitlines() if line.strip()]
        if head_lines and all(
            line.split(":", 1)[0] in ("URL", "Content-Type", "Method", "Path")
            for line in head_lines
        ):
            body = rest.lstrip()
    # Drop trailing truncation markers.
    body = re.sub(
        r"\n?\[Showing lines [^\]]+\]\s*$", "", body.rstrip()
    )
    body = re.sub(
        r"\n?\[Read [^\]]+\]\s*$", "", body.rstrip()
    )
    lines = body.splitlines()
    numbered = [line for line in lines if re.match(r"^\d+: ", line)]
    if lines and len(numbered) >= max(2, int(0.8 * len(lines))):
        lines = [re.sub(r"^\d+: ?", "", line) for line in lines]
        body = "\n".join(lines)
    if not body.strip():
        return None
    return body


def _scan_session(session_file: Path) -> dict | None:
    """Extract failure evidence from one session."""
    trajectory = parse_session(session_file)
    cwd = _session_cwd(session_file)
    corrections = 0
    first_user = None
    test_command = None
    test_failed_late = False
    writes: dict[str, str] = {}
    reads: dict[str, str] = {}
    pending_reads: dict[str, str] = {}
    latest: dict[str, str] = {}
    steps = list(trajectory.steps)
    for index, step in enumerate(steps):
        if isinstance(step, UserStep):
            if first_user is None:
                first_user = step.text
            if step.text and not step.text.startswith("<tool_response"):
                corrections += len(_CORRECTION.findall(step.text))
        elif isinstance(step, AssistantStep):
            for call in step.tool_calls:
                if call.name == "bash":
                    command = str(call.arguments.get("command", ""))
                    found = _TEST_COMMAND.search(command)
                    if found:
                        # The last match is the command the session
                        # ended on, which matches the final file state.
                        test_command = found.group(1).strip()
                elif call.name == "write":
                    path = str(call.arguments.get("path", ""))
                    content = str(call.arguments.get("content", ""))
                    if path and content and "://" not in path:
                        rel = _relative_write_path(path, cwd)
                        writes[rel] = content
                        latest[rel] = content
                elif call.name == "read":
                    path = str(call.arguments.get("path", ""))
                    if path and "://" not in path:
                        pending_reads[call.call_id] = (
                            _relative_write_path(path, cwd)
                        )
        elif isinstance(step, ToolResultStep):
            if step.tool_name == "read" and step.call_id in pending_reads:
                rel = pending_reads[step.call_id]
                cleaned = (
                    None
                    if step.is_error
                    else _clean_read_payload(step.text)
                )
                if (
                    cleaned is not None
                    and rel not in reads
                    and rel not in writes
                ):
                    reads[rel] = cleaned
                if cleaned is not None:
                    latest[rel] = cleaned
            if (
                step.tool_name == "bash"
                and index > len(steps) * 0.6
                and _TEST_FAILED.search(step.text)
            ):
                test_failed_late = True

    signals = corrections + (1 if test_failed_late else 0)
    if not first_user or signals < 2:
        return None
    if test_command is None or not writes:
        return None
    return {
        "prompt": first_user[:4000],
        "signals": signals,
        "test_command": test_command,
        "writes": writes,
        "reads": reads,
        "latest": latest,
    }


def mint_tasks(
    sessions_root: Path,
    out_dir: Path,
    limit: int,
) -> list[MintedTask]:
    """Scan sessions and write minted tasks. Returns what was minted."""
    minted: list[MintedTask] = []
    stamp = time.strftime("%Y%m%d")
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        if len(minted) >= limit:
            break
        evidence = _scan_session(session_file)
        if evidence is None:
            continue
        name = f"minted-{stamp}-{len(minted) + 1}"
        task_dir = out_dir / name
        workspace = task_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        for path, content in evidence["latest"].items():
            if not path or path in (".", "/", "\\"):
                continue
            target = workspace / path
            if target.is_dir():
                continue
            # A stored path can disagree with a deeper path about
            # which one is the file; the deeper path wins.
            ancestor_file = None
            cursor = target.parent
            while cursor != workspace and cursor != cursor.parent:
                if cursor.is_file():
                    ancestor_file = cursor
                    break
                cursor = cursor.parent
            if ancestor_file is not None:
                ancestor_file.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        test_command = evidence["test_command"]
        test_target = re.search(
            r"(\S*test_\S+\.py)", test_command
        )
        selector = re.search(r"-k\s+'([^']+)'", test_command)
        if selector is not None and test_target is not None:
            target_name = Path(test_target.group(1)).name
            target_content = evidence["latest"].get(
                test_target.group(1)
            ) or next(
                (
                    content
                    for path, content in evidence["latest"].items()
                    if Path(path).name == target_name
                ),
                None,
            )
            if target_content is not None:
                names = selector.group(1).split(" or ")
                missing = [
                    name
                    for name in names
                    if f"def test_{name}" not in target_content
                    and name not in target_content
                ]
                if missing:
                    test_command = re.sub(
                        r"\s+-k\s+'[^']+'", "", test_command
                    )
        prompt = (
            evidence["prompt"]
            + "\n\nRun `"
            + test_command
            + "` to confirm the fix.\n"
        )
        fidelity = "partial"
        if test_target is not None:
            expected = test_target.group(1)
            if (
                expected in evidence["writes"]
                or expected in evidence["reads"]
                or expected in evidence["latest"]
            ):
                fidelity = "complete"
        (task_dir / "task.toml").write_text(
            f'prompt = """\n{prompt}"""\n'
            f'test_command = ["sh", "-c", "{test_command}"]\n'
            f'fidelity = "{fidelity}"\n'
        )
        (task_dir / "SOURCE.md").write_text(
            f"source session: {session_file}\n"
            f"failure signals: {evidence['signals']}\n"
        )
        minted.append(
            MintedTask(
                name=name,
                source_session=str(session_file),
                signals=evidence["signals"],
                test_command=test_command,
                fidelity=fidelity,
                task_dir=str(task_dir),
            )
        )
    return minted
