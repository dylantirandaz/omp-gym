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


def _scan_session(session_file: Path) -> dict | None:
    """Extract failure evidence from one session."""
    trajectory = parse_session(session_file)
    corrections = 0
    first_user = None
    test_command = None
    test_failed_late = False
    writes: dict[str, str] = {}
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
                        test_command = found.group(1).strip()
                elif call.name == "write":
                    path = str(call.arguments.get("path", ""))
                    content = str(call.arguments.get("content", ""))
                    if path and content and "://" not in path:
                        writes[path] = content
        elif isinstance(step, ToolResultStep):
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
        for path, content in evidence["writes"].items():
            target = workspace / Path(path).name
            target.write_text(content)
        prompt = (
            evidence["prompt"]
            + "\n\nRun `"
            + evidence["test_command"]
            + "` to confirm the fix.\n"
        )
        test_target = re.search(r"(test_\S+\.py)", evidence["test_command"])
        fidelity = "partial"
        if test_target is not None:
            target_name = Path(test_target.group(1)).name
            if any(
                Path(path).name == target_name
                for path in evidence["writes"]
            ):
                fidelity = "complete"
        (task_dir / "task.toml").write_text(
            f'prompt = """\n{prompt}"""\n'
            f'test_command = ["sh", "-c", "{evidence["test_command"]}"]\n'
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
                test_command=evidence["test_command"],
                fidelity=fidelity,
                task_dir=str(task_dir),
            )
        )
    return minted
