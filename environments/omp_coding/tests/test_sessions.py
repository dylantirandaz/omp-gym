from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from omp_coding.sessions import (
    AssistantTurn,
    SessionLoadError,
    ToolResult,
    UserTurn,
    clean_prompt,
    discover_sessions,
    read_session,
    session_episodes,
)

WINDOWS_CWD = "C:\\Users\\x\\repo"
POSIX_CWD = "/home/x/repo"
SESSION_ID = "01a06507-382c-7426-8a95-29d5af527b2a"
SESSION_STEM = "2026-09-03T02-09-27-084Z_" + SESSION_ID
BASE_MS = 1788228000000
SECOND_MS = 1000
SPILL_TEXT = "line one\nline two\nfull raw output\n"


def _iso(ms: int) -> str:
    stamp = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def _when(seconds: int) -> datetime:
    return datetime.fromtimestamp((BASE_MS + seconds * SECOND_MS) / 1000, tz=UTC)


def _entry(kind: str, entry_id: str, parent: str | None, seconds: int, **fields: object) -> dict[str, object]:
    return {
        "type": kind,
        "id": entry_id,
        "parentId": parent,
        "timestamp": _iso(BASE_MS + seconds * SECOND_MS),
        **fields,
    }


def _user(entry_id: str, parent: str | None, seconds: int, text: str, attribution: str = "user") -> dict[str, object]:
    message = {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "attribution": attribution,
        "timestamp": BASE_MS + seconds * SECOND_MS,
    }
    return _entry("message", entry_id, parent, seconds, message=message)


def _assistant(
    entry_id: str,
    parent: str | None,
    seconds: int,
    calls: list[dict[str, object]],
    *,
    model: str = "claude-fable-5",
    provider: str = "anthropic",
    tokens: int = 10,
) -> dict[str, object]:
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "working"},
            *calls,
        ],
        "api": "anthropic-messages",
        "provider": provider,
        "model": model,
        "usage": {
            "input": tokens,
            "output": tokens * 2,
            "cacheRead": 1,
            "cacheWrite": 2,
            "totalTokens": tokens * 3 + 3,
            "cost": {"input": 0.0, "output": 0.0, "total": 0.5},
        },
        "stopReason": "toolUse" if calls else "stop",
        "timestamp": BASE_MS + seconds * SECOND_MS,
    }
    return _entry("message", entry_id, parent, seconds, message=message)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": arguments, "intent": "doing " + name}


def _start(entry_id: str, parent: str | None, seconds: int, call_id: str, name: str) -> dict[str, object]:
    data = {"toolCallId": call_id, "toolName": name, "startedAt": _iso(BASE_MS + seconds * SECOND_MS)}
    return _entry("custom", entry_id, parent, seconds, customType="tool_execution_start", data=data)


def _result(
    entry_id: str,
    parent: str | None,
    seconds: int,
    call_id: str,
    name: str,
    text: str,
    *,
    details: dict[str, object] | None = None,
    is_error: bool = False,
) -> dict[str, object]:
    message = {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": name,
        "content": [{"type": "text", "text": text}],
        "details": details or {},
        "isError": is_error,
        "timestamp": BASE_MS + seconds * SECOND_MS,
    }
    return _entry("message", entry_id, parent, seconds, message=message)


def _exit(entry_id: str, parent: str | None, seconds: int) -> dict[str, object]:
    data = {"reason": "sighup", "kind": "signal", "recordedAt": _iso(BASE_MS + seconds * SECOND_MS)}
    return _entry("custom", entry_id, parent, seconds, customType="session_exit", data=data)


def _header(session_id: str, cwd: str, seconds: int = 0) -> dict[str, object]:
    return {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": _iso(BASE_MS + seconds * SECOND_MS),
        "cwd": cwd,
    }


def _title_slot() -> dict[str, object]:
    return {"type": "title", "v": 1, "title": "Fix the parser", "source": "auto", "updatedAt": _iso(BASE_MS), "pad": " " * 40}


def _main_entries() -> list[dict[str, object]]:
    """Episode one: read + bash + write; episode two: edit, then a session exit."""
    return [
        _entry("model_change", "e0000001", None, 1, model="anthropic/claude-fable-5", resolvedModelIsFallback=False),
        _user("e0000002", "e0000001", 2, "<system-reminder>\ndate: today\n</system-reminder>\n\nFix the parser\n\n\n\nplease"),
        _assistant(
            "e0000003",
            "e0000002",
            3,
            [
                _call("c-read", "read", {"path": "src/parser.py"}),
                _call("c-bash", "bash", {"command": "pytest -q", "timeout": 60}),
            ],
        ),
        _start("e0000004", "e0000003", 4, "c-read", "read"),
        _result("e0000005", "e0000004", 5, "c-read", "read", "[src/parser.py#AAAA]\n1:x = 1"),
        _start("e0000006", "e0000005", 6, "c-bash", "bash"),
        _result(
            "e0000007",
            "e0000006",
            7,
            "c-bash",
            "bash",
            "FAILED tests/test_parser.py::test_x\n[raw output: artifact://1]\n\nWall time: 1.0 seconds",
            details={"exitCode": 1, "wallTimeMs": 1000.0, "timeoutSeconds": 60},
            is_error=True,
        ),
        _assistant(
            "e0000008",
            "e0000007",
            8,
            [_call("c-write", "write", {"path": "src/parser.py", "content": "x = 2\n"})],
            model="gpt-5.6-sol",
            provider="openai-codex",
        ),
        _start("e0000009", "e0000008", 9, "c-write", "write"),
        _result(
            "e0000010", "e0000009", 10, "c-write", "write", "Successfully wrote 6 bytes", details={"resolvedPath": "src/parser.py"}
        ),
        _assistant("e0000011", "e0000010", 11, [_call("c-bash2", "bash", {"command": "pytest -q"})]),
        _result("e0000012", "e0000011", 12, "c-bash2", "bash", "1 passed", details={"wallTimeMs": 900.0, "timeoutSeconds": 60}),
        _assistant("e0000013", "e0000012", 13, []),
        _user("e0000014", "e0000013", 20, "Now rename the helper"),
        _assistant("e0000015", "e0000014", 21, [_call("c-edit", "edit", {"input": "*** Begin Patch\n*** End Patch\n"})]),
        _start("e0000016", "e0000015", 22, "c-edit", "edit"),
        _result(
            "e0000017",
            "e0000016",
            23,
            "c-edit",
            "edit",
            "[src/helper.py#BBBB]\n1:def helper_two(): ...",
            details={
                "diff": "-1|def helper(): ...\n+1|def helper_two(): ...",
                "firstChangedLine": 1,
                "op": "update",
                "oldText": "def helper(): ...\n",
                "newText": "def helper_two(): ...\n",
                "path": "src/helper.py",
            },
        ),
        _assistant("e0000018", "e0000017", 24, [_call("c-task", "task", {"tasks": [{"task": "help"}]})]),
        _result("e0000019", "e0000018", 30, "c-task", "task", "done"),
        _assistant("e0000020", "e0000019", 31, []),
        _exit("e0000021", "e0000020", 40),
    ]


def _abandoned_entries() -> list[dict[str, object]]:
    """A branch off the first assistant turn that the user navigated away from."""
    return [
        _user("a0000001", "e0000003", 14, "Abandoned prompt"),
        _assistant("a0000002", "a0000001", 15, [_call("c-lost", "bash", {"command": "rm -rf build"})]),
        _result("a0000003", "a0000002", 16, "c-lost", "bash", "gone", details={"exitCode": 2}, is_error=True),
    ]


def _sidecar_entries() -> list[dict[str, object]]:
    """A subagent transcript with one edit inside the second episode."""
    return [
        _entry("session_init", "s0000001", None, 25, systemPrompt="sys", task="help", tools=["read", "edit"]),
        _user("s0000002", "s0000001", 25, "help", attribution="agent"),
        _assistant("s0000003", "s0000002", 26, [_call("s-edit", "edit", {"input": "patch"})]),
        _result(
            "s0000004",
            "s0000003",
            27,
            "s-edit",
            "edit",
            "Deleted src/old.py",
            details={"diff": "", "op": "delete", "oldText": "old\n", "path": "src/old.py"},
        ),
        _assistant("s0000005", "s0000004", 28, []),
    ]


def _write_jsonl(path: Path, objects: list[dict[str, object]], *, torn: int = 0) -> None:
    lines = [json.dumps(obj) for obj in objects]
    # A torn tail is what a crash mid-append leaves behind.
    lines.extend('{"type":"message","id":"zz' for _ in range(torn))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_session(root: Path, cwd: str, *, torn: int = 0) -> Path:
    bucket = root / "-repo"
    path = bucket / f"{SESSION_STEM}.jsonl"
    # The abandoned branch is appended before the main branch resumes, so the
    # leaf is still the last main-branch entry.
    main = _main_entries()
    objects = [_title_slot(), _header(SESSION_ID, cwd), *main[:8], *_abandoned_entries(), *main[8:]]
    _write_jsonl(path, objects, torn=torn)
    artifact_dir = bucket / SESSION_STEM
    artifact_dir.mkdir()
    (artifact_dir / "1.bash.log").write_text(SPILL_TEXT, encoding="utf-8")
    _write_jsonl(artifact_dir / "Worker.jsonl", [_title_slot(), _header("sub-1", cwd, 25), *_sidecar_entries()])
    return path


class ReadSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _session(self, cwd: str = WINDOWS_CWD, *, torn: int = 0):
        session = read_session(_write_session(self.root, cwd, torn=torn))
        assert not isinstance(session, SessionLoadError), session
        return session

    def test_main_branch_drops_abandoned_entries(self) -> None:
        session = self._session()
        ids = [step.entry_id for step in session.steps if isinstance(step, UserTurn)]
        assert ids == ["e0000002", "e0000014"]
        assert all(step.call_id != "c-lost" for step in session.steps if isinstance(step, ToolResult))
        assert session.header.title == "Fix the parser"
        assert session.header.started_at == _when(0)

    def test_tool_results_carry_exit_codes_and_artifacts(self) -> None:
        session = self._session()
        results = {step.call_id: step for step in session.steps if isinstance(step, ToolResult)}
        assert results["c-bash"].exit_code == 1
        assert results["c-bash"].is_error
        assert results["c-bash"].artifact_text == SPILL_TEXT
        assert results["c-bash2"].exit_code == 0
        assert results["c-bash2"].artifact_text is None
        assert results["c-read"].timestamp == _when(5)

    def test_commands_expand_spills(self) -> None:
        session = self._session()
        assert [run.command for run in session.commands] == ["pytest -q", "pytest -q"]
        first, second = session.commands
        assert first.output == SPILL_TEXT
        assert second.output == "1 passed"
        assert (first.exit_code, second.exit_code) == (1, 0)

    def test_mutations_use_windows_paths_for_drive_cwd(self) -> None:
        session = self._session(WINDOWS_CWD)
        kinds = [(m.kind, m.path, m.from_subagent) for m in session.mutations]
        assert kinds == [
            ("write", "src/parser.py", False),
            ("edit", "src/helper.py", False),
            ("delete", "src/old.py", True),
        ]
        write, edit, delete = session.mutations
        assert write.absolute_path == "C:\\Users\\x\\repo\\src\\parser.py"
        assert write.content == "x = 2\n" and write.old_text is None
        assert edit.old_text == "def helper(): ...\n" and edit.content == "def helper_two(): ...\n"
        assert delete.content == "" and delete.old_text == "old\n"

    def test_mutations_use_posix_paths_for_posix_cwd(self) -> None:
        session = self._session(POSIX_CWD)
        assert session.mutations[0].absolute_path == "/home/x/repo/src/parser.py"

    def test_timeline_orders_parent_and_subagent_items_together(self) -> None:
        session = self._session()
        items = sorted((*session.mutations, *session.commands), key=lambda item: item.order)
        assert [item.order for item in items] == list(range(len(items)))
        assert [item.timestamp for item in items] == sorted(item.timestamp for item in items)
        subagent = [m for m in session.mutations if m.from_subagent]
        assert len(subagent) == 1 and subagent[0].timestamp == _when(27)
        assert session.subagent_files == (session.header.artifact_dir / "Worker.jsonl",)

    def test_torn_lines_are_counted(self) -> None:
        session = self._session(torn=1)
        assert session.torn_lines == 1
        assert len(session.steps) == 15

    def test_version_two_header_is_rejected(self) -> None:
        path = self.root / "-repo" / "old.jsonl"
        header = {**_header("old", WINDOWS_CWD), "version": 2}
        _write_jsonl(path, [header, _user("e0000001", None, 1, "hi")])
        result = read_session(path)
        assert isinstance(result, SessionLoadError)
        assert "version" in result.reason

    def test_truncated_content_is_unresolved(self) -> None:
        path = self.root / "-repo" / "trunc.jsonl"
        content = "x" * 10 + "[Session persistence truncated large content]"
        objects = [
            _header("trunc", POSIX_CWD),
            _user("e0000001", None, 1, "hi"),
            _assistant("e0000002", "e0000001", 2, [_call("c-w", "write", {"path": "a.py", "content": content})]),
            _result("e0000003", "e0000002", 3, "c-w", "write", "ok"),
            _assistant("e0000004", "e0000003", 4, [_call("c-e", "edit", {"input": "p"})]),
            _result(
                "e0000005",
                "e0000004",
                5,
                "c-e",
                "edit",
                "ok",
                details={"diff": "", "firstChangedLine": 1, "op": "update", "path": "b.py", "snapshotsPruned": True},
            ),
        ]
        _write_jsonl(path, objects)
        session = read_session(path)
        assert not isinstance(session, SessionLoadError)
        assert session.mutations == ()
        assert session.unresolved == (_when(3), _when(5))
        assert session_episodes(session)[0].unresolved_mutations == 2

    def test_multi_file_edit_yields_one_mutation_per_file(self) -> None:
        path = self.root / "-repo" / "multi.jsonl"
        per_file = [
            {"path": "a.py", "diff": "", "firstChangedLine": 1, "op": "update", "oldText": "1\n", "newText": "2\n"},
            {"path": "b.py", "diff": "", "firstChangedLine": 1, "oldText": "3\n", "newText": "4\n"},
        ]
        objects = [
            _header("multi", POSIX_CWD),
            _user("e0000001", None, 1, "hi"),
            _assistant("e0000002", "e0000001", 2, [_call("c-e", "edit", {"input": "p"})]),
            _result("e0000003", "e0000002", 3, "c-e", "edit", "ok", details={"perFileResults": per_file, "diff": ""}),
        ]
        _write_jsonl(path, objects)
        session = read_session(path)
        assert not isinstance(session, SessionLoadError)
        assert [(m.path, m.content) for m in session.mutations] == [("a.py", "2\n"), ("b.py", "4\n")]


class EpisodesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        session = read_session(_write_session(self.root, WINDOWS_CWD))
        assert not isinstance(session, SessionLoadError), session
        self.session = session
        self.episodes = session_episodes(session)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_boundaries_and_endings(self) -> None:
        first, second = self.episodes
        assert (first.index, second.index) == (0, 1)
        assert first.prompt == "Fix the parser\n\nplease"
        assert (first.started_at, first.ended_at, first.ended_by) == (_when(2), _when(20), "user")
        assert (second.started_at, second.ended_at, second.ended_by) == (_when(20), _when(40), "exit")
        assert second.prompt == "Now rename the helper"

    def test_window_contents(self) -> None:
        first, second = self.episodes
        assert [m.path for m in first.mutations] == ["src/parser.py"]
        assert [c.exit_code for c in first.commands] == [1, 0]
        assert [(m.path, m.from_subagent) for m in second.mutations] == [("src/helper.py", False), ("src/old.py", True)]
        assert second.commands == ()
        assert isinstance(first.steps[0], UserTurn) and isinstance(second.steps[0], UserTurn)
        assert sum(isinstance(step, AssistantTurn) for step in first.steps) == first.assistant_turns

    def test_usage_and_models(self) -> None:
        first, second = self.episodes
        assert first.models == ("anthropic/claude-fable-5", "openai-codex/gpt-5.6-sol")
        assert first.assistant_turns == 4
        assert first.tool_calls == 4
        assert first.usage.input_tokens == 40
        assert first.usage.output_tokens == 80
        assert first.usage.cache_read_tokens == 4
        assert first.usage.total_tokens == 132
        assert first.usage.cost == 2.0
        assert second.models == ("anthropic/claude-fable-5",)
        assert second.tool_calls == 2

    def test_end_of_branch_without_exit(self) -> None:
        path = self.root / "-repo" / "open.jsonl"
        objects = [
            _header("open", POSIX_CWD),
            _user("e0000001", None, 1, "hi"),
            _assistant("e0000002", "e0000001", 2, [_call("c-b", "bash", {"command": "ls"})]),
            _result("e0000003", "e0000002", 3, "c-b", "bash", "a\n"),
        ]
        _write_jsonl(path, objects)
        session = read_session(path)
        assert not isinstance(session, SessionLoadError)
        (episode,) = session_episodes(session)
        assert (episode.ended_by, episode.ended_at) == ("end", _when(3))
        assert [c.exit_code for c in episode.commands] == [0]


class DiscoveryTest(unittest.TestCase):
    def test_discover_excludes_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_session(root, WINDOWS_CWD)
            other = root / "-other" / "2026-09-01T00-00-00-000Z_other.jsonl"
            _write_jsonl(other, [_header("other", POSIX_CWD)])
            assert discover_sessions(root) == (other, first)


class CleanPromptTest(unittest.TestCase):
    def test_strips_reminders_and_wrappers(self) -> None:
        text = (
            '<system-reminder reason="x">\nnoise\n</system-reminder>\n'
            "Do the <thing>\n\n\n\nnow\n<system-reminder>more</system-reminder>"
        )
        assert clean_prompt(text) == "Do the <thing>\n\nnow"
        assert clean_prompt("<user-request>\nHello\n</user-request>") == "Hello"
        assert clean_prompt("<a>one</a> and <b>two</b>") == "<a>one</a> and <b>two</b>"


if __name__ == "__main__":
    unittest.main()
