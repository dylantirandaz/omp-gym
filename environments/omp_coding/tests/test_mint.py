from __future__ import annotations

import json
import os
import subprocess
import tempfile
import tomllib
import unittest
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from omp_coding.mint import (
    PLACEHOLDER_IMAGE_DIGEST,
    Anchor,
    Candidate,
    CommandResult,
    GateFailure,
    GateResult,
    MintOptions,
    MintRejection,
    anchor_repository,
    assign_split,
    detect_runtime,
    find_secret,
    gate_task,
    is_test_command,
    is_test_path,
    materialize_workspace,
    mint_episode,
    reference_patch,
    render_dockerfile,
    render_task_toml,
    select_candidate,
    select_sealed_files,
)
from omp_coding.sessions import (
    CommandRun,
    Episode,
    FileMutation,
    SessionHeader,
    Usage,
)
from omp_coding.task import TaskSpec, load_task
from omp_coding.testcommand import CommandOutcome

GIT = ("git", "-c", "core.autocrlf=false", "-c", "user.name=t", "-c", "user.email=t@x")
SESSION_ID = "0123456789abcdef-session"
BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
OLD_TEST = "import unittest\n"
NEW_TEST = (
    "import unittest\n"
    "from pkg.mod import add\n\n\n"
    "class AddTest(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        self.assertEqual(add(1, 2), 3)\n\n"
    "    def test_zero(self):\n"
    "        self.assertEqual(add(0, 0), 0)\n"
)
TEST_COMMAND = "python3 -m unittest discover -s tests -t ."


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed git executable.
        [*GIT, "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _header(cwd: Path) -> SessionHeader:
    return SessionHeader(
        id=SESSION_ID,
        cwd=str(cwd),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        title="t",
        path=cwd / "s.jsonl",
        artifact_dir=cwd / "s",
    )


def _mutation(
    order: int, kind: str, path: str, content: str, old_text: str | None, cwd: Path
) -> FileMutation:
    return FileMutation(
        order=order,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=order),
        kind=kind,
        path=path,
        absolute_path=str(cwd / path),
        content=content,
        old_text=old_text,
        from_subagent=False,
    )


def _run(order: int, command: str, exit_code: int | None, *, is_error: bool = False) -> CommandRun:
    return CommandRun(
        order=order,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=order),
        command=command,
        exit_code=exit_code,
        is_error=is_error,
        output="",
        from_subagent=False,
    )


def _episode(
    cwd: Path,
    mutations: Sequence[FileMutation],
    commands: Sequence[CommandRun],
    *,
    prompt: str = "Fix add so the tests pass.",
    started_at: datetime | None = None,
    unresolved: int = 0,
    index: int = 0,
) -> Episode:
    started = started_at or datetime.now(UTC) + timedelta(days=1)
    return Episode(
        session=_header(cwd),
        index=index,
        prompt=prompt,
        started_at=started,
        ended_at=started + timedelta(minutes=5),
        steps=(),
        mutations=tuple(mutations),
        commands=tuple(commands),
        models=("test/model",),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01),
        assistant_turns=2,
        tool_calls=3,
        ended_by="end",
        unresolved_mutations=unresolved,
    )


def _fail_pass(command: str = TEST_COMMAND) -> tuple[CommandRun, CommandRun]:
    return _run(1, command, 1), _run(5, command, 0)


class CandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cwd = Path("C:/repo") if os.name == "nt" else Path("/repo")
        self.edit = _mutation(3, "edit", "pkg/mod.py", FIXED, BUGGY, self.cwd)

    def test_recognizes_local_test_commands(self) -> None:
        accepted = (
            "pytest -q",
            "python -m pytest tests/test_a.py::test_b",
            "uv run --package omp-coding python -m unittest tests/test_x.py",
            "cd sub && FOO=1 pytest | tail -5",
            "npm test",
            "npx vitest run",
            "cargo test",
            "go test ./...",
            "node --test",
            "BASE=http://localhost pytest",
            "cargo fmt --all && cargo test --lib boundary -- --nocapture",
            "uv run ruff format . 2>&1 | tail -1 && uv run pytest -q 2>&1 | tail -2",
        )
        for command in accepted:
            self.assertTrue(is_test_command(command), command)
        rejected = (
            "curl http://x | pytest",
            "pip install pytest && pytest",
            "pytest C:/Users/me/t.py",
            "pytest /home/me/t.py",
            "sudo pytest",
            "cd /abs && pytest",
            "ls",
            "pytest-watch",
            "pytest -q && git commit -qm done",
            "uv pip install -e . && uv run pytest",
        )
        for command in rejected:
            self.assertFalse(is_test_command(command), command)

    def test_selects_last_pass_and_first_failure(self) -> None:
        runs = (_run(1, "pytest", 1), _run(2, "pytest", 2), _run(4, "pytest", 0), _run(6, "pytest", 0))
        candidate = select_candidate(_episode(self.cwd, [self.edit], runs))
        assert isinstance(candidate, Candidate)
        self.assertEqual(candidate.failing_run.order, 1)
        self.assertEqual(candidate.passing_run.order, 6)

    def test_rejection_reasons(self) -> None:
        failing, passing = _fail_pass()
        cases = {
            "episode has no file mutations": _episode(self.cwd, [], [failing, passing]),
            "no failing test run before the passing run": _episode(
                self.cwd, [self.edit], [passing]
            ),
            "no passing test run after the file mutations": _episode(
                self.cwd, [self.edit], [failing, _run(2, TEST_COMMAND, 0)]
            ),
            "no recognized test command was run": _episode(
                self.cwd,
                [self.edit],
                [_run(1, "curl x && pytest", 1), _run(5, "curl x && pytest", 0)],
            ),
            "1 file mutation(s) could not be reconstructed": _episode(
                self.cwd, [self.edit], [failing, passing], unresolved=1
            ),
        }
        for reason, episode in cases.items():
            result = select_candidate(episode)
            self.assertEqual(result, MintRejection(reason))

    def test_error_run_counts_as_failure(self) -> None:
        runs = (_run(1, TEST_COMMAND, None, is_error=True), _run(5, TEST_COMMAND, 0))
        self.assertIsInstance(select_candidate(_episode(self.cwd, [self.edit], runs)), Candidate)


class RepoFixture(unittest.TestCase):
    """A two-commit repository whose second commit changed the edited file."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="omp-mint-test-")
        self.root = Path(self._temp.name).resolve()
        self.repo = self.root / "demo-repo"
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (self.repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (self.repo / "pkg" / "mod.py").write_text(BUGGY, encoding="utf-8")
        (self.repo / "tests" / "test_mod.py").write_text(OLD_TEST, encoding="utf-8")
        (self.repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "one")
        self.first = _git(self.repo, "rev-parse", "HEAD")
        (self.repo / "pkg" / "mod.py").write_text(BUGGY + "# later\n", encoding="utf-8")
        _git(self.repo, "commit", "-q", "-am", "two")
        self.second = _git(self.repo, "rev-parse", "HEAD")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def candidate(self, **kwargs: object) -> Candidate:
        mutations = [
            _mutation(3, "edit", "pkg/mod.py", FIXED, BUGGY, self.repo),
            _mutation(4, "edit", "tests/test_mod.py", NEW_TEST, OLD_TEST, self.repo),
        ]
        candidate = select_candidate(_episode(self.repo, mutations, _fail_pass(), **kwargs))
        assert isinstance(candidate, Candidate)
        return candidate

    def anchor(self) -> Anchor:
        anchor = anchor_repository(self.candidate())
        assert isinstance(anchor, Anchor), anchor
        return anchor


class AnchorTest(RepoFixture):
    def test_picks_commit_matching_old_text(self) -> None:
        anchor = self.anchor()
        self.assertEqual(anchor.base_commit, self.first)
        self.assertEqual(anchor.slug, "demo-repo")
        self.assertEqual(anchor.subdir, "")
        self.assertEqual([relative for relative, _ in anchor.mutations], ["pkg/mod.py", "tests/test_mod.py"])

    def test_rejects_when_no_commit_matches(self) -> None:
        mutations = [_mutation(3, "edit", "pkg/mod.py", FIXED, "something else\n", self.repo)]
        candidate = select_candidate(_episode(self.repo, mutations, _fail_pass()))
        assert isinstance(candidate, Candidate)
        self.assertEqual(
            anchor_repository(candidate),
            MintRejection("no commit matches the pre-edit file contents"),
        )

    def test_rejects_when_no_commit_predates_episode(self) -> None:
        candidate = self.candidate(started_at=datetime(2000, 1, 1, tzinfo=UTC))
        self.assertEqual(anchor_repository(candidate), MintRejection("no commit predates the episode"))

    def test_rejects_paths_outside_git(self) -> None:
        outside = self.root / "plain"
        outside.mkdir()
        mutations = [_mutation(3, "write", "a.py", "x\n", None, outside)]
        candidate = select_candidate(_episode(outside, mutations, _fail_pass()))
        assert isinstance(candidate, Candidate)
        result = anchor_repository(candidate)
        assert isinstance(result, MintRejection)
        self.assertTrue(result.reason.startswith("not inside a git repository"))

    def test_overlay_from_earlier_episode_wins(self) -> None:
        earlier = _episode(
            self.repo,
            [_mutation(1, "write", "pkg/mod.py", "overlay\n", None, self.repo)],
            [],
        )
        mutations = [_mutation(3, "edit", "pkg/mod.py", FIXED, "overlay\n", self.repo)]
        candidate = select_candidate(_episode(self.repo, mutations, _fail_pass(), index=1))
        assert isinstance(candidate, Candidate)
        anchor = anchor_repository(candidate, [earlier])
        assert isinstance(anchor, Anchor)
        self.assertEqual(anchor.base_commit, self.second)
        self.assertEqual(anchor.overlay, (("pkg/mod.py", "overlay\n"),))

    def test_fake_git_receives_toplevel_and_rev_list(self) -> None:
        calls: list[list[str]] = []

        def fake(args: Sequence[str]) -> CommandResult:
            calls.append(list(args))
            if args[2] == "rev-parse":
                return CommandResult(0, str(self.repo).encode(), b"")
            if args[2] == "rev-list":
                return CommandResult(0, b"deadbeef\n", b"")
            return CommandResult(1, b"", b"missing")

        result = anchor_repository(self.candidate(), git=fake)
        self.assertEqual(result, MintRejection("no commit matches the pre-edit file contents"))
        self.assertTrue(any(call[2] == "show" and call[3] == "deadbeef:pkg/mod.py" for call in calls))


class WorkspaceTest(RepoFixture):
    def test_workspace_matches_archive_and_patch_applies(self) -> None:
        anchor = self.anchor()
        workspace = self.root / "ws"
        self.assertIsNone(materialize_workspace(anchor, workspace))
        self.assertEqual((workspace / "pkg" / "mod.py").read_text(encoding="utf-8"), BUGGY)
        self.assertFalse((workspace / ".env").exists())
        self.assertFalse((workspace / ".git").exists())
        archived = set(_git(self.repo, "ls-tree", "-r", "--name-only", self.first).splitlines())
        present = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()}
        self.assertEqual(present, archived - {".env"})
        scratch = self.root / "scratch"
        scratch.mkdir()
        patch = reference_patch(anchor, workspace, scratch)
        assert not isinstance(patch, MintRejection), patch
        patch_bytes, end_state = patch
        self.assertEqual((end_state / "pkg" / "mod.py").read_text(encoding="utf-8"), FIXED)
        checkout = self.root / "checkout"
        subprocess.run([*GIT, "clone", "-q", str(self.repo), str(checkout)], check=True)  # noqa: S603
        _git(checkout, "checkout", "-q", self.first)
        patch_path = self.root / "reference.patch"
        patch_path.write_bytes(patch_bytes)
        _git(checkout, "apply", "--check", str(patch_path))
        _git(checkout, "apply", str(patch_path))
        self.assertEqual((checkout / "tests" / "test_mod.py").read_text(encoding="utf-8"), NEW_TEST)

    def test_empty_patch_is_rejected(self) -> None:
        anchor = self.anchor()
        unchanged = Anchor(
            toplevel=anchor.toplevel,
            subdir="",
            base_commit=anchor.base_commit,
            slug=anchor.slug,
            mutations=(("pkg/mod.py", _mutation(3, "edit", "pkg/mod.py", BUGGY, BUGGY, self.repo)),),
            overlay=(),
        )
        workspace = self.root / "ws"
        self.assertIsNone(materialize_workspace(unchanged, workspace))
        scratch = self.root / "scratch"
        scratch.mkdir()
        self.assertEqual(reference_patch(unchanged, workspace, scratch), MintRejection("reference patch is empty"))


class SealedFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.end = Path(self._temp.name)
        for relative in ("tests/test_a.py", "src/a.py", "src/a.spec.ts", "docs/readme.md"):
            target = self.end.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def anchor(self, *paths: str, subdir: str = "") -> Anchor:
        cwd = Path(self._temp.name)
        return Anchor(
            toplevel=str(cwd),
            subdir=subdir,
            base_commit="abc",
            slug="demo",
            mutations=tuple((p, _mutation(i, "edit", p, "y\n", "x\n", cwd)) for i, p in enumerate(paths)),
            overlay=(),
        )

    def test_patterns(self) -> None:
        for relative in ("tests/x.rs", "a/test/b.py", "test_a.py", "a_test.go", "x/y.spec.js", "__tests__/z.ts"):
            self.assertTrue(is_test_path(relative), relative)
        for relative in ("src/a.py", "testing/a.py", "contest.py"):
            self.assertFalse(is_test_path(relative), relative)

    def test_mutated_tests_and_command_files(self) -> None:
        sealed = select_sealed_files(self.anchor("src/a.py", "tests/test_a.py"), "pytest -q", self.end)
        self.assertEqual(sealed, ("tests/test_a.py",))
        sealed = select_sealed_files(self.anchor("src/a.py"), "npx vitest run src/a.spec.ts", self.end)
        self.assertEqual(sealed, ("src/a.spec.ts",))
        sealed = select_sealed_files(self.anchor("a.py", subdir="src"), "pytest ../tests/test_a.py::test_x", self.end)
        self.assertEqual(sealed, ("tests/test_a.py",))

    def test_falls_back_to_all_test_files(self) -> None:
        sealed = select_sealed_files(self.anchor("src/a.py"), "pytest", self.end)
        self.assertEqual(sealed, ("src/a.spec.ts", "tests/test_a.py"))

    def test_rejects_when_nothing_to_seal(self) -> None:
        for relative in ("tests/test_a.py", "src/a.spec.ts"):
            self.end.joinpath(*relative.split("/")).unlink()
        self.assertEqual(select_sealed_files(self.anchor("src/a.py"), "pytest", self.end), MintRejection("no test file to seal"))


class DockerfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def render(self, files: dict[str, str], command: str = "pytest") -> str:
        for relative, content in files.items():
            target = self.workspace.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        runtime = detect_runtime(self.workspace, "", command)
        text = render_dockerfile(runtime, self.workspace)
        self.assertIn("COPY workspace/ /workspace/", text)
        self.assertIn("util-linux", text)
        self.assertIn("git tag omp-gym-start", text)
        self.assertIn("/opt/omp-gym/gitignore", text)
        self.assertLess(text.index("apt-get"), text.index("UV_OFFLINE=1"))
        return text

    def test_python_with_uv_lock(self) -> None:
        text = self.render({"pyproject.toml": "[project]\nname='x'\n", "uv.lock": "version = 1\n"})
        self.assertIn("FROM python:3.12-slim-bookworm", text)
        self.assertIn("uv sync --frozen", text)
        self.assertIn("ENV PATH=/workspace/.venv/bin:$PATH", text)
        self.assertLess(text.index("uv sync"), text.index("UV_OFFLINE=1"))
        self.assertNotIn("pip install --no-cache-dir pytest", text)

    def test_python_requirements_installs_pytest(self) -> None:
        text = self.render({"requirements.txt": "requests\n"})
        self.assertIn("pip install --no-cache-dir -r requirements.txt", text)
        self.assertIn("pip install --no-cache-dir pytest", text)

    def test_node_lockfiles(self) -> None:
        self.assertIn("RUN npm ci", self.render({"package.json": "{}", "package-lock.json": "{}"}, "npm test"))
        (self.workspace / "package-lock.json").unlink()
        self.assertIn("pnpm install --frozen-lockfile", self.render({"pnpm-lock.yaml": ""}, "npm test"))
        (self.workspace / "pnpm-lock.yaml").unlink()
        self.assertIn("bun install --frozen-lockfile", self.render({"bun.lock": ""}, "bun test"))

    def test_rust_go_and_shell(self) -> None:
        self.assertIn("FROM rust:1-bookworm", self.render({"Cargo.toml": ""}, "cargo test"))
        (self.workspace / "Cargo.toml").unlink()
        self.assertIn("FROM golang:1-bookworm", self.render({"go.mod": ""}, "go test ./..."))
        (self.workspace / "go.mod").unlink()
        text = self.render({}, "sh run.sh")
        self.assertIn("FROM debian:bookworm-slim", text)
        self.assertEqual(detect_runtime(self.workspace, "", "x").kind, "shell")

    def test_subdir_project_is_preferred(self) -> None:
        runtime_files = {"apps/api/package.json": "{}", "pyproject.toml": ""}
        text = self.render(runtime_files, "npm test")
        runtime = detect_runtime(self.workspace, "apps/api", "npm test")
        self.assertEqual((runtime.kind, runtime.project_dir), ("node", "apps/api"))
        self.assertIn("WORKDIR /workspace/apps/api", render_dockerfile(runtime, self.workspace))
        self.assertIn("FROM python", text)


class AssemblyTest(unittest.TestCase):
    def test_split_is_deterministic_per_repo(self) -> None:
        self.assertEqual(assign_split("demo-repo", "auto"), assign_split("demo-repo", "auto"))
        self.assertEqual(assign_split("anything", "holdout"), "holdout")
        buckets = {assign_split(f"repo-{index}", "auto") for index in range(64)}
        self.assertEqual(buckets, {"train", "validation", "holdout"})

    def test_secret_patterns(self) -> None:
        for text in (
            "key AKIAABCDEFGHIJKLMNOP here",
            "sk-abcdefghijklmnopqrstuvwxyz",
            "ghp_" + "a" * 36,
            "xoxb-123",
            "-----BEGIN RSA PRIVATE KEY-----",
        ):
            self.assertIsNotNone(find_secret(text), text)
        self.assertIsNone(find_secret("nothing to see"))

    def test_toml_round_trip(self) -> None:
        document = {
            "a": "quote \" me",
            "b": 3,
            "c": 2.5,
            "d": True,
            "environment": {"list": ["x", "y"], "n": 1},
        }
        self.assertEqual(tomllib.loads(render_task_toml(document)), document)


def _fake_docker(args: Sequence[str]) -> CommandResult:
    if args[0] == "build":
        return CommandResult(0, b"", b"")
    if args[:2] == ["image", "inspect"]:
        return CommandResult(0, b"sha256:" + b"a" * 64 + b"|amd64\n", b"")
    raise AssertionError(f"unexpected docker call: {args}")


def _fake_runner(image: str, staging: Path, timeout_seconds: int) -> CommandOutcome:
    script = (staging / "grade.sh").read_text(encoding="utf-8")
    assert "cp -f /run/omp-gym/sealed/tests/test_mod.py /workspace/tests/test_mod.py" in script, script
    assert image.startswith("omp-gym/demo-repo-01234567-e0:")
    if (staging / "changes.patch").is_file():
        return CommandOutcome(0, "", "..\nRan 2 tests in 0.001s\n\nOK\n")
    return CommandOutcome(1, "", "F.\nRan 2 tests in 0.001s\n\nFAILED (failures=1)\n")


class MintAndGateTest(RepoFixture):
    def mint(self, **options: object) -> Path:
        candidate = self.candidate()
        anchor = anchor_repository(candidate)
        assert isinstance(anchor, Anchor)
        task_dir = mint_episode(candidate, anchor, self.root / "tasks", MintOptions(**options))
        assert isinstance(task_dir, Path), task_dir
        return task_dir

    def test_minted_task_loads_and_gate_fills_digest(self) -> None:
        task_dir = self.mint(split="train")
        self.assertEqual(task_dir.name, "demo-repo-01234567-e0")
        spec = load_task(task_dir)
        assert isinstance(spec, TaskSpec), spec
        self.assertEqual(spec.task_id, "omp-session/demo-repo-01234567-e0")
        self.assertEqual(spec.environment.image_digest, PLACEHOLDER_IMAGE_DIGEST)
        self.assertEqual(spec.verifier.command, ("sh", "-c", TEST_COMMAND))
        self.assertEqual(spec.verifier.sealed_files, ("tests/test_mod.py",))
        self.assertEqual(spec.source_revision, self.first)
        self.assertEqual(spec.runtime, "shell")
        self.assertTrue((task_dir / ".dockerignore").is_file())
        provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["base_commit"], self.first)
        self.assertEqual(provenance["failing_run"], {"exit_code": 1, "order": 1})
        self.assertIsNone(provenance["gate"])
        result = gate_task(task_dir, docker=_fake_docker, runner=_fake_runner)
        assert isinstance(result, GateResult), result
        self.assertEqual(result.reference.passed_cases, 2)
        self.assertEqual(result.before.status, "failed")
        gated = load_task(task_dir)
        assert isinstance(gated, TaskSpec), gated
        self.assertEqual(gated.expected_cases, 2)
        self.assertEqual(gated.environment.image_digest, "sha256:" + "a" * 64)
        provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["gate"]["reference"]["status"], "passed")
        self.assertEqual(provenance["gate"]["before"]["exit_code"], 1)

    def test_gate_failure_removes_dir_unless_kept(self) -> None:
        def always_pass(image: str, staging: Path, timeout_seconds: int) -> CommandOutcome:
            return CommandOutcome(0, "", "..\nRan 2 tests in 0.001s\n\nOK\n")

        task_dir = self.mint()
        result = gate_task(task_dir, docker=_fake_docker, runner=always_pass, keep_failed=True)
        self.assertEqual(result.reason if isinstance(result, GateFailure) else result, "tests already pass on the start state")
        self.assertTrue(task_dir.is_dir())
        result = gate_task(task_dir, docker=_fake_docker, runner=always_pass)
        self.assertIsInstance(result, GateFailure)
        self.assertFalse(task_dir.exists())

    def test_secret_in_prompt_rejects(self) -> None:
        candidate = select_candidate(
            _episode(self.repo, self.candidate().episode.mutations, _fail_pass(), prompt="use AKIAABCDEFGHIJKLMNOP")
        )
        assert isinstance(candidate, Candidate)
        anchor = anchor_repository(candidate)
        assert isinstance(anchor, Anchor)
        result = mint_episode(candidate, anchor, self.root / "tasks", MintOptions())
        self.assertEqual(result, MintRejection("prompt contains a secret-looking token"))
        self.assertFalse((self.root / "tasks" / "demo-repo-01234567-e0").exists())

    @unittest.skipUnless(os.environ.get("OMP_GYM_DOCKER_TESTS") == "1", "set OMP_GYM_DOCKER_TESTS=1")
    def test_real_docker_gate(self) -> None:
        task_dir = self.mint()
        result = gate_task(task_dir, keep_failed=True)
        assert isinstance(result, GateResult), result
        self.assertEqual(result.reference.passed_cases, 2)
        self.assertNotEqual(result.before.status, "passed")
        self.assertIsInstance(load_task(task_dir), TaskSpec)


if __name__ == "__main__":
    unittest.main()
