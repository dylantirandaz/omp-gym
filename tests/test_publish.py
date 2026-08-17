"""Behavior tests for ledger page publishing."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from omp_gym.ledger import append_entry
from omp_gym.publish import publish_report


def _git(repo: Path, git_binary: str, *arguments: str) -> str:
    """Run one fixed-argument git command in the test repository."""
    completed = subprocess.run(  # noqa: S603 - resolved git executable
        [git_binary, *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


class PublishReportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "git is not available")
    def test_push_commits_only_the_report_page(self) -> None:
        git_binary = shutil.which("git")
        if git_binary is None:
            self.fail("git disappeared after the skip check")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            origin = root / "origin.git"
            repo.mkdir()
            _git(repo, git_binary, "init", "-b", "main")
            _git(repo, git_binary, "config", "user.name", "Test User")
            _git(repo, git_binary, "config", "user.email", "test@example.com")
            (repo / "README.txt").write_text("initial\n")
            _git(repo, git_binary, "add", "README.txt")
            _git(repo, git_binary, "commit", "-m", "Initial")
            _git(repo, git_binary, "init", "--bare", str(origin))
            _git(repo, git_binary, "remote", "add", "origin", str(origin))
            _git(repo, git_binary, "push", "-u", "origin", "main")

            ledger = repo / "experiments" / "ledger.jsonl"
            append_entry(
                ledger,
                kind="bench",
                config={"model": "test-model"},
                metrics={"reward": 1.0},
                artifacts={},
            )
            (repo / "unrelated.txt").write_text("keep staged\n")
            _git(repo, git_binary, "add", "unrelated.txt")

            result = publish_report(repo, ledger, push=True)

            committed = _git(
                repo,
                git_binary,
                "show",
                "--pretty=format:",
                "--name-only",
                "HEAD",
            ).splitlines()
            staged = _git(
                repo, git_binary, "diff", "--cached", "--name-only"
            ).splitlines()
            self.assertEqual(committed, ["docs/index.html"])
            self.assertEqual(staged, ["unrelated.txt"])
            self.assertTrue(result["pushed"])


if __name__ == "__main__":
    unittest.main()
