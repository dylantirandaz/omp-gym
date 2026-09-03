from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import tomllib
import unittest
import zipfile
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

from omp_coding.prime import (
    FRPC_WINDOWS_ASSETS,
    HOSTED_RUNTIME_ARGS,
    PACKAGED_DOCKERFILE,
    PrimeCredentials,
    PublishFailure,
    PublishResult,
    build_context,
    eval_arguments,
    find_env_file,
    install_windows_frpc,
    load_env_file,
    parse_env_file,
    parse_launch_arguments,
    prime_credentials,
    publish_task,
)
from omp_coding.task import EMPTY_DEPENDENCY_LOCK_DIGEST, TaskLoadError, load_task

IMAGE_DIGEST = "sha256:" + "ab" * 32


def _write_task(root: Path) -> Path:
    task_dir = root / "calc-12345678-e0"
    (task_dir / "workspace" / "tests").mkdir(parents=True)
    (task_dir / "verifier" / "files" / "tests").mkdir(parents=True)
    (task_dir / "workspace" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (task_dir / "workspace" / "tests" / "test_calc.py").write_text("def test_add():\n    pass\n")
    (task_dir / "verifier" / "files" / "tests" / "test_calc.py").write_text(
        "def test_add():\n    assert True\n"
    )
    (task_dir / "verifier" / "reference.patch").write_text("diff --git a/calc.py b/calc.py\n")
    (task_dir / "Dockerfile").write_text("FROM python:3.12-slim-bookworm\n")
    (task_dir / ".dockerignore").write_text("verifier/\nprovenance.json\n")
    (task_dir / "provenance.json").write_text(json.dumps({"schema_version": 1}))
    (task_dir / "task.toml").write_text(
        "\n".join(
            (
                "schema_version = 3",
                'task_id = "omp-session/calc-12345678-e0"',
                "task_revision = 1",
                'family = "calc"',
                'split = "holdout"',
                'prompt = "Fix add."',
                'runtime = "python"',
                'runtime_version = "3.12"',
                "max_time_seconds = 1800",
                "token_budget = 400000",
                "expected_cases = 2",
                'source = "omp-session:x/0"',
                'source_revision = "abc"',
                'license = "NOASSERTION"',
                'sensitive_data = "private"',
                "seed = 0",
                "",
                "[environment]",
                'image = "omp-gym/calc-12345678-e0:1"',
                f'image_digest = "{IMAGE_DIGEST}"',
                'os = "linux"',
                'architecture = "amd64"',
                'network = "none"',
                "cpus = 2.0",
                "memory_bytes = 4294967296",
                "pids = 512",
                "workspace_bytes = 1073741824",
                "temp_bytes = 268435456",
                "home_bytes = 67108864",
                f'dependency_lock_digest = "{EMPTY_DEPENDENCY_LOCK_DIGEST}"',
                "",
                "[verifier]",
                'protocol = "test-command-v1"',
                'command = ["sh", "-c", "python -m pytest -q"]',
                "timeout_seconds = 300",
                'sealed_files = ["tests/test_calc.py"]',
                'reference = "verifier/reference.patch"',
                "",
            )
        )
    )
    return task_dir


class _FakeImageApi:
    """Scripted Prime image API: records calls, serves the given statuses."""

    def __init__(self, statuses: list[str], *, image: str = "prime/alice/calc-12345678-e0:1"):
        self.statuses = list(statuses)
        self.image = image
        self.calls: list[tuple[str, str, Mapping[str, object] | None]] = []
        self.uploads: list[tuple[str, bytes]] = []

    def request(self, method: str, endpoint: str, json: Mapping[str, object] | None = None):
        self.calls.append((method, endpoint, json))
        if endpoint == "/images/build":
            return {"build_id": "build-1", "upload_url": "https://upload/x", "fullImagePath": self.image}
        if endpoint.endswith("/start"):
            return {}
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {"status": status, "errorMessage": "boom" if status == "FAILED" else None}

    def upload(self, url: str, archive: bytes) -> None:
        self.uploads.append((url, archive))


class EnvFileTests(unittest.TestCase):
    def test_parse_handles_comments_exports_and_quotes(self) -> None:
        parsed = parse_env_file(
            "# comment\n\nexport PRIME_API_KEY='abc'\nOPENROUTER_API_KEY=\"x=y\"\nBAD LINE\nEMPTY=\n"
        )
        self.assertEqual(parsed, {"PRIME_API_KEY": "abc", "OPENROUTER_API_KEY": "x=y", "EMPTY": ""})

    def test_load_only_fills_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text("PRIME_API_KEY=file\nOPENROUTER_API_KEY=file\nEMPTY=\n")
            environ = {"PRIME_API_KEY": "already"}
            exported = load_env_file(path, environ)
        self.assertEqual(exported, ("OPENROUTER_API_KEY",))
        self.assertEqual(environ, {"PRIME_API_KEY": "already", "OPENROUTER_API_KEY": "file"})

    def test_find_walks_up_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("A=1\n")
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            self.assertEqual(find_env_file(nested), root / ".env")
            self.assertIsNone(find_env_file(Path(temporary).parent / "definitely-missing-dir"))


class CredentialTests(unittest.TestCase):
    def test_environment_wins_over_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"api_key": "from-file", "team_id": "team-file"}))
            from_env = prime_credentials({"PRIME_API_KEY": "from-env"}, config)
            from_file = prime_credentials({}, config)
            missing = prime_credentials({}, Path(temporary) / "absent.json")
        self.assertEqual((from_env.api_key, from_env.team_id, from_env.source), ("from-env", None, "environment"))
        self.assertEqual((from_file.api_key, from_file.team_id, from_file.source), ("from-file", "team-file", "config"))
        self.assertIsNone(missing)

    def test_empty_config_key_is_not_a_login(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"api_key": ""}))
            self.assertIsNone(prime_credentials({}, config))


class LaunchTests(unittest.TestCase):
    def test_hosted_appends_prime_vm_runtime_after_passthrough(self) -> None:
        launch = parse_launch_arguments(
            ["--hosted", "omp-coding", "--model", "m", "--env-file", "x.env", "--no-push"]
        )
        self.assertTrue(launch.hosted)
        self.assertEqual(launch.env_file, Path("x.env"))
        self.assertEqual(
            eval_arguments(launch, None),
            ("omp-coding", "--model", "m", "--no-push", *HOSTED_RUNTIME_ARGS),
        )

    def test_local_run_passes_arguments_through_unchanged(self) -> None:
        launch = parse_launch_arguments(["omp-coding", "--env.taskset.split", "train"])
        self.assertFalse(launch.hosted)
        self.assertIsNone(launch.env_file)
        self.assertEqual(eval_arguments(launch, None), ("omp-coding", "--env.taskset.split", "train"))

    def test_logged_in_run_defaults_to_prime_inference(self) -> None:
        credentials = PrimeCredentials("key", None, "config", "https://inference.example/v1")
        launch = parse_launch_arguments(["omp-coding", "--model", "m"])
        self.assertEqual(
            eval_arguments(launch, credentials),
            (
                "omp-coding",
                "--model",
                "m",
                "--client.base-url",
                "https://inference.example/v1",
                "--client.api-key-var",
                "PRIME_API_KEY",
            ),
        )
        explicit = parse_launch_arguments(["omp-coding", "--client.base-url", "https://x/v1"])
        self.assertEqual(
            eval_arguments(explicit, credentials), ("omp-coding", "--client.base-url", "https://x/v1")
        )

    def test_export_fills_only_missing_variables(self) -> None:
        environ = {"PRIME_API_KEY": "kept"}
        PrimeCredentials("key", "team", "config").export(environ)
        self.assertEqual(environ, {"PRIME_API_KEY": "kept", "PRIME_TEAM_ID": "team"})


class PublishTests(unittest.TestCase):
    def test_publish_uploads_context_waits_and_records_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = _write_task(Path(temporary))
            api = _FakeImageApi(["BUILDING", "COMPLETED"])
            ticks = iter(range(0, 1000, 5))
            result = publish_task(
                task_dir,
                api=api,
                team_id="team-1",
                clock=lambda: float(next(ticks)),
                sleep=lambda seconds: None,
            )
            self.assertIsInstance(result, PublishResult)
            assert isinstance(result, PublishResult)
            self.assertEqual(result.image, "prime/alice/calc-12345678-e0:1")
            document = tomllib.loads((task_dir / "task.toml").read_text())
            provenance = json.loads((task_dir / "provenance.json").read_text())
            self.assertNotIsInstance(load_task(task_dir), TaskLoadError)
        self.assertEqual(api.calls[0][1], "/images/build")
        self.assertEqual(
            api.calls[0][2],
            {
                "image_name": "calc-12345678-e0",
                "image_tag": "1",
                "dockerfile_path": PACKAGED_DOCKERFILE,
                "platform": "linux/amd64",
                "visibility": "PRIVATE",
                "team_id": "team-1",
            },
        )
        self.assertEqual(api.calls[1][1], "/images/build/build-1/start")
        self.assertEqual(api.uploads[0][0], "https://upload/x")
        self.assertEqual(document["environment"]["image"], "prime/alice/calc-12345678-e0:1")
        self.assertEqual(provenance["published"]["local_image"], "omp-gym/calc-12345678-e0:1")
        self.assertEqual(provenance["published"]["build_id"], "build-1")

    def test_failed_build_reports_the_backend_message(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = _write_task(Path(temporary))
            api = _FakeImageApi(["FAILED"])
            result = publish_task(task_dir, api=api, clock=lambda: 0.0, sleep=lambda s: None)
            self.assertIsInstance(result, PublishFailure)
            assert isinstance(result, PublishFailure)
            self.assertIn("FAILED: boom", result.reason)
            document = tomllib.loads((task_dir / "task.toml").read_text())
            self.assertEqual(document["environment"]["image"], "omp-gym/calc-12345678-e0:1")

    def test_build_timeout_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = _write_task(Path(temporary))
            api = _FakeImageApi(["BUILDING"])
            ticks = iter([0.0, 0.0, 10.0, 5000.0, 5000.0])
            result = publish_task(
                task_dir, api=api, timeout_seconds=100, clock=lambda: next(ticks), sleep=lambda s: None
            )
            self.assertIsInstance(result, PublishFailure)
            assert isinstance(result, PublishFailure)
            self.assertIn("did not finish", result.reason)

    def test_build_context_excludes_sealed_files_and_packages_the_dockerfile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = _write_task(Path(temporary))
            with tarfile.open(fileobj=io.BytesIO(build_context(task_dir)), mode="r:gz") as archive:
                names = set(archive.getnames())
        self.assertIn("workspace/calc.py", names)
        self.assertIn("Dockerfile", names)
        self.assertIn(PACKAGED_DOCKERFILE, names)
        self.assertFalse(any(name.startswith("verifier") for name in names))
        self.assertNotIn("provenance.json", names)

    def test_call_cases_tasks_have_no_image_to_publish(self) -> None:
        packaged = Path(__file__).resolve().parents[1] / "omp_coding" / "tasks" / "fizzbuzz-fix"
        result = publish_task(packaged, api=_FakeImageApi(["COMPLETED"]))
        self.assertIsInstance(result, PublishFailure)


class FrpcTests(unittest.TestCase):
    def _archive(self, version: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"frp_{version}_windows_amd64/frpc.exe", b"MZ-frpc")
            archive.writestr(f"frp_{version}_windows_amd64/frps.exe", b"MZ-frps")
        return buffer.getvalue()

    def test_install_verifies_checksum_and_seeds_the_cache(self) -> None:
        version = "0.66.0"
        data = self._archive(version)
        expected = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary) / "bin"
            with mock.patch.dict(FRPC_WINDOWS_ASSETS, {"AMD64": ("windows_amd64", expected)}):
                binary = install_windows_frpc(
                    bin_dir, version, download=lambda url, limit: data, machine="AMD64"
                )
                self.assertEqual(binary.read_bytes(), b"MZ-frpc")
                self.assertEqual((bin_dir / ".frpc_version").read_text(), version)
                # A matching cache is reused without a download.
                again = install_windows_frpc(
                    bin_dir, version, download=lambda url, limit: b"", machine="AMD64"
                )
                self.assertEqual(again, binary)

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                install_windows_frpc(
                    Path(temporary), "0.66.0", download=lambda url, limit: b"nope", machine="AMD64"
                )

    @unittest.skipUnless(sys.platform == "win32", "msvcrt locks exist only on Windows")
    def test_flock_shim_locks_and_unlocks(self) -> None:
        from omp_coding.prime import _flock_shim

        shim = _flock_shim()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            with path.open("a+") as handle:
                shim.flock(handle.fileno(), shim.LOCK_EX)
                handle.seek(0)
                handle.write("1.0")
                handle.flush()
                shim.flock(handle.fileno(), shim.LOCK_UN)
            self.assertEqual(path.read_text(), "1.0")


if __name__ == "__main__":
    unittest.main()
