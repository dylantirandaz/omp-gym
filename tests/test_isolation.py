"""Behavior tests for the episode isolation boundary."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from omp_gym.isolation import (
    ResourceLimits,
    build_sandbox_profile,
    limited_command,
    sandbox_available,
    scrub_secret_environment,
    write_sandbox_profile,
)


class SecretEnvironmentTests(unittest.TestCase):
    def test_only_the_selected_provider_key_survives(self) -> None:
        environment = {
            "PATH": "/bin",
            "ANTHROPIC_API_KEY": "selected",
            "OPENAI_API_KEY": "other",
            "GITHUB_TOKEN": "other",
            "DATABASE_PASSWORD": "other",
            "NORMAL_VALUE": "kept",
        }
        scrubbed = scrub_secret_environment(
            environment,
            keep=("ANTHROPIC_API_KEY",),
        )
        self.assertEqual(
            scrubbed,
            {
                "PATH": "/bin",
                "ANTHROPIC_API_KEY": "selected",
                "NORMAL_VALUE": "kept",
            },
        )

    def test_exact_secret_names_are_removed(self) -> None:
        scrubbed = scrub_secret_environment(
            {
                "HF_HOME": "/private/cache",
                "HF_TOKEN": "secret",
                "OMP_GYM_SHIM_TOKEN": "secret",
            }
        )
        self.assertEqual(scrubbed, {})


class ResourceLimitTests(unittest.TestCase):
    def test_limit_launcher_applies_open_file_ceiling(self) -> None:
        command = limited_command(
            (
                sys.executable,
                "-c",
                "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])",
            ),
            ResourceLimits(cpu_seconds=30, open_files=256),
        )
        completed = subprocess.run(  # noqa: S603 - internal launcher
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "256")


class SandboxProfileTests(unittest.TestCase):
    def test_unknown_network_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_sandbox_profile(writable=(), network="all")

    def test_paths_are_canonical_and_root_access_is_literal_only(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:  # noqa: S108
            alias = Path(temporary) / "workspace"
            alias.mkdir()
            profile = build_sandbox_profile(
                writable=(alias,),
                network="deny",
            )
        self.assertIn(f'(subpath "{alias.resolve()}")', profile)
        self.assertIn('(literal "/")', profile)
        self.assertNotIn('(subpath "/")', profile)
        self.assertIn("(deny network*)", profile)

    def test_remote_profile_contains_resolver_and_https_grants(self) -> None:
        profile = build_sandbox_profile(writable=(), network="open-443")
        self.assertIn('(literal "/var")', profile)
        self.assertIn(
            '(literal "/private/var/run/mDNSResponder")',
            profile,
        )
        self.assertIn(
            '(global-name "com.apple.SystemConfiguration.configd")',
            profile,
        )
        self.assertIn(
            '(allow network-outbound (remote tcp "*:443"))',
            profile,
        )

    @unittest.skipUnless(sandbox_available(), "macOS sandbox-exec is not available")
    def test_real_profile_allows_workspace_and_denies_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            sibling = root / "outside"
            workspace.mkdir()
            sibling.mkdir()
            secret = sibling / "secret.txt"
            secret.write_text("must not be read")
            interpreter = Path(sys.executable).resolve()
            profile = write_sandbox_profile(
                root,
                "proof.sb",
                writable=(workspace,),
                network="deny",
                readable=(interpreter.parent.parent,),
            )
            code = (
                "from pathlib import Path\n"
                "Path('result.txt').write_text('allowed')\n"
                f"target = Path({str(secret)!r})\n"
                "try:\n"
                "    target.read_text()\n"
                "except OSError:\n"
                "    print('read-denied')\n"
                "else:\n"
                "    raise SystemExit('sibling read was allowed')\n"
            )
            completed = subprocess.run(  # noqa: S603 - fixed argv
                [
                    "/usr/bin/sandbox-exec",
                    "-f",
                    str(profile),
                    str(interpreter),
                    "-c",
                    code,
                ],
                cwd=workspace,
                env={
                    "PATH": os.environ["PATH"],
                    "HOME": str(workspace),
                    "TMPDIR": str(workspace),
                },
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("read-denied", completed.stdout)
            self.assertEqual((workspace / "result.txt").read_text(), "allowed")


if __name__ == "__main__":
    unittest.main()
