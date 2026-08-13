"""Doctor checks and the first-win init flow.

doctor reports what is present and what is missing, with the fix
for each failure. init runs doctor and then one episode on the
smallest task, so a new install ends with a measured result.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from .preflight import PreflightError, require_metal_gpu
from .runner import EpisodeFailure, run_episode
from .task import TaskLoadError, load_task


@dataclass(frozen=True)
class Check:
    """One doctor check."""

    name: str
    ok: bool
    detail: str


def run_doctor(env_path: Path, sessions_root: Path) -> list[Check]:
    """Run every environment check. Never raises."""
    checks: list[Check] = []

    omp_path = shutil.which("omp")
    checks.append(
        Check(
            "omp on PATH",
            omp_path is not None,
            omp_path or "install omp first: https://omp.sh",
        )
    )

    checks.append(
        Check(
            "uv on PATH",
            shutil.which("uv") is not None,
            shutil.which("uv") or "install uv",
        )
    )

    try:
        report = require_metal_gpu()
        checks.append(
            Check(
                "Metal GPU",
                True,
                f"{report.device_name}, "
                f"{report.memory_bytes // (1024 * 1024)} MiB",
            )
        )
    except PreflightError as error:
        checks.append(Check("Metal GPU", False, str(error)))

    keys = {}
    if env_path.is_file():
        from .envfile import load_env_file

        keys = load_env_file(env_path)
    checks.append(
        Check(
            "provider keys in .env",
            len(keys) > 0,
            (
                f"{len(keys)} keys"
                if keys
                else "add OPENROUTER_API_KEY or rely on omp auth"
            ),
        )
    )

    session_count = (
        len(list(sessions_root.rglob("*.jsonl")))
        if sessions_root.is_dir()
        else 0
    )
    checks.append(
        Check(
            "omp sessions on disk",
            session_count > 0,
            (
                f"{session_count} sessions"
                if session_count
                else "no sessions yet; export will only see episodes"
            ),
        )
    )

    free_gb = shutil.disk_usage(Path.cwd()).free // (1024**3)
    checks.append(
        Check(
            "disk space",
            free_gb >= 5,
            f"{free_gb} GiB free",
        )
    )
    return checks


def print_doctor(checks: list[Check]) -> int:
    """Print checks and return a process exit code."""
    failures = 0
    for check in checks:
        mark = "ok" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")
        if not check.ok:
            failures += 1
    return 0 if failures == 0 else 1


def run_init(
    env_path: Path,
    sessions_root: Path,
    runs_dir: Path,
    task_dir: Path,
) -> int:
    """Doctor, then one scored episode as the first win."""
    code = print_doctor(run_doctor(env_path, sessions_root))
    if code != 0:
        return code
    task = load_task(task_dir)
    if isinstance(task, TaskLoadError):
        print(f"bad task {task.path}: {task.reason}")
        return 1
    print(f"\nfirst episode: {task.name} with the default model")
    result = run_episode(task, runs_dir, None)
    if isinstance(result, EpisodeFailure):
        print(f"episode failed: {result.reason}")
        return 1
    print(
        f"reward {result.reward} in {result.duration_seconds}s; "
        f"ledger and dashboard now have data"
    )
    return 0
