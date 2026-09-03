"""Prime Intellect account, hosted sandboxes, and the eval launcher.

Prime evaluations need three things this module provides in one place:

- credentials, read the way the Prime CLI and SDK read them (``PRIME_API_KEY``
  or ``~/.prime/config.json`` written by ``prime login``), with the ignored
  ``.env`` file loaded first;
- task images that Prime's hosted sandboxes can pull, built server-side from
  a minted task directory through the Prime image API;
- the ``omp-coding-eval`` launcher, which turns ``--hosted`` into the Prime
  VM runtime flags and makes the Verifiers v1 ``eval`` command run on Windows,
  where its POSIX-only imports otherwise stop it before any work happens.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import time
import tomllib
import types
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from .task import TaskLoadError, TestCommandVerifier, load_task, render_task_toml

ENV_FILE_NAME = ".env"
PRIME_CONFIG_FILE = Path.home() / ".prime" / "config.json"
PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"
PRIME_TOKENS_URL = "https://app.primeintellect.ai/dashboard/tokens"
HOSTED_DISK_GB = 20
BUILD_POLL_SECONDS = 10.0
DEFAULT_BUILD_TIMEOUT_SECONDS = 1800
UPLOAD_TIMEOUT_SECONDS = 600
PACKAGED_DOCKERFILE = ".__prime_dockerfile__"
BUILD_CONTEXT_EXCLUDES = frozenset({"verifier", "provenance.json", ".dockerignore"})
TERMINAL_BUILD_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
FRPC_RELEASE_URL = "https://github.com/fatedier/frp/releases/download"
FRPC_WINDOWS_ASSETS: Mapping[str, tuple[str, str]] = {
    "AMD64": (
        "windows_amd64",
        "3e2925b65a85938b936ea85072657c6c8e62b095c233e739da3eb5615b25ca55",
    ),
    "ARM64": (
        "windows_arm64",
        "dfd112469c91e6fa05274dc4929725b062b176b103463196908d24c7888e54b8",
    ),
}
FRPC_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
HOSTED_RUNTIME_ARGS = (
    "--env.agent.runtime.type",
    "prime",
    "--env.agent.runtime.vm",
    "--env.agent.runtime.disk",
    str(HOSTED_DISK_GB),
)
USAGE = "\n".join(
    (
        "usage: omp-coding-eval [--hosted] [--env-file PATH] <eval arguments>",
        "  --hosted     run rollouts and grading in Prime VM sandboxes (no local Docker)",
        "  --env-file   load this file instead of the nearest .env",
        "Everything else goes to the Verifiers v1 `eval` command. After `prime login`,",
        "the model endpoint defaults to Prime Inference with the same key, for example:",
        "  omp-coding-eval --hosted omp-coding --model openai/gpt-4.1-mini --no-push"
        " --env.taskset.split validation",
    )
)

Platform = Literal["linux/amd64", "linux/arm64"]


@dataclass(frozen=True)
class PrimeCredentials:
    """The API key Prime tools use, and where it came from."""

    api_key: str
    team_id: str | None
    source: Literal["environment", "config"]
    inference_url: str = PRIME_INFERENCE_URL

    def export(self, environ: dict[str, str]) -> None:
        """Make the key visible to the eval client and the Prime SDKs."""
        environ.setdefault("PRIME_API_KEY", self.api_key)
        if self.team_id:
            environ.setdefault("PRIME_TEAM_ID", self.team_id)


@dataclass(frozen=True)
class PublishResult:
    """One task image built by Prime and recorded in the task."""

    task_dir: str
    image: str
    build_id: str
    platform: Platform
    seconds: float


@dataclass(frozen=True)
class PublishFailure:
    """The task image could not be published."""

    task_dir: str
    reason: str


@dataclass(frozen=True)
class LaunchArguments:
    """The launcher's own options, split from the ``eval`` arguments."""

    hosted: bool
    env_file: Path | None
    eval_arguments: tuple[str, ...]


class ImageApi(Protocol):
    """The subset of the Prime image API the publisher calls."""

    def request(
        self, method: str, endpoint: str, json: Mapping[str, object] | None = None
    ) -> Mapping[str, object]: ...

    def upload(self, url: str, archive: bytes) -> None: ...


class SdkImageApi:
    """Prime image API through the ``prime_sandboxes`` client."""

    def __init__(self) -> None:
        from prime_sandboxes import APIClient

        self._client = APIClient()

    def request(
        self, method: str, endpoint: str, json: Mapping[str, object] | None = None
    ) -> Mapping[str, object]:
        return self._client.request(method, endpoint, json=dict(json) if json else None)

    def upload(self, url: str, archive: bytes) -> None:
        import httpx

        response = httpx.put(
            url,
            content=archive,
            headers={"Content-Type": "application/octet-stream"},
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()


# ---------------------------------------------------------------------------
# Credentials


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; comments, blanks, ``export`` and quotes allowed."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key.replace("_", "").isalnum():
            values[key] = value
    return values


def find_env_file(start: Path) -> Path | None:
    """Return the nearest ``.env`` in ``start`` or its parents."""
    for directory in (start, *start.parents):
        candidate = directory / ENV_FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def load_env_file(path: Path, environ: dict[str, str] | None = None) -> tuple[str, ...]:
    """Export the file's keys that are not already set; return the exported names."""
    target = os.environ if environ is None else environ
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    exported: list[str] = []
    for key, value in parse_env_file(text).items():
        if value and not target.get(key):
            target[key] = value
            exported.append(key)
    return tuple(exported)


def prime_credentials(
    environ: Mapping[str, str] | None = None, config_file: Path = PRIME_CONFIG_FILE
) -> PrimeCredentials | None:
    """Resolve the Prime API key the way the Prime CLI and SDK do."""
    env = os.environ if environ is None else environ
    api_key = env.get("PRIME_API_KEY", "")
    team_id = env.get("PRIME_TEAM_ID") or None
    inference_url = env.get("PRIME_INFERENCE_URL") or PRIME_INFERENCE_URL
    if api_key:
        return PrimeCredentials(api_key, team_id, "environment", inference_url)
    try:
        document = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    stored = document.get("api_key")
    if not isinstance(stored, str) or not stored:
        return None
    stored_team = document.get("team_id")
    if team_id is None and isinstance(stored_team, str) and stored_team:
        team_id = stored_team
    stored_url = document.get("inference_url")
    if not env.get("PRIME_INFERENCE_URL") and isinstance(stored_url, str) and stored_url:
        inference_url = stored_url
    return PrimeCredentials(stored, team_id, "config", inference_url)


def login_instructions() -> str:
    return "\n".join(
        (
            "No Prime Intellect credentials were found.",
            "Log in once with the Prime CLI (writes ~/.prime/config.json):",
            "  uv tool install prime",
            "  prime login          # browser challenge; or: prime config set-api-key",
            "  prime whoami         # verify the key and its permissions",
            f"Or create a key at {PRIME_TOKENS_URL} and put PRIME_API_KEY=... in .env.",
        )
    )


# ---------------------------------------------------------------------------
# Windows support for the Verifiers v1 eval process


def _flock_shim() -> types.ModuleType:
    """A ``fcntl`` stand-in backed by ``msvcrt`` byte-range locks.

    Verifiers locks its creation-limiter files with ``flock``. Locking byte
    zero gives every process the same lock region regardless of file size,
    which is what an advisory whole-file lock provides on POSIX.
    """
    import msvcrt

    shim = types.ModuleType("fcntl")
    shim.LOCK_SH = 1
    shim.LOCK_EX = 2
    shim.LOCK_NB = 4
    shim.LOCK_UN = 8

    def flock(fd: int, operation: int) -> None:
        position = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if operation & shim.LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            elif operation & shim.LOCK_NB:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        finally:
            os.lseek(fd, position, os.SEEK_SET)

    shim.flock = flock
    return shim


def install_fcntl_shim() -> bool:
    """Make ``import fcntl`` succeed on Windows; return whether it was needed."""
    if sys.platform != "win32" or "fcntl" in sys.modules:
        return False
    sys.modules["fcntl"] = _flock_shim()
    return True


def _download(url: str, limit: int) -> bytes:
    # The URL is derived from the pinned HTTPS release URL above.
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"download exceeds {limit} bytes: {url}")
    return data


def install_windows_frpc(
    bin_dir: Path,
    version: str,
    *,
    download: Callable[[str, int], bytes] = _download,
    machine: str | None = None,
) -> Path:
    """Seed prime_tunnel's binary cache with the official Windows frpc build.

    prime_tunnel downloads frpc only for macOS and Linux. It runs whatever
    file sits at ``<bin_dir>/frpc`` when the version marker matches, and
    Windows executes an extension-less PE file given by full path, so placing
    the Windows build there is enough for Prime tunnels to work.
    """
    key = (machine or platform.machine()).upper()
    asset = FRPC_WINDOWS_ASSETS.get(key)
    if asset is None:
        raise RuntimeError(f"no frpc release for Windows {key}")
    suffix, sha256 = asset
    binary = bin_dir / "frpc"
    marker = bin_dir / ".frpc_version"
    if binary.is_file() and marker.is_file() and marker.read_text().strip() == version:
        return binary
    archive_name = f"frp_{version}_{suffix}.zip"
    data = download(f"{FRPC_RELEASE_URL}/v{version}/{archive_name}", FRPC_MAX_ARCHIVE_BYTES)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"frpc archive checksum mismatch: {archive_name}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [name for name in archive.namelist() if name.endswith("/frpc.exe")]
        if len(members) != 1:
            raise RuntimeError(f"frpc.exe not found in {archive_name}")
        executable = archive.read(members[0])
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(executable)
    marker.write_text(version)
    return binary


def prepare_windows() -> None:
    """Apply every shim the Verifiers eval needs on Windows."""
    if sys.platform != "win32":
        return
    install_fcntl_shim()
    from prime_tunnel.binary import FRPC_VERSION
    from prime_tunnel.core.config import Config

    install_windows_frpc(Config().bin_dir, FRPC_VERSION)


# ---------------------------------------------------------------------------
# Publish task images to Prime


def build_context(task_dir: Path) -> bytes:
    """Tar the Dockerfile and workspace of one task the way ``prime images push`` does."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for child in sorted(task_dir.iterdir(), key=lambda path: path.name):
            if child.name not in BUILD_CONTEXT_EXCLUDES:
                archive.add(child, arcname=child.name)
        archive.add(task_dir / "Dockerfile", arcname=PACKAGED_DOCKERFILE)
    return buffer.getvalue()


def _image_reference(task_dir: Path) -> tuple[str, str] | PublishFailure:
    spec = load_task(task_dir)
    if isinstance(spec, TaskLoadError):
        return PublishFailure(str(task_dir), f"task does not load: {spec.reason}")
    if not isinstance(spec.verifier, TestCommandVerifier):
        return PublishFailure(str(task_dir), "only test-command-v1 tasks have task images")
    return spec.name, str(spec.task_revision)


def _wait_for_build(
    api: ImageApi,
    build_id: str,
    *,
    timeout_seconds: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> str | None:
    """Poll one build until it ends; return the failure reason or ``None``."""
    deadline = clock() + timeout_seconds
    while True:
        status = api.request("GET", f"/images/build/{build_id}")
        state = str(status.get("status") or "").upper()
        if state == "COMPLETED":
            return None
        if state in TERMINAL_BUILD_STATUSES:
            detail = status.get("errorMessage") or status.get("error_message") or ""
            return f"build ended as {state}{f': {detail}' if detail else ''}"
        if clock() > deadline:
            return f"build did not finish within {timeout_seconds}s (build {build_id})"
        sleep(BUILD_POLL_SECONDS)


def _record_publication(task_dir: Path, result: PublishResult) -> None:
    config_path = task_dir / "task.toml"
    document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    environment = dict(document["environment"])
    local_image = environment["image"]
    environment["image"] = result.image
    document["environment"] = environment
    config_path.write_text(render_task_toml(document), encoding="utf-8")
    provenance_path = task_dir / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        provenance = {}
    provenance["published"] = {
        **asdict(result),
        "local_image": local_image,
        "published_at": datetime.now(UTC).isoformat(),
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_task(
    task_dir: Path,
    *,
    api: ImageApi | None = None,
    team_id: str | None = None,
    target_platform: Platform = "linux/amd64",
    timeout_seconds: int = DEFAULT_BUILD_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PublishResult | PublishFailure:
    """Build one task image on Prime and point the task at it.

    The image is private to the account or team and named after the task, so
    ``environment.image`` becomes ``prime/<owner>/<task-name>:<revision>``,
    which Prime sandboxes pull. The previous local tag is kept in
    ``provenance.json``.
    """
    reference = _image_reference(task_dir)
    if isinstance(reference, PublishFailure):
        return reference
    name, tag = reference
    client = api if api is not None else SdkImageApi()
    started = clock()
    payload: dict[str, object] = {
        "image_name": name,
        "image_tag": tag,
        "dockerfile_path": PACKAGED_DOCKERFILE,
        "platform": target_platform,
        "visibility": "PRIVATE",
    }
    if team_id:
        payload["team_id"] = team_id
    try:
        initiated = client.request("POST", "/images/build", json=payload)
        build_id = str(initiated.get("build_id") or "")
        upload_url = str(initiated.get("upload_url") or "")
        image = str(initiated.get("fullImagePath") or "")
        if not build_id or not upload_url or not image:
            return PublishFailure(str(task_dir), "image build response is incomplete")
        client.upload(upload_url, build_context(task_dir))
        client.request("POST", f"/images/build/{build_id}/start", json={"context_uploaded": True})
        failure = _wait_for_build(
            client, build_id, timeout_seconds=timeout_seconds, clock=clock, sleep=sleep
        )
    except Exception as error:  # the SDK, HTTP, and file errors all end the publish
        return PublishFailure(str(task_dir), f"publish failed: {error}")
    if failure is not None:
        return PublishFailure(str(task_dir), failure)
    result = PublishResult(
        str(task_dir), image, build_id, target_platform, round(clock() - started, 3)
    )
    try:
        _record_publication(task_dir, result)
    except (OSError, ValueError, KeyError) as error:
        return PublishFailure(str(task_dir), f"publication could not be recorded: {error}")
    return result


# ---------------------------------------------------------------------------
# The eval launcher


def parse_launch_arguments(argv: Sequence[str]) -> LaunchArguments:
    """Split the launcher's options from the ``eval`` arguments."""
    parser = argparse.ArgumentParser(prog="omp-coding-eval", add_help=False)
    parser.add_argument("--hosted", action="store_true")
    parser.add_argument("--env-file", type=Path, default=None)
    known, rest = parser.parse_known_args(list(argv))
    return LaunchArguments(known.hosted, known.env_file, tuple(rest))


def eval_arguments(
    launch: LaunchArguments, credentials: PrimeCredentials | None
) -> tuple[str, ...]:
    """The ``eval`` argument list.

    ``--hosted`` adds the Prime VM runtime. A logged-in run without an explicit
    model endpoint uses Prime Inference with the same key, so one login covers
    sandboxes, tunnels, and the model.
    """
    arguments = list(launch.eval_arguments)
    if credentials is not None and not any(
        item == "--client.base-url" or item.startswith("--client.base-url=")
        for item in arguments
    ):
        arguments.extend(
            ("--client.base-url", credentials.inference_url, "--client.api-key-var", "PRIME_API_KEY")
        )
    if launch.hosted:
        arguments.extend(HOSTED_RUNTIME_ARGS)
    return tuple(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    launch = parse_launch_arguments(arguments)
    env_file = launch.env_file or find_env_file(Path.cwd())
    if env_file is not None:
        load_env_file(env_file)
    credentials = prime_credentials()
    if credentials is not None:
        credentials.export(os.environ)
    elif launch.hosted:
        print(login_instructions(), file=sys.stderr)
        return 2
    try:
        prepare_windows()
    except (RuntimeError, OSError, urllib.error.URLError) as error:
        print(f"Windows setup failed: {error}", file=sys.stderr)
        return 2
    from verifiers.v1.cli.eval.main import main as eval_main

    eval_main(list(eval_arguments(launch, credentials)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
