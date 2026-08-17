"""Episode isolation: sandbox profiles, resource limits, env scrub.

One episode spawns two kinds of child processes: the agent (omp)
and the test commands. Neither is trusted. This module is the
single place that decides what those processes may reach.

The boundary is macOS `sandbox-exec`: a deny-default profile that
grants read access to system and interpreter prefixes, write
access only to the episode's own directories, and a network mode
per process kind. Test processes get no network at all. Episodes
serving a local model get loopback only; episodes calling a
remote provider API get outbound 443, because the provider
request itself needs it. On hosts without sandbox-exec the
caller must decide explicitly: refuse to run, or accept the
documented residual risk. Nothing here silently degrades.

Resource limits come from setrlimit in the child pre-exec hook,
and stream drains are byte-capped so a chatty child cannot
exhaust parent memory.

Residual risk, stated plainly: a sandboxed process can still
abuse anything the profile grants, sandbox-exec is deprecated by
Apple, and a descendant that escapes its process group before the
killer runs can survive. A VM boundary removes those classes;
this module removes the cheap ones.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_STREAM_BYTES = 8 * 1024 * 1024
"""Most bytes kept from one child output stream; excess kills it."""


_SECRET_NAME_PATTERN = re.compile(
    r"(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIALS|APIKEY)$"
)
_SECRET_NAME_EXACT = frozenset(
    (
        "OMP_GYM_SHIM_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "HF_TOKEN",
        "HF_HOME",
    )
)

_READ_PREFIXES = (
    "/usr",
    "/bin",
    "/sbin",
    "/System",
    "/Library",
    "/Applications",
    "/opt",
    "/private/etc",
    "/private/var/db",
    "/dev",
    "/var/db",
    "/etc",
)

_MDNS_SOCKET = "/private/var/run/mDNSResponder"
_REMOTE_NETWORK_READ_LITERALS = (
    "/var",
    "/private",
    "/private/var",
    "/private/var/run",
    _MDNS_SOCKET,
)

_LIMIT_EXEC_PATH = Path(__file__).with_name("_limit_exec.py").resolve()
_PYTHON_EXECUTABLE = Path(sys.executable).resolve()
_RUNTIME_READ_PATHS = (_LIMIT_EXEC_PATH, _PYTHON_EXECUTABLE)


class IsolationError(RuntimeError):
    """The requested isolation is not available on this host."""


def scrub_secret_environment(
    environment: Mapping[str, str],
    *,
    keep: Sequence[str] = (),
) -> dict[str, str]:
    """Drop variables whose names mark them as credentials.

    `keep` names the exact variables a caller proved it needs, for
    example the one provider key of the model under test. Anything
    not named there and matching a credential shape stays out.
    """
    keep_set = set(keep)
    return {
        name: value
        for name, value in environment.items()
        if name in keep_set
        or (name not in _SECRET_NAME_EXACT and not _SECRET_NAME_PATTERN.search(name))
    }


def sandbox_available() -> bool:
    """True when this host can enforce a sandbox-exec profile."""
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _sandbox_escape(value: str) -> str:
    """Quote one literal for a sandbox profile string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_sandbox_profile(
    *,
    writable: Sequence[Path],
    readable: Sequence[Path] = (),
    network: str = "deny",
) -> str:
    """One deny-default sandbox profile.

    `writable` paths allow read and write; `readable` adds extra
    read-only prefixes beyond the system defaults. `network` is
    "deny" for no network, "loopback" for loopback TCP only, or
    "open-443" for loopback plus outbound TCP on port 443. Any
    other value is rejected, because an open network must be a
    conscious caller decision.
    """
    if network not in ("deny", "loopback", "open-443"):
        raise ValueError(f"unknown network mode: {network}")
    writable_paths = tuple(path.resolve() for path in writable)
    readable_paths = tuple(
        dict.fromkeys(
            path.resolve()
            for path in (*writable_paths, *readable, *_RUNTIME_READ_PATHS)
        )
    )
    readable_ancestors = tuple(
        dict.fromkeys(
            parent
            for path in readable_paths
            for parent in reversed(path.parents)
            if parent != Path("/")
        )
    )
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))',
        "(allow sysctl-read)",
        "(allow file-read*",
    ]
    for ancestor in readable_ancestors:
        lines.append(f'    (literal "{_sandbox_escape(str(ancestor))}")')
    lines.append('    (literal "/")')
    if network == "open-443":
        for path in _REMOTE_NETWORK_READ_LITERALS:
            lines.append(f'    (literal "{path}")')
    for prefix in _READ_PREFIXES:
        lines.append(f'    (subpath "{_sandbox_escape(prefix)}")')
    for path in readable_paths:
        lines.append(f'    (subpath "{_sandbox_escape(str(path))}")')
    lines.append(")")
    lines.append('(allow file-write* (subpath "/dev"))')
    for path in writable_paths:
        lines.append(f'(allow file-write* (subpath "{_sandbox_escape(str(path))}"))')
    if network == "deny":
        lines.append("(deny network*)")
    elif network == "loopback":
        lines.append('(allow network-outbound (remote tcp "localhost:*"))')
    else:
        lines.append(
            '(allow mach-lookup (global-name "com.apple.SystemConfiguration.configd"))'
        )
        lines.append(f'(allow network-outbound (literal "{_MDNS_SOCKET}"))')
        lines.append("(allow system-socket)")
        lines.append('(allow network-outbound (remote tcp "localhost:*"))')
        lines.append('(allow network-outbound (remote tcp "*:443"))')
    return "\n".join(lines) + "\n"


def write_sandbox_profile(
    directory: Path,
    name: str,
    *,
    writable: Sequence[Path],
    readable: Sequence[Path] = (),
    network: str = "deny",
) -> Path:
    """Write one profile file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / name
    profile_path.write_text(
        build_sandbox_profile(writable=writable, readable=readable, network=network)
    )
    return profile_path


def sandbox_command(command: Sequence[str], profile: Path) -> list[str]:
    """Prefix a command with sandbox-exec, or refuse when absent."""
    if not sandbox_available():
        raise IsolationError(
            "sandbox-exec is required for episode isolation on this host; "
            "run on macOS or set OMP_GYM_SANDBOX=0 to accept the risk"
        )
    return ["sandbox-exec", "-f", str(profile), *command]


@dataclass(frozen=True)
class ResourceLimits:
    """Child ceilings applied by a launcher before target execution."""

    cpu_seconds: int = 600
    memory_bytes: int = 4 * 1024 * 1024 * 1024
    file_bytes: int = 512 * 1024 * 1024
    open_files: int = 65_536
    processes: int = 512


def limited_command(command: Sequence[str], limits: ResourceLimits) -> list[str]:
    """Prefix a command with the resource-limit launcher."""
    if not command:
        raise ValueError("limited command must not be empty")
    return [
        str(_PYTHON_EXECUTABLE),
        str(_LIMIT_EXEC_PATH),
        str(limits.cpu_seconds),
        str(limits.memory_bytes),
        str(limits.file_bytes),
        str(limits.open_files),
        str(limits.processes),
        "--",
        *command,
    ]


def fresh_home(scratch_dir: Path, name: str) -> Path:
    """An empty HOME for one child run, so user config stays out."""
    home = scratch_dir / name
    home.mkdir(parents=True, exist_ok=True)
    return home


def make_temp_dir(prefix: str = "omp-gym-") -> Path:
    """One fresh temporary directory for profiles and homes."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def sandbox_enabled(environ: Mapping[str, str]) -> bool:
    """Isolation is on unless OMP_GYM_SANDBOX=0 opts out."""
    return environ.get("OMP_GYM_SANDBOX", "1") != "0"
