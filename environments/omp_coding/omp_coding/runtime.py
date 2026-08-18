"""Shared contracts for the Prime Verifiers v1 runtime."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RUNTIME_IMAGE = (
    "node@sha256:934240a162082fd8b8a2f90cd5114446443f1eba1c5378f6687167ca405e6584"
)
RUNTIME_IMAGE_DIGEST = (
    "sha256:934240a162082fd8b8a2f90cd5114446443f1eba1c5378f6687167ca405e6584"
)
WORKSPACE_BYTES = 256 * 1024 * 1024
TEMP_BYTES = 64 * 1024 * 1024
HOME_BYTES = 16 * 1024 * 1024
MAX_FILES = 4_000
MAX_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
MEMORY_BYTES = 768 * 1024 * 1024
PID_LIMIT = 64
CPU_LIMIT = 1.0
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
TREE_DIGEST_DOMAIN = b"omp-gym-tree-v2\0"


@dataclass(frozen=True)
class RuntimeFailure:
    """One runtime operation did not complete."""

    kind: Literal[
        "command_failed",
        "invalid_workspace",
        "output_limit",
        "protocol_error",
        "process_leak",
        "timeout",
        "unavailable",
    ]
    reason: str
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class WorkspaceInventory:
    """The regular-file inventory of one task tree."""

    files: int
    bytes: int
    digest: str


def inspect_plain_tree(root: Path) -> WorkspaceInventory | RuntimeFailure:
    """Hash a bounded tree and reject links and special files."""
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        return RuntimeFailure(
            "invalid_workspace", f"workspace is not readable: {error}"
        )
    if not resolved_root.is_dir():
        return RuntimeFailure("invalid_workspace", "workspace is not a directory")

    digest = hashlib.sha256(TREE_DIGEST_DOMAIN)
    file_count = 0
    total_bytes = 0
    stack = [resolved_root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            return RuntimeFailure(
                "invalid_workspace", f"workspace scan failed: {error}"
            )
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                return RuntimeFailure(
                    "invalid_workspace",
                    f"workspace scan failed: {error}",
                )
            path = Path(entry.path)
            relative = path.relative_to(resolved_root).as_posix()
            mode = metadata.st_mode
            if stat.S_ISDIR(mode):
                encoded_path = relative.encode()
                digest.update(b"D")
                digest.update(len(encoded_path).to_bytes(8, "big"))
                digest.update(encoded_path)
                stack.append(path)
                continue
            if not stat.S_ISREG(mode):
                return RuntimeFailure(
                    "invalid_workspace",
                    f"workspace contains a link or special file: {relative}",
                )

            file_count += 1
            total_bytes += metadata.st_size
            if file_count > MAX_FILES:
                return RuntimeFailure(
                    "invalid_workspace",
                    f"workspace has more than {MAX_FILES} files",
                )
            if total_bytes > WORKSPACE_BYTES:
                return RuntimeFailure(
                    "invalid_workspace",
                    f"workspace is larger than {WORKSPACE_BYTES // (1024 * 1024)} MiB",
                )

            encoded_path = relative.encode()
            digest.update(b"F")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            digest.update(metadata.st_size.to_bytes(8, "big"))
            bytes_read = 0
            try:
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        bytes_read += len(chunk)
                        digest.update(chunk)
            except OSError as error:
                return RuntimeFailure(
                    "invalid_workspace",
                    f"workspace read failed: {error}",
                )
            if bytes_read != metadata.st_size:
                return RuntimeFailure(
                    "invalid_workspace",
                    "workspace changed while it was read",
                )
    return WorkspaceInventory(file_count, total_bytes, digest.hexdigest())
