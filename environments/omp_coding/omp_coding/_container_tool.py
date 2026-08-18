"""Small file service that runs inside an episode container."""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import NoReturn

WORKSPACE = Path("/workspace")
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_READ_LINES = 4000


def _fail(reason: str) -> NoReturn:
    print(json.dumps({"ok": False, "error": reason}, separators=(",", ":")))
    raise SystemExit(1)


def _request() -> dict[str, object]:
    if len(sys.argv) > 2:
        _fail("file service accepts zero or one request path")
    try:
        if len(sys.argv) == 2:
            with Path(sys.argv[1]).open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        else:
            raw = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            _fail("request is too large")
        value = json.loads(raw)
    except OSError as error:
        _fail(f"request is not readable: {error}")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid request: {error}")
    if not isinstance(value, dict):
        _fail("request must be an object")
    return value


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("path must be relative and normalized")
    return path


def _checked_parent(relative: Path, *, create: bool) -> Path:
    current = WORKSPACE
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                _fail(
                    f"parent directory does not exist: {current.relative_to(WORKSPACE)}"
                )
            current.mkdir(mode=0o755)
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(f"parent is not a plain directory: {current.relative_to(WORKSPACE)}")
    return current


def _read_plain_file(relative: Path) -> str:
    _checked_parent(relative, create=False)
    target = WORKSPACE / relative
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        _fail(f"file does not exist: {relative.as_posix()}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"path is not a plain file: {relative.as_posix()}")
    if metadata.st_size > MAX_FILE_BYTES:
        _fail(f"file is larger than {MAX_FILE_BYTES} bytes")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        data = os.read(descriptor, MAX_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) > MAX_FILE_BYTES:
        _fail(f"file is larger than {MAX_FILE_BYTES} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"file is not UTF-8 text: {relative.as_posix()}")


def _write_plain_file(relative: Path, content: str) -> None:
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        _fail(f"content is larger than {MAX_FILE_BYTES} bytes")
    _checked_parent(relative, create=True)
    target = WORKSPACE / relative
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None and not stat.S_ISREG(metadata.st_mode):
        _fail(f"path is not a plain file: {relative.as_posix()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o644)
    try:
        encoded = content.encode("utf-8")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_action(request: dict[str, object]) -> dict[str, object]:
    relative = _relative_path(request.get("path"))
    target = WORKSPACE / relative
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        _fail(f"path does not exist: {relative.as_posix()}")
    if stat.S_ISDIR(metadata.st_mode):
        entries: list[str] = []
        with os.scandir(target) as iterator:
            for entry in sorted(iterator, key=lambda item: item.name):
                item_metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(item_metadata.st_mode):
                    suffix = " -> symlink (blocked)"
                elif stat.S_ISDIR(item_metadata.st_mode):
                    suffix = "/"
                elif stat.S_ISREG(item_metadata.st_mode):
                    suffix = f" ({item_metadata.st_size} bytes)"
                else:
                    suffix = " (special file blocked)"
                entries.append(entry.name + suffix)
        return {"ok": True, "text": "\n".join(entries)}
    text = _read_plain_file(relative)
    start_value = request.get("line", 1)
    limit_value = request.get("limit", 400)
    if (
        isinstance(start_value, bool)
        or not isinstance(start_value, int)
        or start_value < 1
    ):
        _fail("line must be a positive integer")
    if (
        isinstance(limit_value, bool)
        or not isinstance(limit_value, int)
        or not 1 <= limit_value <= MAX_READ_LINES
    ):
        _fail(f"limit must be from 1 to {MAX_READ_LINES}")
    lines = text.splitlines()
    selected = lines[start_value - 1 : start_value - 1 + limit_value]
    numbered = [
        f"{index}:{line}" for index, line in enumerate(selected, start=start_value)
    ]
    return {
        "ok": True,
        "text": "\n".join(numbered),
        "total_lines": len(lines),
        "truncated": start_value - 1 + len(selected) < len(lines),
    }


def _write_action(request: dict[str, object]) -> dict[str, object]:
    relative = _relative_path(request.get("path"))
    content = request.get("content")
    if not isinstance(content, str):
        _fail("content must be a string")
    _write_plain_file(relative, content)
    return {"ok": True, "bytes": len(content.encode("utf-8"))}


def _edit_action(request: dict[str, object]) -> dict[str, object]:
    relative = _relative_path(request.get("path"))
    old_text = request.get("old_text")
    new_text = request.get("new_text")
    if not isinstance(old_text, str) or not old_text:
        _fail("old_text must be a non-empty string")
    if not isinstance(new_text, str):
        _fail("new_text must be a string")
    content = _read_plain_file(relative)
    count = content.count(old_text)
    if count != 1:
        _fail(f"old_text must occur exactly once; found {count}")
    updated = content.replace(old_text, new_text, 1)
    _write_plain_file(relative, updated)
    return {"ok": True, "bytes": len(updated.encode("utf-8"))}


def _scan_action() -> dict[str, object]:
    files = 0
    total_bytes = 0
    special: list[str] = []
    stack = [WORKSPACE]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as iterator:
            for entry in iterator:
                metadata = entry.stat(follow_symlinks=False)
                relative = Path(entry.path).relative_to(WORKSPACE).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    stack.append(Path(entry.path))
                elif stat.S_ISREG(metadata.st_mode):
                    files += 1
                    total_bytes += metadata.st_size
                else:
                    special.append(relative)
    return {
        "ok": not special,
        "files": files,
        "bytes": total_bytes,
        "special": sorted(special),
    }


def _reap_action() -> dict[str, object]:
    killed: set[int] = set()
    empty_scans = 0
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        active: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            process_id = int(entry.name)
            try:
                status = (entry / "status").read_text()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            user_line = next(
                (line for line in status.splitlines() if line.startswith("Uid:")),
                None,
            )
            if user_line is None:
                continue
            fields = user_line.split()
            if len(fields) >= 2 and fields[1] == "65534":
                active.append(process_id)
        if not active:
            empty_scans += 1
            if empty_scans == 2:
                return {"ok": True, "killed": sorted(killed)}
            time.sleep(0.02)
            continue
        empty_scans = 0
        for process_id in active:
            try:
                os.kill(process_id, signal.SIGKILL)
                killed.add(process_id)
            except (PermissionError, ProcessLookupError):
                continue
        time.sleep(0.02)
    _fail("sandbox processes did not stop")


def _remove_entry(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        with os.scandir(path) as iterator:
            for entry in iterator:
                _remove_entry(Path(entry.path))
        path.rmdir()
    else:
        path.unlink()


def _is_runtime_pidfile(entry: os.DirEntry[str]) -> bool:
    if not entry.name.startswith("vf-process-") or not entry.name.endswith(".pid"):
        return False
    try:
        metadata = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return metadata.st_uid == 0 and stat.S_ISREG(metadata.st_mode)


def _clean_action() -> dict[str, object]:
    reap = _reap_action()
    # Remove every scratch location writable by the candidate user.
    for root, keep in (
        (Path("/tmp"), {".omp-gym-ready"}),  # noqa: S108
        (Path("/var/tmp"), set()),  # noqa: S108
        (Path("/dev/shm"), set()),  # noqa: S108
        (Path("/home/solver"), set()),
    ):
        with os.scandir(root) as iterator:
            for entry in iterator:
                if entry.name in keep or (
                    root == Path("/tmp") and _is_runtime_pidfile(entry)  # noqa: S108
                ):
                    continue
                _remove_entry(Path(entry.path))
    return {"ok": True, "killed": reap["killed"]}


def main() -> int:
    request = _request()
    action = request.get("action")
    if action == "read":
        result = _read_action(request)
    elif action == "write":
        result = _write_action(request)
    elif action == "edit":
        result = _edit_action(request)
    elif action == "scan":
        result = _scan_action()
    elif action == "reap":
        result = _reap_action()
    elif action == "clean":
        result = _clean_action()
    else:
        _fail("unknown action")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
