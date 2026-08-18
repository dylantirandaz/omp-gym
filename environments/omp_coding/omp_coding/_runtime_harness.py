"""Run OMP RPC with tools that call the host controller."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, "/opt")

from omp_rpc import RpcClient, RpcError, host_tool  # noqa: E402

MAX_CONFIG_BYTES = 1024 * 1024
TOOL_TIMEOUT_SECONDS = 620


@dataclass(frozen=True)
class HarnessConfig:
    executable: str
    provider: str
    model: str
    prompt: str
    system_prompt: str


@dataclass(frozen=True)
class ToolEndpoint:
    url: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("tool controller URL must use HTTP or HTTPS")

    def call(self, action: str, arguments: Mapping[str, object]) -> str:
        body = json.dumps(
            {"action": action, "arguments": dict(arguments)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        request = Request(  # noqa: S310 - the host supplies a validated HTTP URL.
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            # The host supplies a validated HTTP URL.
            with urlopen(request, timeout=TOOL_TIMEOUT_SECONDS) as response:  # noqa: S310
                raw = response.read(MAX_CONFIG_BYTES + 1)
        except HTTPError as error:
            detail = error.read(MAX_CONFIG_BYTES).decode(errors="replace")
            raise RuntimeError(
                f"tool controller returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"tool controller is unavailable: {error.reason}"
            ) from error
        if len(raw) > MAX_CONFIG_BYTES:
            raise RuntimeError("tool controller response is too large")
        try:
            reply = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"tool controller returned invalid JSON: {error}"
            ) from error
        if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
            raise RuntimeError("tool controller returned an invalid reply")
        if reply["ok"] is not True:
            reason = reply.get("error")
            raise RuntimeError(
                reason if isinstance(reason, str) else "tool operation failed"
            )
        text = reply.get("text")
        if isinstance(text, str):
            return text
        details = {key: value for key, value in reply.items() if key != "ok"}
        return json.dumps(details, ensure_ascii=False, sort_keys=True)


def _required_string(document: Mapping[str, object], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _load_config(path: Path) -> HarnessConfig:
    try:
        size = path.stat().st_size
        if size > MAX_CONFIG_BYTES:
            raise ValueError("harness configuration is too large")
        document = json.loads(path.read_bytes())
    except OSError as error:
        raise ValueError(f"harness configuration is not readable: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"harness configuration is invalid: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("harness configuration must be an object")
    return HarnessConfig(
        executable=_required_string(document, "executable"),
        provider=_required_string(document, "provider"),
        model=_required_string(document, "model"),
        prompt=_required_string(document, "prompt"),
        system_prompt=_required_string(document, "system_prompt"),
    )


def _tool_endpoint() -> ToolEndpoint:
    url = os.environ.get("OMP_CODING_TOOL_URL")
    if not url:
        raise ValueError("OMP_CODING_TOOL_URL is not set")
    return ToolEndpoint(url=url)


def _tools(endpoint: ToolEndpoint) -> tuple[object, ...]:
    empty_object = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return (
        host_tool(
            name="sandbox_read",
            description=(
                "Read one UTF-8 file or list one directory in the task workspace. "
                "Paths are relative to /workspace."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 4000},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            execute=lambda arguments, _context: endpoint.call("read", arguments),
        ),
        host_tool(
            name="sandbox_write",
            description="Replace one declared editable file in the task workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            execute=lambda arguments, _context: endpoint.call("write", arguments),
        ),
        host_tool(
            name="sandbox_edit",
            description=(
                "Replace one exact and unique text part in one declared editable file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            execute=lambda arguments, _context: endpoint.call("edit", arguments),
        ),
        host_tool(
            name="sandbox_exec",
            description=(
                "Run one shell command in the task workspace. "
                "The command has no network or secret data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            execute=lambda arguments, _context: endpoint.call("exec", arguments),
        ),
        host_tool(
            name="run_tests",
            description=(
                "Run the sealed verifier in a separate container. "
                "Return only the pass count."
            ),
            parameters=empty_object,
            execute=lambda arguments, _context: endpoint.call("tests", arguments),
        ),
    )


def _fail(reason: str) -> NoReturn:
    print(reason, file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if len(sys.argv) != 2:
        _fail("usage: _runtime_harness.py CONFIG")
    try:
        config = _load_config(Path(sys.argv[1]))
        endpoint = _tool_endpoint()
    except ValueError as error:
        _fail(str(error))

    client = RpcClient(
        executable=config.executable,
        provider=config.provider,
        model=config.model,
        cwd="/workspace",
        append_system_prompt=config.system_prompt,
        tools=(),
        custom_tools=_tools(endpoint),
        no_session=True,
        no_skills=True,
        no_rules=True,
        no_title=True,
        extra_args=(
            "--no-extensions",
            "--no-lsp",
            "--no-pty",
            "--auto-approve",
        ),
        startup_timeout=30.0,
        request_timeout=60.0,
        max_event_history=20_000,
        max_stderr_chunks=512,
    )
    try:
        client.start()
        client.install_headless_ui()
        turn = client.prompt_and_wait(config.prompt, timeout=TOOL_TIMEOUT_SECONDS)
        print(
            json.dumps(
                {"assistant_text": turn.assistant_text},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 0
    except (OSError, RpcError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        client.stop()


if __name__ == "__main__":
    raise SystemExit(main())
