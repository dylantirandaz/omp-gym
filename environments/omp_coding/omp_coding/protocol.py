"""Measure the native OpenAI tool protocol of one model endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from .hardware import MetalFailure, metal_preflight

ToolName: TypeAlias = Literal[
    "sandbox_read",
    "sandbox_write",
    "sandbox_edit",
    "sandbox_exec",
    "run_tests",
]
ProbeName: TypeAlias = Literal["read", "write", "edit", "execute", "test"]
JsonScalar: TypeAlias = str | int | float | bool | None

MAX_HTTP_RESPONSE_BYTES = 1024 * 1024
LOCAL_SERVER_START_SECONDS = 900.0
LOCAL_SERVER_STOP_SECONDS = 10.0
PROTOCOL_MODEL_NAME = "default_model"
PROTOCOL_TASK_MARKER = "\nYou work in an isolated Linux workspace"
PROTOCOL_TASK_CONTEXT = """\

You work in an isolated Linux workspace at /workspace.
This request checks the OMP tool protocol.
The declared files are fixture.txt, answer.txt, and test_fixture.py.
Use only sandbox_read, sandbox_write, sandbox_edit, sandbox_exec, and run_tests.
Follow the user request and call exactly the requested tool one time.
Do not call a different tool."""


@dataclass(frozen=True)
class ProtocolFailure:
    reason: str


@dataclass(frozen=True)
class ProtocolGateFailure:
    reason: str


@dataclass(frozen=True)
class ProtocolContext:
    system_prompt: str
    tools: tuple[Mapping[str, object], ...]
    sha256: str


@dataclass(frozen=True)
class ProtocolProbe:
    name: ProbeName
    expected_tool: ToolName
    expected_arguments: tuple[tuple[str, JsonScalar], ...]
    prompt: str


@dataclass(frozen=True)
class PassedProbe:
    status: Literal["passed"]
    name: ProbeName
    expected_tool: ToolName
    finish_reason: str
    parsed_calls: int
    invalid_calls: int
    completion_tokens: int
    ended: bool
    looped: bool


@dataclass(frozen=True)
class RejectedProbe:
    status: Literal["rejected"]
    name: ProbeName
    expected_tool: ToolName
    reason: str
    finish_reason: str | None
    parsed_calls: int
    invalid_calls: int
    completion_tokens: int
    ended: bool
    looped: bool
    response_text: str | None


ProbeResult: TypeAlias = PassedProbe | RejectedProbe


@dataclass(frozen=True)
class ProtocolReport:
    model: str
    parser: str
    context_sha256: str
    probes: tuple[ProbeResult, ...]
    valid_call_rate: float
    parsed_call_rate: float
    invalid_tool_rate: float
    end_token_rate: float
    loop_rate: float


@dataclass(frozen=True)
class ModelProtocolSuccess:
    status: Literal["passed"]
    model: str
    report: ProtocolReport


@dataclass(frozen=True)
class ModelProtocolFailure:
    status: Literal["failed"]
    model: str
    reason: str
    report: ProtocolReport | None


ModelProtocolResult: TypeAlias = ModelProtocolSuccess | ModelProtocolFailure


@dataclass(frozen=True)
class PassedModelSelection:
    status: Literal["passed"]
    selected_model: str
    candidates: tuple[ModelProtocolResult, ...]


@dataclass(frozen=True)
class RejectedModelSelection:
    status: Literal["rejected"]
    reason: str
    candidates: tuple[ModelProtocolResult, ...]


ModelSelectionReport: TypeAlias = PassedModelSelection | RejectedModelSelection


@dataclass(frozen=True)
class PassedProtocolCheck:
    status: Literal["passed"]
    report: ProtocolReport


@dataclass(frozen=True)
class RejectedProtocolCheck:
    status: Literal["rejected"]
    reason: str
    report: ProtocolReport


ProtocolCheck: TypeAlias = PassedProtocolCheck | RejectedProtocolCheck


@dataclass(frozen=True)
class RunningLocalModelServer:
    base_url: str
    api_key: str
    model: str
    parser: str
    process: subprocess.Popen[bytes]
    temporary_root: Path
    log_path: Path


PROBES = (
    ProtocolProbe(
        name="read",
        expected_tool="sandbox_read",
        expected_arguments=(("i", "Read protocol fixture"), ("path", "fixture.txt")),
        prompt=(
            "Call sandbox_read exactly once. Use i=Read protocol fixture and "
            "path=fixture.txt. Do not call a different tool."
        ),
    ),
    ProtocolProbe(
        name="write",
        expected_tool="sandbox_write",
        expected_arguments=(
            ("i", "Write protocol fixture"),
            ("path", "answer.txt"),
            ("content", "ok\n"),
        ),
        prompt=(
            "Call sandbox_write exactly once. Use i=Write protocol fixture, "
            "path=answer.txt, and content equal to ok followed by one newline."
        ),
    ),
    ProtocolProbe(
        name="edit",
        expected_tool="sandbox_edit",
        expected_arguments=(
            ("i", "Edit protocol fixture"),
            ("path", "answer.txt"),
            ("old_text", "old"),
            ("new_text", "new"),
        ),
        prompt=(
            "Call sandbox_edit exactly once. Use i=Edit protocol fixture, "
            "path=answer.txt, old_text=old, and new_text=new."
        ),
    ),
    ProtocolProbe(
        name="execute",
        expected_tool="sandbox_exec",
        expected_arguments=(
            ("i", "Execute protocol fixture"),
            ("command", "python test_fixture.py"),
        ),
        prompt=(
            "Call sandbox_exec exactly once. Use i=Execute protocol fixture and "
            "command=python test_fixture.py."
        ),
    ),
    ProtocolProbe(
        name="test",
        expected_tool="run_tests",
        expected_arguments=(("i", "Run sealed protocol tests"),),
        prompt=(
            "Call run_tests exactly once. Use i=Run sealed protocol tests. "
            "Do not call a different tool."
        ),
    ),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json_lines(path: Path) -> list[dict[str, object]] | ProtocolFailure:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return ProtocolFailure(f"protocol data is not readable: {path}: {error}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            return ProtocolFailure(
                f"protocol data has invalid JSON at {path}:{line_number}: {error}"
            )
        if not isinstance(value, dict):
            return ProtocolFailure(
                f"protocol data row is not an object at {path}:{line_number}"
            )
        rows.append(value)
    if not rows:
        return ProtocolFailure(f"protocol data file is empty: {path}")
    return rows


def _row_protocol_context(
    row: Mapping[str, object],
) -> tuple[str, tuple[Mapping[str, object], ...]] | ProtocolFailure:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return ProtocolFailure("protocol data messages are missing")
    first_message = raw_messages[0]
    if not isinstance(first_message, Mapping) or first_message.get("role") != "system":
        return ProtocolFailure("protocol data does not start with a system message")
    system_prompt = first_message.get("content")
    if not isinstance(system_prompt, str) or not system_prompt:
        return ProtocolFailure("protocol system prompt is empty")
    raw_tools = row.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        return ProtocolFailure("protocol tool contract is missing")
    tools: list[Mapping[str, object]] = []
    names: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            return ProtocolFailure("protocol tool entry is not an object")
        function = raw_tool.get("function")
        if raw_tool.get("type") != "function" or not isinstance(function, Mapping):
            return ProtocolFailure("protocol tool entry is not an OpenAI function")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            return ProtocolFailure("protocol tool function is invalid")
        if name in names:
            return ProtocolFailure(f"protocol tool name is duplicated: {name}")
        names.add(name)
        tools.append(dict(raw_tool))
    expected_names = {probe.expected_tool for probe in PROBES}
    if names != expected_names:
        return ProtocolFailure(
            f"protocol tool names differ: expected {sorted(expected_names)}, "
            f"got {sorted(names)}"
        )
    return system_prompt, tuple(tools)


def _protocol_system_core(system_prompt: str) -> str | ProtocolFailure:
    marker_index = system_prompt.find(PROTOCOL_TASK_MARKER)
    if marker_index < 0:
        return ProtocolFailure("protocol system prompt has no task context marker")
    lines = system_prompt[:marker_index].splitlines(keepends=True)
    model_lines = [line for line in lines if line.startswith("- Model: ")]
    date_lines = [line for line in lines if line.startswith("Today: ")]
    if len(model_lines) != 1 or len(date_lines) != 1:
        return ProtocolFailure("protocol system prompt metadata is invalid")
    return "".join(
        line
        for line in lines
        if not line.startswith("- Model: ") and not line.startswith("Today: ")
    )


def load_protocol_context(data_dir: Path) -> ProtocolContext | ProtocolFailure:
    """Load one normalized OMP prompt and tool contract from the training data."""
    system_core: str | None = None
    baseline_tools: tuple[Mapping[str, object], ...] | None = None
    for file_name in ("train.jsonl", "valid.jsonl"):
        rows = _read_json_lines(data_dir / file_name)
        if isinstance(rows, ProtocolFailure):
            return rows
        for row in rows:
            context = _row_protocol_context(row)
            if isinstance(context, ProtocolFailure):
                return context
            row_system_prompt, row_tools = context
            row_core = _protocol_system_core(row_system_prompt)
            if isinstance(row_core, ProtocolFailure):
                return row_core
            if system_core is None or baseline_tools is None:
                system_core = row_core
                baseline_tools = row_tools
            elif row_core != system_core:
                return ProtocolFailure(
                    "training samples do not use one fixed OMP system prompt"
                )
            elif _canonical_json(row_tools) != _canonical_json(baseline_tools):
                return ProtocolFailure(
                    "training samples do not use one fixed tool contract"
                )
    if system_core is None or baseline_tools is None:
        return ProtocolFailure("protocol data has no rows")
    system_prompt = system_core + PROTOCOL_TASK_CONTEXT
    context_document = {
        "system_prompt": system_prompt,
        "tools": baseline_tools,
    }
    return ProtocolContext(
        system_prompt=system_prompt,
        tools=baseline_tools,
        sha256=hashlib.sha256(_canonical_json(context_document)).hexdigest(),
    )


def _schema_type_error(value: object, expected: str) -> str | None:
    if expected == "string":
        return None if isinstance(value, str) else "must be a string"
    if expected == "integer":
        return (
            None
            if isinstance(value, int) and not isinstance(value, bool)
            else "must be an integer"
        )
    if expected == "number":
        return (
            None
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "must be a number"
        )
    if expected == "boolean":
        return None if isinstance(value, bool) else "must be a boolean"
    return f"uses unsupported schema type {expected}"


def validate_tool_arguments(
    schema: Mapping[str, object], arguments: Mapping[str, object]
) -> str | None:
    """Validate the strict object schema used by the five OMP tools."""
    if schema.get("type") != "object":
        return "tool parameters are not an object schema"
    raw_properties = schema.get("properties")
    if not isinstance(raw_properties, Mapping):
        return "tool parameter properties are missing"
    raw_required = schema.get("required", [])
    if not isinstance(raw_required, list) or not all(
        isinstance(name, str) for name in raw_required
    ):
        return "tool parameter required list is invalid"
    unknown = set(arguments).difference(raw_properties)
    if unknown and schema.get("additionalProperties") is False:
        return f"tool arguments contain unknown fields: {sorted(unknown)}"
    missing = set(raw_required).difference(arguments)
    if missing:
        return f"tool arguments are missing fields: {sorted(missing)}"
    for name, value in arguments.items():
        raw_property = raw_properties.get(name)
        if not isinstance(raw_property, Mapping):
            return f"tool argument schema is missing: {name}"
        expected_type = raw_property.get("type")
        if not isinstance(expected_type, str):
            return f"tool argument type is missing: {name}"
        type_error = _schema_type_error(value, expected_type)
        if type_error is not None:
            return f"tool argument {name} {type_error}"
        if isinstance(value, int) and not isinstance(value, bool):
            minimum = raw_property.get("minimum")
            maximum = raw_property.get("maximum")
            if isinstance(minimum, int) and value < minimum:
                return f"tool argument {name} is below its minimum"
            if isinstance(maximum, int) and value > maximum:
                return f"tool argument {name} is above its maximum"
    return None


def _tool_schemas(
    tools: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]] | ProtocolFailure:
    schemas: dict[str, Mapping[str, object]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            return ProtocolFailure("OpenAI tool function is missing")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            return ProtocolFailure("OpenAI tool function schema is invalid")
        schemas[name] = parameters
    return schemas


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        _request: urllib.request.Request,
        _file_pointer: object,
        _code: int,
        _message: str,
        _headers: Mapping[str, str],
        _new_url: str,
    ) -> None:
        return None


def _validate_base_url(base_url: str) -> ProtocolFailure | None:
    try:
        parsed = urllib.parse.urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        return ProtocolFailure(f"protocol endpoint URL is invalid: {error}")
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return ProtocolFailure("protocol endpoint URL is invalid")
    is_loopback_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "::1",
    }
    if parsed.scheme != "https" and not is_loopback_http:
        return ProtocolFailure(
            "protocol endpoint must use HTTPS or an HTTP loopback address"
        )
    if port is not None and not 1 <= port <= 65535:
        return ProtocolFailure("protocol endpoint port is invalid")
    return None


def _post_json(
    url: str,
    payload: Mapping[str, object],
    *,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, object] | ProtocolFailure:
    request = urllib.request.Request(  # noqa: S310 - explicit API endpoint.
        url,
        data=_canonical_json(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            body = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        return ProtocolFailure(f"protocol request failed: {error}")
    if len(body) > MAX_HTTP_RESPONSE_BYTES:
        return ProtocolFailure("protocol response exceeded the size limit")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return ProtocolFailure(f"protocol response is not JSON: {error}")
    if not isinstance(value, dict):
        return ProtocolFailure("protocol response is not an object")
    return value


def _response_choice(
    response: Mapping[str, object],
) -> tuple[Mapping[str, object], str, int] | ProtocolFailure:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return ProtocolFailure("protocol response must contain one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ProtocolFailure("protocol response choice is invalid")
    message = choice.get("message")
    finish_reason = choice.get("finish_reason")
    if not isinstance(message, Mapping) or not isinstance(finish_reason, str):
        return ProtocolFailure("protocol response message is invalid")
    usage = response.get("usage")
    completion_tokens = 0
    if isinstance(usage, Mapping):
        raw_completion_tokens = usage.get("completion_tokens")
        if isinstance(raw_completion_tokens, int) and not isinstance(
            raw_completion_tokens, bool
        ):
            completion_tokens = raw_completion_tokens
    return message, finish_reason, completion_tokens


def _call_signature(name: str, arguments: Mapping[str, object]) -> bytes:
    return _canonical_json({"name": name, "arguments": arguments})


def _probe_result(
    probe: ProtocolProbe,
    response: Mapping[str, object],
    schemas: Mapping[str, Mapping[str, object]],
) -> ProbeResult:
    choice = _response_choice(response)
    if isinstance(choice, ProtocolFailure):
        return RejectedProbe(
            status="rejected",
            name=probe.name,
            expected_tool=probe.expected_tool,
            reason=choice.reason,
            finish_reason=None,
            parsed_calls=0,
            invalid_calls=0,
            completion_tokens=0,
            ended=False,
            looped=False,
            response_text=None,
        )
    message, finish_reason, completion_tokens = choice
    raw_content = message.get("content")
    response_text = raw_content if isinstance(raw_content, str) else None
    raw_calls = message.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return RejectedProbe(
            status="rejected",
            name=probe.name,
            expected_tool=probe.expected_tool,
            reason="OpenAI message.tool_calls is not a list",
            finish_reason=finish_reason,
            parsed_calls=0,
            invalid_calls=0,
            completion_tokens=completion_tokens,
            ended=finish_reason != "length",
            looped=finish_reason == "length",
            response_text=response_text,
        )
    invalid_reasons: list[str] = []
    parsed: list[tuple[str, Mapping[str, object]]] = []
    signatures: list[bytes] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping) or raw_call.get("type") != "function":
            invalid_reasons.append(f"call {index} is not an OpenAI function call")
            continue
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            invalid_reasons.append(f"call {index} has no function object")
            continue
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(raw_arguments, str):
            invalid_reasons.append(f"call {index} name or JSON arguments are invalid")
            continue
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            invalid_reasons.append(f"call {index} arguments are not JSON: {error}")
            continue
        if not isinstance(arguments, dict):
            invalid_reasons.append(f"call {index} arguments are not an object")
            continue
        schema = schemas.get(name)
        if schema is None:
            invalid_reasons.append(f"call {index} uses unavailable tool {name}")
            continue
        schema_error = validate_tool_arguments(schema, arguments)
        if schema_error is not None:
            invalid_reasons.append(f"call {index}: {schema_error}")
            continue
        parsed.append((name, arguments))
        signatures.append(_call_signature(name, arguments))
    repeated = len(signatures) != len(set(signatures))
    looped = finish_reason == "length" or repeated
    ended = finish_reason in {"stop", "tool_calls"}
    expected_arguments = dict(probe.expected_arguments)
    arguments_match = (
        len(parsed) == 1
        and parsed[0][0] == probe.expected_tool
        and all(
            name in parsed[0][1] and parsed[0][1][name] == value
            for name, value in probe.expected_arguments
        )
    )
    if invalid_reasons:
        reason = "; ".join(invalid_reasons)
    elif not arguments_match:
        reason = (
            f"expected one {probe.expected_tool} call with "
            f"{expected_arguments}, got {parsed}"
        )
    elif finish_reason != "tool_calls":
        reason = f"expected finish_reason tool_calls, got {finish_reason}"
    else:
        return PassedProbe(
            status="passed",
            name=probe.name,
            expected_tool=probe.expected_tool,
            finish_reason=finish_reason,
            parsed_calls=len(parsed),
            invalid_calls=0,
            completion_tokens=completion_tokens,
            ended=ended,
            looped=looped,
        )
    return RejectedProbe(
        status="rejected",
        name=probe.name,
        expected_tool=probe.expected_tool,
        reason=reason,
        finish_reason=finish_reason,
        parsed_calls=len(parsed),
        invalid_calls=max(len(invalid_reasons), len(raw_calls) - len(parsed)),
        completion_tokens=completion_tokens,
        ended=ended,
        looped=looped,
        response_text=response_text,
    )


def run_protocol_gate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    parser: str,
    context: ProtocolContext,
    timeout_seconds: float = 120.0,
    max_tokens: int = 256,
) -> ProtocolReport | ProtocolFailure:
    """Require native parsed tool calls from a real OpenAI endpoint."""
    if timeout_seconds <= 0:
        return ProtocolFailure("protocol timeout must be positive")
    if max_tokens < 1:
        return ProtocolFailure("protocol maximum tokens must be positive")
    endpoint_failure = _validate_base_url(base_url)
    if endpoint_failure is not None:
        return endpoint_failure
    schemas = _tool_schemas(context.tools)
    if isinstance(schemas, ProtocolFailure):
        return schemas
    results: list[ProbeResult] = []
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    for probe in PROBES:
        response = _post_json(
            endpoint,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": context.system_prompt},
                    {"role": "user", "content": probe.prompt},
                ],
                "tools": list(context.tools),
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(response, ProtocolFailure):
            return response
        results.append(_probe_result(probe, response, schemas))
    total = len(results)
    valid = sum(isinstance(result, PassedProbe) for result in results)
    parsed_turns = sum(result.parsed_calls > 0 for result in results)
    parsed_calls = sum(result.parsed_calls for result in results)
    invalid_calls = sum(result.invalid_calls for result in results)
    ended = sum(result.ended for result in results)
    looped = sum(result.looped for result in results)
    return ProtocolReport(
        model=model,
        parser=parser,
        context_sha256=context.sha256,
        probes=tuple(results),
        valid_call_rate=valid / total,
        parsed_call_rate=parsed_turns / total,
        invalid_tool_rate=(
            invalid_calls / (parsed_calls + invalid_calls)
            if parsed_calls + invalid_calls
            else 0.0
        ),
        end_token_rate=ended / total,
        loop_rate=looped / total,
    )


def protocol_gate_failure(
    report: ProtocolReport,
) -> ProtocolGateFailure | None:
    """Require exact parsed calls and clean termination for every probe."""
    if report.valid_call_rate != 1.0:
        return ProtocolGateFailure(
            f"valid parsed tool-call rate is {report.valid_call_rate:.3f}, not 1.000"
        )
    if report.parsed_call_rate != 1.0:
        return ProtocolGateFailure(
            f"parsed tool-call rate is {report.parsed_call_rate:.3f}, not 1.000"
        )
    if report.invalid_tool_rate != 0.0:
        return ProtocolGateFailure(
            f"invalid tool-call rate is {report.invalid_tool_rate:.3f}, not 0.000"
        )
    if report.end_token_rate != 1.0:
        return ProtocolGateFailure(
            f"end-token rate is {report.end_token_rate:.3f}, not 1.000"
        )
    if report.loop_rate != 0.0:
        return ProtocolGateFailure(
            f"tool-call loop rate is {report.loop_rate:.3f}, not 0.000"
        )
    return None


def _loopback_port() -> int | ProtocolFailure:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
    except OSError as error:
        return ProtocolFailure(f"could not allocate a local server port: {error}")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        return ProtocolFailure("local server returned an invalid port")
    return port


def _tokenizer_parser(model: str) -> str | ProtocolFailure:
    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError as error:
        return ProtocolFailure(f"MLX-LM is not installed: {error}")
    try:
        tokenizer = load_tokenizer(model)
    except Exception as error:
        return ProtocolFailure(f"model tokenizer could not be loaded: {error}")
    tool_parser = getattr(tokenizer, "tool_parser", None)
    if tool_parser is None:
        return "none"
    module = getattr(tool_parser, "__module__", None)
    name = getattr(tool_parser, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        return ProtocolFailure("model tool parser identity is invalid")
    return f"{module}.{name}"


def _server_ready(base_url: str, process: subprocess.Popen[bytes]) -> str | None:
    deadline = time.monotonic() + LOCAL_SERVER_START_SECONDS
    models_url = f"{base_url}/models"
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            return f"MLX server exited with status {exit_code}"
        request = urllib.request.Request(  # noqa: S310 - loopback health check.
            models_url,
            headers={"Authorization": "Bearer local-protocol-gate"},
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - loopback health check.
                request, timeout=2.0
            ) as response:
                if response.status == 200:
                    return None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    return "MLX server did not become ready"


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=LOCAL_SERVER_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=LOCAL_SERVER_STOP_SECONDS)


def start_local_model_server(
    *,
    model: str,
    max_tokens: int,
) -> RunningLocalModelServer | ProtocolFailure:
    """Start one loopback MLX server on the Metal GPU."""
    if max_tokens < 1:
        return ProtocolFailure("local server maximum tokens must be positive")
    metal = metal_preflight()
    if isinstance(metal, MetalFailure):
        return ProtocolFailure(metal.reason)
    parser = _tokenizer_parser(model)
    if isinstance(parser, ProtocolFailure):
        return parser
    port = _loopback_port()
    if isinstance(port, ProtocolFailure):
        return port
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix="omp-protocol-"))
    except OSError as error:
        return ProtocolFailure(
            f"MLX server log directory could not be created: {error}"
        )
    log_path = temporary_root / "server.log"
    base_url = f"http://127.0.0.1:{port}/v1"
    api_key = "local-protocol-gate"
    server_code = (
        "import mlx.core as mx; "
        "mx.set_default_device(mx.gpu); "
        "from mlx_lm.server import main; "
        "main()"
    )
    try:
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(  # noqa: S603 - fixed Python module.
                [
                    sys.executable,
                    "-c",
                    server_code,
                    "--model",
                    model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--max-tokens",
                    str(max_tokens),
                    "--decode-concurrency",
                    "1",
                    "--prompt-concurrency",
                    "1",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
    except OSError as error:
        shutil.rmtree(temporary_root, ignore_errors=True)
        return ProtocolFailure(f"MLX server could not start: {error}")
    readiness_error = _server_ready(base_url, process)
    if readiness_error is not None:
        _stop_server(process)
        try:
            detail = log_path.read_text(encoding="utf-8")[-4000:]
        except (OSError, UnicodeDecodeError):
            detail = ""
        shutil.rmtree(temporary_root, ignore_errors=True)
        return ProtocolFailure(f"{readiness_error}: {detail or 'no server output'}")
    return RunningLocalModelServer(
        base_url=base_url,
        api_key=api_key,
        model=model,
        parser=parser,
        process=process,
        temporary_root=temporary_root,
        log_path=log_path,
    )


def stop_local_model_server(
    server: RunningLocalModelServer,
) -> ProtocolFailure | None:
    """Stop one server and remove its private log directory."""
    try:
        _stop_server(server.process)
        shutil.rmtree(server.temporary_root)
    except (OSError, subprocess.SubprocessError) as error:
        return ProtocolFailure(f"MLX server could not stop cleanly: {error}")
    return None


def run_local_protocol_gate(
    *,
    model: str,
    context: ProtocolContext,
    max_tokens: int = 256,
) -> ProtocolReport | ProtocolFailure:
    """Start one Metal MLX server and test its native tool response."""
    server = start_local_model_server(model=model, max_tokens=max_tokens)
    if isinstance(server, ProtocolFailure):
        return server
    try:
        report = run_protocol_gate(
            base_url=server.base_url,
            api_key=server.api_key,
            model=PROTOCOL_MODEL_NAME,
            parser=server.parser,
            context=context,
            max_tokens=max_tokens,
        )
    finally:
        stop_failure = stop_local_model_server(server)
    if stop_failure is not None:
        return stop_failure
    if isinstance(report, ProtocolFailure):
        return report
    return ProtocolReport(
        model=model,
        parser=report.parser,
        context_sha256=report.context_sha256,
        probes=report.probes,
        valid_call_rate=report.valid_call_rate,
        parsed_call_rate=report.parsed_call_rate,
        invalid_tool_rate=report.invalid_tool_rate,
        end_token_rate=report.end_token_rate,
        loop_rate=report.loop_rate,
    )


def select_protocol_model(
    *,
    models: Sequence[str],
    context: ProtocolContext,
) -> ModelSelectionReport | ProtocolFailure:
    """Select the model with the best measured native protocol result."""
    if len(models) < 2:
        return ProtocolFailure("model selection needs at least two candidates")
    if len(set(models)) != len(models):
        return ProtocolFailure("model candidates must be unique")
    results: list[ModelProtocolResult] = []
    passing: list[ModelProtocolSuccess] = []
    for model in models:
        report = run_local_protocol_gate(model=model, context=context)
        if isinstance(report, ProtocolFailure):
            results.append(
                ModelProtocolFailure(
                    status="failed",
                    model=model,
                    reason=report.reason,
                    report=None,
                )
            )
            continue
        gate_failure = protocol_gate_failure(report)
        if gate_failure is not None:
            results.append(
                ModelProtocolFailure(
                    status="failed",
                    model=model,
                    reason=gate_failure.reason,
                    report=report,
                )
            )
            continue
        success = ModelProtocolSuccess(status="passed", model=model, report=report)
        results.append(success)
        passing.append(success)
    if not passing:
        return RejectedModelSelection(
            status="rejected",
            reason="no base model passed the exact native protocol gate",
            candidates=tuple(results),
        )
    selected = max(
        passing,
        key=lambda item: (
            item.report.valid_call_rate,
            item.report.parsed_call_rate,
            -item.report.invalid_tool_rate,
            item.report.end_token_rate,
            -item.report.loop_rate,
        ),
    )
    return PassedModelSelection(
        status="passed",
        selected_model=selected.model,
        candidates=tuple(results),
    )


def _write_report(path: Path, value: object) -> ProtocolFailure | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
    except OSError as error:
        return ProtocolFailure(f"protocol output could not be created: {error}")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                asdict(value),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return ProtocolFailure(f"protocol output could not be written: {error}")
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("check-local")
    local.add_argument("--data", type=Path, required=True)
    local.add_argument("--model", required=True)
    local.add_argument("--output", type=Path)
    select = subparsers.add_parser("select-local")
    select.add_argument("--data", type=Path, required=True)
    select.add_argument("--output", type=Path)
    select.add_argument("models", nargs="+")
    endpoint = subparsers.add_parser("check-endpoint")
    endpoint.add_argument("--data", type=Path, required=True)
    endpoint.add_argument("--base-url", required=True)
    endpoint.add_argument("--api-key-var", required=True)
    endpoint.add_argument("--model", required=True)
    endpoint.add_argument("--parser", required=True)
    endpoint.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    context = load_protocol_context(arguments.data)
    if isinstance(context, ProtocolFailure):
        result: object = context
    elif arguments.command == "check-local":
        result = run_local_protocol_gate(
            model=arguments.model,
            context=context,
        )
    elif arguments.command == "select-local":
        result = select_protocol_model(
            models=arguments.models,
            context=context,
        )
    elif arguments.command == "check-endpoint":
        api_key = os.environ.get(arguments.api_key_var)
        if api_key is None or not api_key:
            result = ProtocolFailure(
                f"API key environment variable is missing: {arguments.api_key_var}"
            )
        else:
            result = run_protocol_gate(
                base_url=arguments.base_url,
                api_key=api_key,
                model=arguments.model,
                parser=arguments.parser,
                context=context,
            )
    else:
        raise AssertionError(f"unknown command: {arguments.command}")
    if isinstance(result, ProtocolReport):
        gate_failure = protocol_gate_failure(result)
        if gate_failure is None:
            result = PassedProtocolCheck(status="passed", report=result)
        else:
            result = RejectedProtocolCheck(
                status="rejected",
                reason=gate_failure.reason,
                report=result,
            )
    if not isinstance(result, ProtocolFailure) and arguments.output is not None:
        write_failure = _write_report(arguments.output, result)
        if write_failure is not None:
            result = write_failure
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    failed = isinstance(
        result,
        (ProtocolFailure, RejectedProtocolCheck, RejectedModelSelection),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
