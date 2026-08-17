"""OpenAI-compatible shim that restores tool calls over mlx-lm.

The mlx-lm server (0.32) accepts a `tools` parameter but swallows
the generated tool call: the response carries neither content nor
`tool_calls`. This shim sits in front of the server and closes the
gap with the same convention the omp-gym adapters are trained on:

- incoming `tools` schemas become a compact list appended to the
  system message;
- the upstream request runs without `tools`, so the model answers
  in plain text;
- tool-call JSON in the text is parsed into real OpenAI
  `tool_calls`, whether tagged, fenced, or bare;
- tool-free streaming requests proxy the upstream SSE stream chunk
  by chunk; streaming requests with tools buffer the upstream
  response, because tool-call extraction needs the full text, and
  are answered as one SSE burst;
- generated arguments are validated against each tool's JSON schema
  (value types, enum membership, `additionalProperties: false`)
  before they reach the agent; invalid calls are dropped and
  counted, shape-remapped rescues are counted and logged;
- capture records are dicts carrying the upstream messages, the raw
  completion text, a uuid request id, finish_reason, usage, and
  model, so RL training reads the text the policy actually sampled.
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_FENCED_CALL_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TAG_LINE_PATTERN = re.compile(r"\s*(?:\ufffd+|</?[A-Za-z_][\w-]*>)\s*")
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_UPSTREAM_BYTES = 64 * 1024 * 1024
_UPSTREAM_SLOTS = threading.BoundedSemaphore(8)

TOOL_PROTOCOL = (
    "\n\nTo call a tool, write one JSON object on its own line: "
    '{"name": ..., "arguments": {...}}. '
    "Available tools:\n"
)

_COUNTS: dict[str, int] = {"invalid": 0, "remapped": 0}


def counts_snapshot() -> dict[str, int]:
    """Return dropped-invalid and shape-remapped call totals."""
    return dict(_COUNTS)


def counts_reset() -> None:
    """Reset the drop/remap counters. Used by tests."""
    _COUNTS["invalid"] = 0
    _COUNTS["remapped"] = 0


@dataclass(frozen=True)
class AvailableTool:
    """One offered tool, its argument shape, and its JSON schema."""

    name: str
    argument_names: frozenset[str]
    required_argument_names: frozenset[str]
    schema: dict


def _available_tools(tools: object) -> tuple[AvailableTool, ...]:
    """Read valid function-tool shapes from an OpenAI request."""
    if not isinstance(tools, list):
        return ()
    tools_by_name: dict[str, AvailableTool] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        properties = parameters.get("properties")
        required = parameters.get("required")
        argument_names = (
            frozenset(key for key in properties if isinstance(key, str))
            if isinstance(properties, dict)
            else frozenset()
        )
        required_argument_names = (
            frozenset(item for item in required if isinstance(item, str))
            if isinstance(required, list)
            else frozenset()
        )
        tools_by_name[name] = AvailableTool(
            name=name,
            argument_names=argument_names,
            required_argument_names=required_argument_names,
            schema=parameters,
        )
    return tuple(tools_by_name.values())


def _value_matches_type(value: object, expected: str) -> bool:
    """Check one JSON value against a JSON-schema type name."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _arguments_match_schema(tool: AvailableTool, arguments: dict) -> bool:
    """Validate argument values against the tool's JSON schema.

    Checks declared value types, enum membership, and refuses extra
    keys when the schema declares `additionalProperties: false`.
    Properties without a type or enum are accepted as-is.
    """
    properties = tool.schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    if tool.schema.get("additionalProperties") is False:
        if not set(arguments) <= set(properties):
            return False
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if isinstance(expected, str) and not _value_matches_type(value, expected):
            return False
        enum = spec.get("enum")
        if isinstance(enum, list) and value not in enum:
            return False
    return True


def _resolve_tool_name(
    generated_name: str,
    arguments: dict,
    available_tools: tuple[AvailableTool, ...],
    remap: bool,
) -> str | None:
    """Resolve a valid exact name or one unique argument-shape match.

    The exact name must carry its required arguments and valid
    argument values (types, enums, and `additionalProperties: false`
    bounds come from the schema). With the remap knob on, an unknown
    name falls back to one unique argument-shape match; every rescue
    is counted and logged. It never reroutes an offered name to a
    different tool: episodes run with auto-approval, so a malformed
    read must not become a bash call.
    """
    argument_names = frozenset(arguments)
    for tool in available_tools:
        if tool.name != generated_name:
            continue
        if tool.required_argument_names <= argument_names and _arguments_match_schema(
            tool, arguments
        ):
            return generated_name
        return None
    if not remap or not argument_names:
        return None
    matches = [
        tool.name
        for tool in available_tools
        if tool.argument_names
        and tool.required_argument_names <= argument_names
        and argument_names <= tool.argument_names
        and _arguments_match_schema(tool, arguments)
    ]
    if len(matches) == 1:
        _COUNTS["remapped"] += 1
        print(f'shim: remapped call "{generated_name}" -> "{matches[0]}"')
        return matches[0]
    return None


def _tools_to_system_suffix(tools: list[dict]) -> str:
    """Render OpenAI tool schemas as a compact system-prompt list."""
    lines = []
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name", "")
        description = str(function.get("description", "")).split("\n")[0]
        parameters = json.dumps(function.get("parameters", {}))
        lines.append(f"- {name}: {description} parameters={parameters}")
    return TOOL_PROTOCOL + "\n".join(lines)


def _call_from_payload(
    payload: object,
    available_tools: tuple[AvailableTool, ...],
    remap: bool,
) -> dict | None:
    """Build one OpenAI tool call for an offered tool.

    Invalid payloads (bad name shape, wrong argument shape, or
    schema-invalid argument values) are dropped and counted.
    """
    if not isinstance(payload, dict):
        _COUNTS["invalid"] += 1
        return None
    generated_name = payload.get("name")
    arguments = payload.get("arguments")
    if (
        not isinstance(generated_name, str)
        or not generated_name
        or not isinstance(arguments, dict)
    ):
        _COUNTS["invalid"] += 1
        return None
    name = _resolve_tool_name(generated_name, arguments, available_tools, remap)
    if name is None:
        _COUNTS["invalid"] += 1
        return None
    return {
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _decode_one(text: str) -> object | None:
    """Parse strict JSON; raw control characters inside strings fail."""
    try:
        return json.loads(text, strict=True)
    except json.JSONDecodeError:
        return None


def _bare_json_calls(
    text: str,
    available_tools: tuple[AvailableTool, ...],
    remap: bool,
) -> tuple[list[dict], bool] | None:
    """Read a text that is only JSON calls between decoration lines.

    Small tuned models wrap bare JSON calls in improvised angle tags
    (for example <lemma>) or replacement characters. Keep the JSON
    objects when nothing else is present. Refuse prose. A truncated
    JSON tail drops; the completed calls before it still run, and
    the caller marks the response partial.
    """
    kept_lines = [
        line for line in text.splitlines() if not _TAG_LINE_PATTERN.fullmatch(line)
    ]
    stripped = "\n".join(kept_lines).strip()
    if not stripped.startswith("{"):
        return None
    decoder = json.JSONDecoder(strict=True)
    calls: list[dict] = []
    position = 0
    while position < len(stripped):
        if stripped[position].isspace():
            position += 1
            continue
        if stripped[position] != "{":
            return None
        try:
            payload, position = decoder.raw_decode(stripped, position)
        except json.JSONDecodeError:
            if calls:
                return calls, True
            return None
        call = _call_from_payload(payload, available_tools, remap)
        if call is not None:
            calls.append(call)
    if calls:
        return calls, False
    return None


def _extract_tool_calls(
    text: str,
    available_tools: tuple[AvailableTool, ...],
    remap: bool = True,
) -> tuple[str, list[dict], bool]:
    """Split generated text into plain content and valid tool calls.

    Three envelopes carry the same payload schema, depending on the
    prompt the model saw: <tool_call>{json}</tool_call> blocks (the
    trained form), fenced ```json blocks (markdown-heavy prompts),
    and a bare top-level JSON object (terse prompts). Returns the
    content, the calls, and a partial flag set when a truncated
    trailing JSON object dropped earlier completed calls from view.
    """
    for pattern in (_TOOL_CALL_PATTERN, _FENCED_CALL_PATTERN):
        found_calls = list(pattern.finditer(text))
        calls: list[dict] = []
        for found in found_calls:
            payload = _decode_one(found.group(1))
            if payload is None:
                _COUNTS["invalid"] += 1
                continue
            call = _call_from_payload(payload, available_tools, remap)
            if call is not None:
                calls.append(call)
        if calls:
            return pattern.sub("", text).strip(), calls, False

    scanned_calls = _bare_json_calls(text, available_tools, remap)
    if scanned_calls is not None:
        calls, partial = scanned_calls
        if calls:
            return "", calls, partial

    trimmed = text.strip().strip("\ufffd").strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        payload = _decode_one(trimmed)
        if payload is not None:
            call = _call_from_payload(payload, available_tools, remap)
            if call is not None:
                return "", [call], False
    return trimmed, [], False


def _forced_tool_name(
    tool_choice: object,
    available_tools: tuple[AvailableTool, ...],
) -> str | None:
    """Read a named function choice that names an offered tool."""
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    name = (
        function.get("name") if isinstance(function, dict) else tool_choice.get("name")
    )
    if not isinstance(name, str):
        return None
    if any(tool.name == name for tool in available_tools):
        return name
    return None


def _rewrite_request(
    body: dict,
    adapter_path: str | None = None,
    sample_temp: float | None = None,
) -> dict:
    """Move tools into the system message and shape sampling options.

    A system or developer prompt that already defines the tool-call
    protocol stays unchanged. This keeps training and episode prompts
    identical.

    A request-supplied temperature always wins. The OMP_GYM_SAMPLE_TEMP
    env default applies only when the request omits the temperature;
    an explicit `sample_temp` argument forces that value on every
    request. RL sampling depends on rollout diversity; the served
    default of temperature 0 makes every episode identical.

    The `<|im_end|>` stop string is applied only for qwen model ids
    or when OMP_GYM_IM_END_STOP=1: other models never emit that
    marker, and an unused stop string costs nothing but confuses
    none, so scoping keeps the request faithful to the caller.

    Streaming passes through only when the request carries no tools;
    with tools the response must be buffered for call extraction.
    """
    upstream = dict(body)
    tools = upstream.pop("tools", None)
    tool_choice = upstream.get("tool_choice")
    if tool_choice == "none" or isinstance(tool_choice, dict):
        # A named function choice is enforced on the extracted calls;
        # "none" means the upstream request carries no choice at all.
        upstream.pop("tool_choice", None)
    if sample_temp is not None:
        upstream["temperature"] = sample_temp
    elif "temperature" not in upstream:
        sample_env = os.environ.get("OMP_GYM_SAMPLE_TEMP")
        if sample_env:
            try:
                upstream["temperature"] = float(sample_env)
            except ValueError:
                pass
    # Tuned qwen models sometimes emit the chat end marker as literal
    # text; the server then generates to the token cap. A stop string
    # ends the turn at the first marker either way.
    model_id = str(body.get("model") or "")
    want_im_end_stop = (
        "qwen" in model_id.lower() or os.environ.get("OMP_GYM_IM_END_STOP") == "1"
    )
    if want_im_end_stop:
        stops = upstream.get("stop")
        stop_list = [stops] if isinstance(stops, str) else list(stops or [])
        if "<|im_end|>" not in stop_list:
            stop_list.append("<|im_end|>")
        upstream["stop"] = stop_list
    if adapter_path is not None:
        upstream["adapters"] = adapter_path
    if isinstance(tools, list) and tools and tool_choice != "none":
        messages = [dict(m) for m in upstream.get("messages", [])]
        has_protocol = any(
            message.get("role") in {"system", "developer"}
            and (
                "<tool_call>" in str(message.get("content", ""))
                or "JSON object on its own line" in str(message.get("content", ""))
            )
            for message in messages
        )
        if not has_protocol:
            suffix = _tools_to_system_suffix(tools)
            if messages and messages[0].get("role") == "system":
                content = str(messages[0].get("content", ""))
                messages[0]["content"] = content + suffix
            else:
                messages.insert(0, {"role": "system", "content": suffix})
        upstream["messages"] = messages
    return upstream


def _shimmed_response(
    upstream_response: dict,
    available_tools: tuple[AvailableTool, ...],
    tool_choice: object = None,
    remap: bool = True,
) -> dict:
    """Convert upstream text output into valid tool calls.

    A named function choice forces the first generated call to that
    name when any call exists. A truncated generation keeps its
    completed calls and marks the response with `"partial": true`.
    """
    forced_name = _forced_tool_name(tool_choice, available_tools)
    response = dict(upstream_response)
    choices = []
    partial = False
    for choice in response.get("choices", []):
        message = dict(choice.get("message", {}))
        text = message.get("content") or ""
        content, calls, choice_partial = _extract_tool_calls(
            text, available_tools, remap
        )
        partial = partial or choice_partial
        if forced_name and calls:
            calls[0]["function"]["name"] = forced_name
        message["content"] = content if content else None
        if calls:
            message["tool_calls"] = calls
        rewritten = dict(choice)
        rewritten["message"] = message
        rewritten["finish_reason"] = (
            "tool_calls" if calls else choice.get("finish_reason")
        )
        choices.append(rewritten)
    response["choices"] = choices
    if partial:
        response["partial"] = True
    return response


def _sse_chunks(response: dict) -> list[dict]:
    """Render a finished response as chat.completion.chunk events."""
    chunks: list[dict] = []
    base = {
        "id": response.get("id", "shim"),
        "object": "chat.completion.chunk",
        "model": response.get("model", ""),
        "created": response.get("created", 0),
    }
    for choice in response.get("choices", []):
        message = choice.get("message", {})
        delta: dict = {"role": "assistant"}
        if message.get("content"):
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {**call, "index": position}
                for position, call in enumerate(message["tool_calls"])
            ]
        chunks.append(
            {
                **base,
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
        )
        chunks.append(
            {
                **base,
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": {},
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
            }
        )
    if "usage" in response:
        chunks.append({**base, "choices": [], "usage": response["usage"]})
    return chunks


def read_capture(entry: object) -> tuple[list, str] | None:
    """Read one current capture record as (messages, text)."""
    if not isinstance(entry, dict):
        return None
    messages = entry.get("messages")
    text = entry.get("text")
    if isinstance(messages, list) and isinstance(text, str):
        return messages, text
    return None


def make_handler(
    backend_port: int,
    adapter_path: str | None = None,
    *,
    capture: list[dict] | None = None,
    sample_temp: float | None = None,
    remap: bool = True,
) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to the backend port.

    With `capture`, the handler appends one record per completed
    request: the exact messages sent upstream and the raw completion
    text, before any tool-call rewriting, plus a uuid request id,
    finish_reason, usage, and model. RL training reads these records
    so the log-probability sees the text the policy sampled.

    `sample_temp` forces one temperature on every upstream request;
    None leaves the request temperature in charge, with the
    OMP_GYM_SAMPLE_TEMP env default filling in only requests that
    omit a temperature. `remap=False` disables the argument-shape
    rescue for unknown tool names.
    """

    class ShimHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"shim: {format % args}")

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            token = os.environ.get("OMP_GYM_SHIM_TOKEN")
            if not token:
                return True
            if self.headers.get("Authorization") == f"Bearer {token}":
                return True
            self._send_json(401, {"error": "unauthorized"})
            return False

        def _read_upstream_capped(self, upstream) -> bytes | None:
            """Read one upstream body with the 64 MiB cap applied."""
            raw = upstream.read(_MAX_UPSTREAM_BYTES + 1)
            if len(raw) > _MAX_UPSTREAM_BYTES:
                self._send_json(502, {"error": "upstream response too large"})
                return None
            return raw

        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self._send_json(404, {"error": "not found"})
                return
            if not _UPSTREAM_SLOTS.acquire(blocking=False):
                self._send_json(503, {"error": "upstream concurrency limit reached"})
                return
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{backend_port}/v1/models",
                    timeout=30,
                ) as upstream:
                    raw = self._read_upstream_capped(upstream)
                    if raw is None:
                        return
                    self._send_json(200, json.loads(raw))
            except (
                urllib.error.URLError,
                json.JSONDecodeError,
                OSError,
            ) as error:
                self._send_json(502, {"error": f"upstream request failed: {error}"})
            finally:
                _UPSTREAM_SLOTS.release()

        def _proxy_stream(self, request) -> None:
            """Proxy one upstream SSE stream to the client chunk by chunk.

            Tool-free streaming requests carry no tool-call JSON to
            extract, so the upstream chunks are forwarded verbatim;
            this keeps token-by-token latency for plain completions.
            """
            try:
                with urllib.request.urlopen(  # noqa: S310 - loopback backend only
                    request, timeout=600
                ) as upstream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    while True:
                        chunk = upstream.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (urllib.error.URLError, OSError) as error:
                self._send_json(502, {"error": f"upstream request failed: {error}"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(400, {"error": "missing content length"})
                return
            if length <= 0:
                self._send_json(400, {"error": "empty request body"})
                return
            if length > _MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "request body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "malformed JSON body"})
                return
            if not isinstance(body, dict):
                self._send_json(400, {"error": "body must be an object"})
                return
            tool_choice = body.get("tool_choice")
            available_tools = (
                () if tool_choice == "none" else _available_tools(body.get("tools"))
            )
            wants_stream = bool(body.get("stream", False))
            upstream_request = _rewrite_request(body, adapter_path, sample_temp)
            stream_upstream = wants_stream and not available_tools
            upstream_request["stream"] = stream_upstream
            upstream_body = json.dumps(upstream_request).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{backend_port}/v1/chat/completions",
                data=upstream_body,
                headers={"Content-Type": "application/json"},
            )
            if not _UPSTREAM_SLOTS.acquire(blocking=False):
                self._send_json(503, {"error": "upstream concurrency limit reached"})
                return
            try:
                if stream_upstream:
                    self._proxy_stream(request)
                    return
                try:
                    with urllib.request.urlopen(  # noqa: S310 - loopback backend only
                        request, timeout=600
                    ) as upstream:
                        raw = self._read_upstream_capped(upstream)
                        if raw is None:
                            return
                        upstream_response = json.loads(raw)
                        if (
                            not isinstance(upstream_response, dict)
                            or not isinstance(upstream_response.get("choices"), list)
                            or any(
                                not isinstance(choice, dict)
                                or not isinstance(choice.get("message"), dict)
                                for choice in upstream_response.get("choices", [])
                            )
                        ):
                            self._send_json(
                                502,
                                {"error": "upstream response has invalid shape"},
                            )
                            return
                except (
                    urllib.error.URLError,
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    OSError,
                ) as error:
                    self._send_json(
                        502,
                        {"error": f"upstream request failed: {error}"},
                    )
                    return
            finally:
                _UPSTREAM_SLOTS.release()
            if capture is not None:
                choices = upstream_response.get("choices") or [{}]
                message = choices[0].get("message") or {}
                capture.append(
                    {
                        "messages": upstream_request.get("messages", []),
                        "text": message.get("content") or "",
                        "request_id": str(uuid.uuid4()),
                        "finish_reason": choices[0].get("finish_reason"),
                        "usage": upstream_response.get("usage"),
                        "model": upstream_response.get("model"),
                    }
                )
            response = _shimmed_response(
                upstream_response, available_tools, tool_choice, remap
            )
            if not wants_stream:
                self._send_json(200, response)
                return
            # With tools offered, tool-call extraction needs the full
            # upstream text, so the response is buffered and emitted
            # as one complete SSE burst instead of a live proxy.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in _sse_chunks(response):
                data = json.dumps(chunk).encode()
                self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    return ShimHandler


def serve_shim(port: int, backend_port: int, adapter_path: str) -> None:
    """Block serving the shim until interrupted."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(backend_port, adapter_path),
    )
    print(f"shim: listening on 127.0.0.1:{port} -> {backend_port}")
    server.serve_forever()
