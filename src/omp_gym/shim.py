"""OpenAI-compatible shim that restores tool calls over mlx-lm.

The mlx-lm server (0.32) accepts a `tools` parameter but swallows
the generated tool call: the response carries neither content nor
`tool_calls`. This shim sits in front of the server and closes the
gap with the same convention the omp-gym adapters are trained on:

- incoming `tools` schemas become a compact list appended to the
  system message;
- the upstream request runs without `tools`, so the model answers
  in plain text;
- `<tool_call>{json}</tool_call>` blocks in the text are parsed
  into real OpenAI `tool_calls`;
- streaming requests are answered as a short SSE burst built from
  the finished upstream response.
"""

from dataclasses import dataclass
import json
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCED_CALL_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_SHELL_FENCE_PATTERN = re.compile(
    r"```(?:python|bash|sh)?\s*\n?((?:python3?\s+test_\S+|pytest\b[^\n]*"
    r"|npm\s+test[^\n]*|cargo\s+test[^\n]*)\s*\n?)```",
    re.DOTALL,
)

TOOL_PROTOCOL = (
    "\n\nTo call a tool, write a <tool_call> block that contains one "
    'JSON object: {"name": ..., "arguments": {...}}. '
    "Available tools:\n"
)


@dataclass(frozen=True)
class AvailableTool:
    """One offered tool and the keys in its argument object."""

    name: str
    argument_names: frozenset[str]
    required_argument_names: frozenset[str]


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
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        required = parameters.get("required") if isinstance(parameters, dict) else None
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
        )
    return tuple(tools_by_name.values())


def _resolve_tool_name(
    generated_name: str,
    arguments: dict,
    available_tools: tuple[AvailableTool, ...],
) -> str | None:
    """Resolve an exact name or one unique argument-shape match."""
    for tool in available_tools:
        if tool.name == generated_name:
            return generated_name
    argument_names = frozenset(arguments)
    if not argument_names:
        return None
    matches = [
        tool.name
        for tool in available_tools
        if tool.argument_names
        and tool.required_argument_names <= argument_names
        and argument_names <= tool.argument_names
    ]
    return matches[0] if len(matches) == 1 else None


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
    index: int,
    available_tools: tuple[AvailableTool, ...],
) -> dict | None:
    """Build one OpenAI tool call for an offered tool."""
    if not isinstance(payload, dict):
        return None
    generated_name = payload.get("name")
    arguments = payload.get("arguments")
    if (
        not isinstance(generated_name, str)
        or not generated_name
        or not isinstance(arguments, dict)
    ):
        return None
    name = _resolve_tool_name(generated_name, arguments, available_tools)
    if name is None:
        return None
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _extract_tool_calls(
    text: str,
    available_tools: tuple[AvailableTool, ...],
) -> tuple[str, list[dict]]:
    """Split generated text into plain content and valid tool calls.

    Three envelopes carry the same payload schema, depending on the
    prompt the model saw: <tool_call>{json}</tool_call> blocks (the
    trained form), fenced ```json blocks (markdown-heavy prompts),
    and a bare top-level JSON object (terse prompts).
    """
    for pattern in (_TOOL_CALL_PATTERN, _FENCED_CALL_PATTERN):
        found_calls = list(pattern.finditer(text))
        calls: list[dict] = []
        for index, found in enumerate(found_calls):
            try:
                payload = json.loads(found.group(1))
            except json.JSONDecodeError:
                calls = []
                break
            call = _call_from_payload(payload, index, available_tools)
            if call is None:
                calls = []
                break
            calls.append(call)
        if calls:
            return pattern.sub("", text).strip(), calls

    shell = _SHELL_FENCE_PATTERN.search(text)
    if shell:
        command = shell.group(1).strip()
        call = _call_from_payload(
            {"name": "bash", "arguments": {"command": command}},
            0,
            available_tools,
        )
        if call is not None:
            return _SHELL_FENCE_PATTERN.sub("", text).strip(), [call]

    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            payload = json.loads(trimmed)
        except json.JSONDecodeError:
            return trimmed, []
        call = _call_from_payload(payload, 0, available_tools)
        if call is not None:
            return "", [call]
    return trimmed, []


def _rewrite_request(body: dict) -> dict:
    """Move tools into the system message and force non-streaming.

    When OMP_GYM_SAMPLE_TEMP is set and the request carries no
    temperature, the shim injects it. RL sampling depends on
    rollout diversity; the server default of temperature 0 makes
    every episode identical.
    """
    upstream = dict(body)
    tools = upstream.pop("tools", None)
    upstream.pop("tool_choice", None)
    upstream["stream"] = False
    sample_temp = os.environ.get("OMP_GYM_SAMPLE_TEMP")
    if sample_temp and "temperature" not in upstream:
        upstream["temperature"] = float(sample_temp)
    if isinstance(tools, list) and tools:
        suffix = _tools_to_system_suffix(tools)
        messages = [dict(m) for m in upstream.get("messages", [])]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = str(messages[0].get("content", "")) + suffix
        else:
            messages.insert(0, {"role": "system", "content": suffix})
        upstream["messages"] = messages
    return upstream


def _shimmed_response(
    upstream_response: dict,
    available_tools: tuple[AvailableTool, ...],
) -> dict:
    """Convert upstream text output into valid tool calls."""
    response = dict(upstream_response)
    choices = []
    for choice in response.get("choices", []):
        message = dict(choice.get("message", {}))
        text = message.get("content") or ""
        content, calls = _extract_tool_calls(text, available_tools)
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


def make_handler(backend_port: int) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to the backend port."""

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

        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self._send_json(404, {"error": "not found"})
                return
            with urllib.request.urlopen(
                f"http://127.0.0.1:{backend_port}/v1/models", timeout=30
            ) as upstream:
                self._send_json(200, json.loads(upstream.read()))

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            available_tools = _available_tools(body.get("tools"))
            wants_stream = bool(body.get("stream", False))
            upstream_body = json.dumps(_rewrite_request(body)).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{backend_port}/v1/chat/completions",
                data=upstream_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as upstream:
                upstream_response = json.loads(upstream.read())
            response = _shimmed_response(upstream_response, available_tools)
            if not wants_stream:
                self._send_json(200, response)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in _sse_chunks(response):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    return ShimHandler


def serve_shim(port: int, backend_port: int) -> None:
    """Block serving the shim until interrupted."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(backend_port))
    print(f"shim: listening on 127.0.0.1:{port} -> {backend_port}")
    server.serve_forever()
