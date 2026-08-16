import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from omp_gym.export import SYSTEM_PROMPT
from omp_gym.shim import (
    _available_tools,
    _extract_tool_calls,
    _rewrite_request,
    make_handler,
)


def tool_schema(
    name: str,
    properties: tuple[str, ...],
    required: tuple[str, ...],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}.",
            "parameters": {
                "type": "object",
                "properties": {
                    property_name: {"type": "string"} for property_name in properties
                },
                "required": list(required),
            },
        },
    }


class ToolCallExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_tools = _available_tools(
            [
                tool_schema(
                    "bash",
                    ("command", "cwd", "timeout"),
                    ("command",),
                ),
                tool_schema("read", ("path", "offset"), ("path",)),
                tool_schema(
                    "write",
                    ("path", "content"),
                    ("path", "content"),
                ),
            ]
        )

    def extract(self, name: str, arguments: dict) -> tuple[str, list[dict]]:
        payload = json.dumps({"name": name, "arguments": arguments})
        return _extract_tool_calls(f"```json\n{payload}\n```", self.available_tools)

    def test_keeps_an_offered_tool_name(self) -> None:
        content, calls = self.extract("bash", {"command": "python3 test.py"})

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_accepts_bare_json_between_replacement_markers(self) -> None:
        payload = json.dumps(
            {
                "name": "write",
                "arguments": {"path": "app.py", "content": "answer = 42\n"},
            }
        )

        content, calls = _extract_tool_calls(
            f"\ufffd\n{payload}\n\ufffd",
            self.available_tools,
        )

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "write")

    def test_accepts_angle_tag_separated_json_calls(self) -> None:
        read_call = json.dumps(
            {"name": "read", "arguments": {"path": "slug.py"}}
        )
        bash_call = json.dumps(
            {"name": "bash", "arguments": {"command": "python3 test_slug.py"}}
        )

        content, calls = _extract_tool_calls(
            f"<lemma>\n{read_call}\n<lemma>\n<lemma>\n{bash_call}\n<lemma>",
            self.available_tools,
        )

        self.assertEqual(content, "")
        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["read", "bash"],
        )

    def test_keeps_calls_before_a_truncated_tail(self) -> None:
        read_call = json.dumps(
            {"name": "read", "arguments": {"path": "slug.py"}}
        )

        content, calls = _extract_tool_calls(
            f'<lemma>\n{read_call}\n<lemma>\n{{"name": "write", "argu',
            self.available_tools,
        )

        self.assertEqual(content, "")
        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["read"],
        )

    def test_accepts_raw_newlines_inside_fenced_json_strings(self) -> None:
        fenced = (
            '```json\n{\n  "name": "write",\n  "arguments": {\n'
            '    "path": "intervals.py",\n'
            '    "content": "def merge(a):\n    return a\n"\n  }\n}\n```'
        )

        content, calls = _extract_tool_calls(fenced, self.available_tools)

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "write")
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertIn("def merge(a):\n", arguments["content"])

    def test_keeps_prose_around_json_as_content(self) -> None:
        payload = json.dumps(
            {"name": "read", "arguments": {"path": "slug.py"}}
        )

        content, calls = _extract_tool_calls(
            f"I will read the file first.\n{payload}",
            self.available_tools,
        )

        self.assertEqual(calls, [])
        self.assertIn("read the file", content)

    def test_rejects_an_offered_name_with_wrong_arguments(self) -> None:
        content, calls = self.extract(
            "read",
            {"path": "app.py", "content": "print('ok')"},
        )

        self.assertEqual(calls, [])
        self.assertIn("app.py", content)

    def test_maps_an_unknown_name_by_command_shape(self) -> None:
        content, calls = self.extract(
            "run test.py",
            {"command": "python3 test.py", "cwd": "/tmp", "timeout": 30},
        )

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_maps_an_unknown_name_by_path_shape(self) -> None:
        content, calls = self.extract("inspect test.py", {"path": "test.py"})

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_keeps_an_incomplete_unknown_call_as_text(self) -> None:
        text = '```json\n{"name":"navigate","arguments":{"cwd":"/tmp"}}\n```'

        content, calls = _extract_tool_calls(text, self.available_tools)

        self.assertEqual(content, text)
        self.assertEqual(calls, [])

    def test_keeps_an_ambiguous_unknown_call_as_text(self) -> None:
        ambiguous_tools = _available_tools(
            [
                tool_schema("read", ("path",), ("path",)),
                tool_schema("inspect", ("path",), ("path",)),
            ]
        )
        text = '```json\n{"name":"open","arguments":{"path":"test.py"}}\n```'

        content, calls = _extract_tool_calls(text, ambiguous_tools)

        self.assertEqual(content, text)
        self.assertEqual(calls, [])

    def test_rejects_calls_when_no_tools_are_offered(self) -> None:
        text = '```json\n{"name":"bash","arguments":{"command":"true"}}\n```'

        content, calls = _extract_tool_calls(text, ())

        self.assertEqual(content, text)
        self.assertEqual(calls, [])

    def test_keeps_a_prose_shell_fence_as_text(self) -> None:
        text = "Try running: ```bash\npytest -q\n```"
        content, calls = _extract_tool_calls(text, self.available_tools)
        self.assertEqual(calls, [])
        self.assertIn("pytest -q", content)

    def test_keeps_a_bare_command_fence_as_text(self) -> None:
        text = "First check the suite.\n```\npython3 test_app.py\n```"
        content, calls = _extract_tool_calls(text, self.available_tools)
        self.assertEqual(calls, [])
        self.assertIn("python3 test_app.py", content)


class RequestRewriteTests(unittest.TestCase):
    def test_finds_an_explicit_protocol_after_context(self) -> None:
        body = {
            "messages": [
                {"role": "system", "content": "Runtime context."},
                {"role": "developer", "content": SYSTEM_PROMPT},
            ],
            "tools": [tool_schema("read", ("path",), ("path",))],
        }

        rewritten = _rewrite_request(body)

        self.assertEqual(rewritten["messages"], body["messages"])
        self.assertNotIn("tools", rewritten)

    def test_forwards_the_adapter_path_to_mlx(self) -> None:
        rewritten = _rewrite_request(
            {"messages": []},
            "adapters/tuned",
        )

        self.assertEqual(rewritten["adapters"], "adapters/tuned")

    def test_adds_tool_protocol_to_a_plain_system_prompt(self) -> None:
        body = {
            "messages": [{"role": "system", "content": "Code carefully."}],
            "tools": [tool_schema("read", ("path",), ("path",))],
        }

        rewritten = _rewrite_request(body)

        content = rewritten["messages"][0]["content"]
        self.assertTrue(content.startswith("Code carefully."))
        self.assertIn("JSON object on its own line", content)
        self.assertIn("- read:", content)


_RAW_COMPLETION = (
    '<tool_call>{"name": "bash", "arguments":'
    ' {"command": "true"}}</tool_call>'
)


class _FakeUpstream(BaseHTTPRequestHandler):
    """A local upstream that answers with a fixed raw completion."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "m",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _RAW_COMPLETION,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class CaptureTests(unittest.TestCase):
    def test_capture_appends_the_raw_completion(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        captured: list[dict] = []
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1], capture=captured),
        )
        for server in (upstream, shim):
            threading.Thread(
                target=server.serve_forever, daemon=True
            ).start()
        try:
            body = json.dumps(
                {
                    "model": "m",
                    "messages": [
                        {"role": "user", "content": "run the tests"}
                    ],
                    "tools": [
                        tool_schema("bash", ("command",), ("command",))
                    ],
                }
            ).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{shim.server_address[1]}"
                "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                shimmed = json.loads(response.read())
        finally:
            for server in (shim, upstream):
                server.shutdown()
                server.server_close()

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["text"], _RAW_COMPLETION)
        user_messages = [
            m for m in captured[0]["messages"] if m["role"] == "user"
        ]
        self.assertIn("run the tests", user_messages[0]["content"])
        message = shimmed["choices"][0]["message"]
        self.assertEqual(
            message["tool_calls"][0]["function"]["name"], "bash"
        )


if __name__ == "__main__":
    unittest.main()
