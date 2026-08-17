import contextlib
import http.client
import io
import json
import os
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from omp_gym.export import SYSTEM_PROMPT
from omp_gym.shim import (
    _available_tools,
    _extract_tool_calls,
    _rewrite_request,
    _shimmed_response,
    counts_reset,
    counts_snapshot,
    make_handler,
    read_capture,
)


def tool_schema(
    name: str,
    properties: tuple[str, ...],
    required: tuple[str, ...],
    types: dict[str, str] | None = None,
    extra: dict | None = None,
) -> dict:
    types = types or {}
    parameters = {
        "type": "object",
        "properties": {
            property_name: {"type": types.get(property_name, "string")}
            for property_name in properties
        },
        "required": list(required),
    }
    if extra:
        parameters.update(extra)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}.",
            "parameters": parameters,
        },
    }


class ToolCallExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        counts_reset()
        self.available_tools = _available_tools(
            [
                tool_schema(
                    "bash",
                    ("command", "cwd", "timeout"),
                    ("command",),
                    types={"timeout": "integer"},
                ),
                tool_schema(
                    "read",
                    ("path", "offset"),
                    ("path",),
                    types={"offset": "integer"},
                ),
                tool_schema(
                    "write",
                    ("path", "content"),
                    ("path", "content"),
                ),
            ]
        )

    def extract(self, name: str, arguments: dict) -> tuple[str, list[dict]]:
        payload = json.dumps({"name": name, "arguments": arguments})
        content, calls, _ = _extract_tool_calls(
            f"```json\n{payload}\n```", self.available_tools
        )
        return content, calls

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

        content, calls, _ = _extract_tool_calls(
            f"\ufffd\n{payload}\n\ufffd",
            self.available_tools,
        )

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "write")

    def test_accepts_angle_tag_separated_json_calls(self) -> None:
        read_call = json.dumps({"name": "read", "arguments": {"path": "slug.py"}})
        bash_call = json.dumps(
            {"name": "bash", "arguments": {"command": "python3 test_slug.py"}}
        )

        content, calls, partial = _extract_tool_calls(
            f"<lemma>\n{read_call}\n<lemma>\n<lemma>\n{bash_call}\n<lemma>",
            self.available_tools,
        )

        self.assertEqual(content, "")
        self.assertFalse(partial)
        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["read", "bash"],
        )

    def test_invalid_first_call_does_not_hide_a_valid_later_call(self) -> None:
        invalid = json.dumps({"name": "missing", "arguments": {"unknown": True}})
        valid = json.dumps({"name": "read", "arguments": {"path": "slug.py"}})

        content, calls, partial = _extract_tool_calls(
            f"{invalid}\n{valid}",
            self.available_tools,
            remap=False,
        )

        self.assertEqual(content, "")
        self.assertFalse(partial)
        self.assertEqual([call["function"]["name"] for call in calls], ["read"])
        self.assertEqual(counts_snapshot()["invalid"], 1)

    def test_keeps_calls_before_a_truncated_tail(self) -> None:
        read_call = json.dumps({"name": "read", "arguments": {"path": "slug.py"}})

        content, calls, partial = _extract_tool_calls(
            f'<lemma>\n{read_call}\n<lemma>\n{{"name": "write", "argu',
            self.available_tools,
        )

        self.assertEqual(content, "")
        self.assertTrue(partial)
        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["read"],
        )

    def test_rejects_raw_newlines_inside_fenced_json_strings(self) -> None:
        # Strict JSON parsing: raw control characters inside strings
        # are not valid JSON, so the call is not extracted.
        fenced = (
            '```json\n{\n  "name": "write",\n  "arguments": {\n'
            '    "path": "intervals.py",\n'
            '    "content": "def merge(a):\n    return a\n"\n  }\n}\n```'
        )

        content, calls, _ = _extract_tool_calls(fenced, self.available_tools)

        self.assertEqual(calls, [])
        self.assertIn("intervals.py", content)

    def test_keeps_prose_around_json_as_content(self) -> None:
        payload = json.dumps({"name": "read", "arguments": {"path": "slug.py"}})

        content, calls, _ = _extract_tool_calls(
            f"I will read the file first.\n{payload}",
            self.available_tools,
        )

        self.assertEqual(calls, [])
        self.assertIn("read the file", content)

    def test_rejects_an_offered_name_with_wrong_arguments(self) -> None:
        # The schema declares additionalProperties: false, so the
        # extra "content" key makes the call invalid.
        tools = _available_tools(
            [
                tool_schema(
                    "read",
                    ("path",),
                    ("path",),
                    extra={"additionalProperties": False},
                )
            ]
        )
        payload = json.dumps(
            {
                "name": "read",
                "arguments": {"path": "app.py", "content": "print('ok')"},
            }
        )

        content, calls, _ = _extract_tool_calls(f"```json\n{payload}\n```", tools)

        self.assertEqual(calls, [])
        self.assertIn("app.py", content)

    def test_maps_an_unknown_name_by_command_shape(self) -> None:
        content, calls = self.extract(
            "run test.py",
            {"command": "python3 test.py", "cwd": "/workdir", "timeout": 30},
        )

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_maps_an_unknown_name_by_path_shape(self) -> None:
        content, calls = self.extract("inspect test.py", {"path": "test.py"})

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_keeps_an_incomplete_unknown_call_as_text(self) -> None:
        text = '```json\n{"name":"navigate","arguments":{"cwd":"/tmp"}}\n```'

        content, calls, _ = _extract_tool_calls(text, self.available_tools)

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

        content, calls, _ = _extract_tool_calls(text, ambiguous_tools)

        self.assertEqual(content, text)
        self.assertEqual(calls, [])

    def test_rejects_calls_when_no_tools_are_offered(self) -> None:
        text = '```json\n{"name":"bash","arguments":{"command":"true"}}\n```'

        content, calls, _ = _extract_tool_calls(text, ())

        self.assertEqual(content, text)
        self.assertEqual(calls, [])

    def test_keeps_a_prose_shell_fence_as_text(self) -> None:
        text = "Try running: ```bash\npytest -q\n```"
        content, calls, _ = _extract_tool_calls(text, self.available_tools)
        self.assertEqual(calls, [])
        self.assertIn("pytest -q", content)

    def test_keeps_a_bare_command_fence_as_text(self) -> None:
        text = "First check the suite.\n```\npython3 test_app.py\n```"
        content, calls, _ = _extract_tool_calls(text, self.available_tools)
        self.assertEqual(calls, [])
        self.assertIn("python3 test_app.py", content)


class SchemaValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        counts_reset()

    def extract(self, tools: list[dict], name: str, arguments: dict):
        payload = json.dumps({"name": name, "arguments": arguments})
        return _extract_tool_calls(f"```json\n{payload}\n```", _available_tools(tools))

    def test_drops_a_wrong_value_type_and_counts_it(self) -> None:
        tools = [
            tool_schema(
                "bash",
                ("command", "timeout"),
                ("command",),
                types={"timeout": "integer"},
            )
        ]

        _, calls, _ = self.extract(
            tools, "bash", {"command": "true", "timeout": "soon"}
        )

        self.assertEqual(calls, [])
        self.assertEqual(counts_snapshot()["invalid"], 1)

    def test_drops_an_enum_violation_and_counts_it(self) -> None:
        tool = tool_schema("sort", ("mode",), ("mode",))
        tool["function"]["parameters"]["properties"]["mode"] = {
            "type": "string",
            "enum": ["asc", "desc"],
        }

        _, calls, _ = self.extract([tool], "sort", {"mode": "up"})

        self.assertEqual(calls, [])
        self.assertEqual(counts_snapshot()["invalid"], 1)

    def test_accepts_an_enum_member(self) -> None:
        tool = tool_schema("sort", ("mode",), ("mode",))
        tool["function"]["parameters"]["properties"]["mode"] = {
            "type": "string",
            "enum": ["asc", "desc"],
        }

        _, calls, _ = self.extract([tool], "sort", {"mode": "asc"})

        self.assertEqual(calls[0]["function"]["name"], "sort")

    def test_drops_extra_keys_under_additional_properties_false(self) -> None:
        tools = [
            tool_schema(
                "read",
                ("path",),
                ("path",),
                extra={"additionalProperties": False},
            )
        ]

        _, calls, _ = self.extract(tools, "read", {"path": "a.py", "sneaky": 1})

        self.assertEqual(calls, [])
        self.assertEqual(counts_snapshot()["invalid"], 1)

    def test_allows_extra_keys_without_additional_properties(self) -> None:
        tools = [tool_schema("read", ("path",), ("path",))]

        _, calls, _ = self.extract(tools, "read", {"path": "a.py", "sneaky": 1})

        self.assertEqual(calls[0]["function"]["name"], "read")


class RemapTests(unittest.TestCase):
    def setUp(self) -> None:
        counts_reset()
        self.available_tools = _available_tools(
            [tool_schema("bash", ("command",), ("command",))]
        )

    def test_remap_is_counted_and_logged(self) -> None:
        payload = json.dumps({"name": "run", "arguments": {"command": "make test"}})
        log = io.StringIO()

        with contextlib.redirect_stdout(log):
            _, calls, _ = _extract_tool_calls(
                f"```json\n{payload}\n```", self.available_tools
            )

        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertEqual(counts_snapshot()["remapped"], 1)
        self.assertIn('remapped call "run" -> "bash"', log.getvalue())

    def test_remap_knob_disables_the_rescue(self) -> None:
        payload = json.dumps({"name": "run", "arguments": {"command": "make test"}})

        content, calls, _ = _extract_tool_calls(
            f"```json\n{payload}\n```", self.available_tools, remap=False
        )

        self.assertEqual(calls, [])
        self.assertIn("make test", content)
        self.assertEqual(counts_snapshot()["remapped"], 0)


class StrictJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        counts_reset()
        self.available_tools = _available_tools(
            [tool_schema("read", ("path",), ("path",))]
        )

    def test_strict_json_rejects_control_characters(self) -> None:
        text = '```json\n{"name": "read", "arguments": {"path": "a\nb"}}\n```'

        content, calls, _ = _extract_tool_calls(text, self.available_tools)

        self.assertEqual(calls, [])
        self.assertIn('"path"', content)

    def test_partial_marker_reaches_the_response(self) -> None:
        read_call = json.dumps({"name": "read", "arguments": {"path": "slug.py"}})
        text = f'<lemma>\n{read_call}\n<lemma>\n{{"name": "read", "argu'
        upstream_response = {
            "id": "chatcmpl-1",
            "model": "m",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
        }

        response = _shimmed_response(upstream_response, self.available_tools)

        self.assertTrue(response["partial"])
        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")


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

    def test_tool_choice_none_strips_tools_upstream(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool_schema("read", ("path",), ("path",))],
            "tool_choice": "none",
        }

        rewritten = _rewrite_request(body)

        self.assertNotIn("tools", rewritten)
        self.assertNotIn("tool_choice", rewritten)
        self.assertNotIn(
            "JSON object on its own line",
            rewritten["messages"][0]["content"],
        )

    def test_tool_choice_required_passes_through(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool_schema("read", ("path",), ("path",))],
            "tool_choice": "required",
        }

        rewritten = _rewrite_request(body)

        self.assertEqual(rewritten["tool_choice"], "required")
        self.assertNotIn("tools", rewritten)


class TemperatureTests(unittest.TestCase):
    def test_request_temperature_wins_over_env_default(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"OMP_GYM_SAMPLE_TEMP": "0.9"}):
            rewritten = _rewrite_request({"messages": [], "temperature": 0.2})

        self.assertEqual(rewritten["temperature"], 0.2)

    def test_env_default_fills_in_a_missing_temperature(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"OMP_GYM_SAMPLE_TEMP": "0.9"}):
            rewritten = _rewrite_request({"messages": []})

        self.assertEqual(rewritten["temperature"], 0.9)

    def test_sample_temp_argument_forces_the_temperature(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"OMP_GYM_SAMPLE_TEMP": "0.3"}):
            rewritten = _rewrite_request(
                {"messages": [], "temperature": 0.2},
                sample_temp=0.7,
            )

        self.assertEqual(rewritten["temperature"], 0.7)

    def test_no_env_no_request_leaves_temperature_unset(self) -> None:
        env = dict(os.environ)
        env.pop("OMP_GYM_SAMPLE_TEMP", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            rewritten = _rewrite_request({"messages": []})

        self.assertNotIn("temperature", rewritten)


class ImEndStopTests(unittest.TestCase):
    def test_qwen_models_get_the_im_end_stop(self) -> None:
        rewritten = _rewrite_request({"model": "Qwen3-4B-Instruct", "messages": []})

        self.assertIn("<|im_end|>", rewritten["stop"])

    def test_non_qwen_models_get_no_stop(self) -> None:
        env = dict(os.environ)
        env.pop("OMP_GYM_IM_END_STOP", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            rewritten = _rewrite_request({"model": "llama-3.2", "messages": []})

        self.assertNotIn("stop", rewritten)

    def test_env_opt_in_adds_the_stop(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"OMP_GYM_IM_END_STOP": "1"}):
            rewritten = _rewrite_request({"model": "llama-3.2", "messages": []})

        self.assertIn("<|im_end|>", rewritten["stop"])


_RAW_COMPLETION = (
    '<tool_call>{"name": "bash", "arguments": {"command": "true"}}</tool_call>'
)


class _FakeUpstream(BaseHTTPRequestHandler):
    """A local upstream that answers with a fixed raw completion."""

    recorded_bodies: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            type(self).recorded_bodies.append(json.loads(body))
        except json.JSONDecodeError:
            pass
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

    def do_GET(self) -> None:
        payload = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _InvalidShapeUpstream(_FakeUpstream):
    """A local upstream that returns valid JSON with the wrong shape."""

    def do_POST(self) -> None:
        payload = b"[]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _stop_servers(*servers: ThreadingHTTPServer) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()


def _post(shim_port: int, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{shim_port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback test server
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _chat_body(**overrides) -> dict:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "run the tests"}],
        "tools": [tool_schema("bash", ("command",), ("command",))],
    }
    body.update(overrides)
    return body


class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        counts_reset()
        _FakeUpstream.recorded_bodies = []

    def test_capture_appends_the_raw_completion(self) -> None:
        captured: list[dict] = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1], capture=captured),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, shimmed = _post(shim.server_address[1], _chat_body())
        finally:
            _stop_servers(shim, upstream)

        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["text"], _RAW_COMPLETION)
        user_messages = [m for m in captured[0]["messages"] if m["role"] == "user"]
        self.assertIn("run the tests", user_messages[0]["content"])
        message = shimmed["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "bash")

    def test_capture_record_carries_full_metadata(self) -> None:
        captured: list[dict] = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1], capture=captured),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            _post(shim.server_address[1], _chat_body())
            _post(shim.server_address[1], _chat_body())
        finally:
            _stop_servers(shim, upstream)

        self.assertEqual(len(captured), 2)
        expected = {
            "messages",
            "text",
            "request_id",
            "finish_reason",
            "usage",
            "model",
        }
        self.assertEqual(set(captured[0]), expected)
        self.assertEqual(captured[0]["finish_reason"], "stop")
        self.assertEqual(captured[0]["model"], "m")
        self.assertIsInstance(captured[0]["request_id"], str)
        self.assertNotEqual(captured[0]["request_id"], captured[1]["request_id"])


class ToolChoiceHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        counts_reset()
        _FakeUpstream.recorded_bodies = []

    def _roundtrip(self, body: dict) -> tuple[dict, dict]:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1]),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, response = _post(shim.server_address[1], body)
        finally:
            _stop_servers(shim, upstream)
        self.assertEqual(status, 200)
        return response, _FakeUpstream.recorded_bodies[-1]

    def test_tool_choice_none_drops_generated_calls(self) -> None:
        response, upstream_body = self._roundtrip(_chat_body(tool_choice="none"))

        message = response["choices"][0]["message"]
        self.assertNotIn("tool_calls", message)
        self.assertIn("bash", message["content"])
        self.assertNotIn("tools", upstream_body)
        self.assertNotIn("tool_choice", upstream_body)

    def test_tool_choice_required_keeps_extraction(self) -> None:
        response, upstream_body = self._roundtrip(_chat_body(tool_choice="required"))

        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(upstream_body["tool_choice"], "required")
        self.assertNotIn("tools", upstream_body)

    def test_named_tool_choice_forces_the_first_call(self) -> None:
        body = _chat_body(
            tools=[
                tool_schema("bash", ("command",), ("command",)),
                tool_schema("read", ("path",), ("path",)),
            ],
            tool_choice={"type": "function", "function": {"name": "read"}},
        )

        response, _ = self._roundtrip(body)

        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")


class CallIdTests(unittest.TestCase):
    def test_call_ids_are_unique_uuids(self) -> None:
        tools = _available_tools(
            [
                tool_schema("read", ("path",), ("path",)),
                tool_schema("bash", ("command",), ("command",)),
            ]
        )
        read_call = json.dumps({"name": "read", "arguments": {"path": "slug.py"}})
        bash_call = json.dumps({"name": "bash", "arguments": {"command": "true"}})

        _, calls, _ = _extract_tool_calls(
            f"<lemma>\n{read_call}\n<lemma>\n{bash_call}", tools
        )

        ids = [call["id"] for call in calls]
        self.assertEqual(len(set(ids)), 2)
        for call_id in ids:
            self.assertTrue(call_id.startswith("call_"))
        self.assertNotIn("call_0", ids)
        self.assertNotIn("call_1", ids)


class ReadCaptureTests(unittest.TestCase):
    def test_reads_the_dict_shape(self) -> None:
        entry = {
            "messages": [{"role": "user", "content": "hi"}],
            "text": "raw",
            "request_id": "abc",
            "finish_reason": "stop",
            "usage": None,
            "model": "m",
        }

        self.assertEqual(read_capture(entry), (entry["messages"], "raw"))

    def test_rejects_the_removed_tuple_shape(self) -> None:
        entry = ([{"role": "user", "content": "hi"}], "raw")
        self.assertIsNone(read_capture(entry))


class AuthTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeUpstream.recorded_bodies = []

    def test_missing_token_gets_401(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"OMP_GYM_SHIM_TOKEN": "s3cret"}):
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
            shim = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(upstream.server_address[1]),
            )
            for server in (upstream, shim):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                status, body = _post(shim.server_address[1], _chat_body())
                ok_status, _ = _post(
                    shim.server_address[1],
                    _chat_body(),
                    headers={"Authorization": "Bearer s3cret"},
                )
            finally:
                _stop_servers(shim, upstream)

        self.assertEqual(status, 401)
        self.assertIn("error", body)
        self.assertEqual(ok_status, 200)

    def test_no_token_configured_allows_requests(self) -> None:
        env = dict(os.environ)
        env.pop("OMP_GYM_SHIM_TOKEN", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
            shim = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(upstream.server_address[1]),
            )
            for server in (upstream, shim):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                status, _ = _post(shim.server_address[1], _chat_body())
            finally:
                _stop_servers(shim, upstream)

        self.assertEqual(status, 200)


class _BlockingUpstream(_FakeUpstream):
    """An upstream that parks each request until released."""

    arrived = threading.Semaphore(0)
    release = threading.Event()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).arrived.release()
        type(self).release.wait(timeout=10)
        payload = json.dumps(
            {
                "id": "chatcmpl-1",
                "model": "m",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
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


class ConcurrencyTests(unittest.TestCase):
    def test_the_ninth_in_flight_request_gets_503(self) -> None:
        _BlockingUpstream.arrived = threading.Semaphore(0)
        _BlockingUpstream.release = threading.Event()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _BlockingUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1]),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        results: list[int] = []

        def fill_slot() -> None:
            status, _ = _post(shim.server_address[1], _chat_body())
            results.append(status)

        threads = [threading.Thread(target=fill_slot) for _ in range(8)]
        try:
            for thread in threads:
                thread.start()
            for _ in range(8):
                self.assertTrue(_BlockingUpstream.arrived.acquire(timeout=10))
            status, body = _post(shim.server_address[1], _chat_body())
        finally:
            _BlockingUpstream.release.set()
            for thread in threads:
                thread.join(timeout=10)
            _stop_servers(shim, upstream)

        self.assertEqual(status, 503)
        self.assertIn("error", body)
        self.assertEqual(results, [200] * 8)


class BodySizeTests(unittest.TestCase):
    def test_oversized_content_length_gets_413_without_a_body(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1]),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", shim.server_address[1], timeout=10
            )
            connection.putrequest("POST", "/v1/chat/completions")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(33 * 1024 * 1024))
            connection.endheaders()
            response = connection.getresponse()
            status = response.status
            body = json.loads(response.read())
            connection.close()
        finally:
            _stop_servers(shim, upstream)

        self.assertEqual(status, 413)
        self.assertIn("error", body)


class UpstreamShapeTests(unittest.TestCase):
    def test_non_object_response_gets_502(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _InvalidShapeUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1]),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, body = _post(shim.server_address[1], _chat_body())
        finally:
            _stop_servers(shim, upstream)

        self.assertEqual(status, 502)
        self.assertIn("invalid shape", body["error"])


class ModelsErrorTests(unittest.TestCase):
    def test_upstream_failure_gets_502(self) -> None:
        dead = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        dead_port = dead.server_address[1]
        dead.server_close()
        shim = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dead_port))
        threading.Thread(target=shim.serve_forever, daemon=True).start()
        try:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{shim.server_address[1]}/v1/models",
                    timeout=10,
                ) as response:
                    status = response.status
                    body = json.loads(response.read())
            except urllib.error.HTTPError as error:
                status = error.code
                body = json.loads(error.read())
        finally:
            _stop_servers(shim)

        self.assertEqual(status, 502)
        self.assertIn("error", body)


class StreamingProxyTests(unittest.TestCase):
    def test_tool_free_stream_proxies_upstream_sse(self) -> None:
        class SseUpstream(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                type(self).seen_stream = body.get("stream")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for token in ("hel", "lo"):
                    self.wfile.write(
                        b"data: " + json.dumps({"t": token}).encode() + b"\n\n"
                    )
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        SseUpstream.seen_stream = None
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), SseUpstream)
        shim = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(upstream.server_address[1]),
        )
        for server in (upstream, shim):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            data = json.dumps(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            ).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{shim.server_address[1]}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback test server
                content_type = response.headers["Content-Type"]
                raw = response.read()
        finally:
            _stop_servers(shim, upstream)

        self.assertEqual(content_type, "text/event-stream")
        self.assertTrue(SseUpstream.seen_stream)
        self.assertEqual(raw.count(b"data: "), 3)
        self.assertIn(b"[DONE]", raw)


if __name__ == "__main__":
    unittest.main()
