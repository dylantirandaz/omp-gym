import json
import unittest

from omp_gym.shim import _available_tools, _extract_tool_calls


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


if __name__ == "__main__":
    unittest.main()
