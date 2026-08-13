"""Import Claude Code and Codex sessions into the omp-gym session shape.

The output files match the omp session schema, so the exporter and
the clustering commands treat imported sessions like native ones.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportStats:
    """What one import produced."""

    source: str
    files_seen: int
    files_written: int
    out_dir: str


def _text_entry(role: str, text: str) -> str:
    return json.dumps(
        {
            "type": "message",
            "message": {
                "role": role,
                "content": [{"type": "text", "text": text}],
            },
        }
    )


def _convert_claude(session_file: Path) -> list[str]:
    """Convert one Claude Code session file."""
    tool_names: dict[str, str] = {}
    entries: list[str] = []
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") not in ("user", "assistant"):
            continue
        message = record.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "tool_use":
                call_id = str(block.get("id", ""))
                tool_names[call_id] = str(block.get("name", "tool"))
                arguments = block.get("input")
                blocks.append(
                    {
                        "type": "toolCall",
                        "id": call_id,
                        "name": tool_names[call_id],
                        "arguments": arguments
                        if isinstance(arguments, dict)
                        else {},
                    }
                )
            elif block_type == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, list):
                    text = "\n".join(
                        str(part.get("text", ""))
                        for part in result_content
                        if isinstance(part, dict)
                    )
                else:
                    text = str(result_content or "")
                call_id = str(block.get("tool_use_id", ""))
                entries.append(
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "toolResult",
                                "toolCallId": call_id,
                                "toolName": tool_names.get(call_id, "tool"),
                                "content": [{"type": "text", "text": text}],
                                "isError": bool(block.get("is_error", False)),
                            },
                        }
                    )
                )
                continue
        if blocks:
            role = "assistant" if record["type"] == "assistant" else "user"
            entries.append(
                json.dumps(
                    {
                        "type": "message",
                        "message": {"role": role, "content": blocks},
                    }
                )
            )
    return entries


def _convert_codex(session_file: Path) -> list[str]:
    """Convert one Codex rollout file."""
    entries: list[str] = []
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload", {})
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role", "user")
            parts = [
                str(part.get("text", ""))
                for part in payload.get("content", [])
                if isinstance(part, dict)
            ]
            text = "\n".join(part for part in parts if part)
            if text:
                entries.append(_text_entry(role, text))
        elif kind == "function_call":
            try:
                arguments = json.loads(payload.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            entries.append(
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": str(payload.get("call_id", "")),
                                    "name": str(payload.get("name", "tool")),
                                    "arguments": arguments,
                                }
                            ],
                        },
                    }
                )
            )
        elif kind == "function_call_output":
            entries.append(
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": str(payload.get("call_id", "")),
                            "toolName": "tool",
                            "content": [
                                {
                                    "type": "text",
                                    "text": str(payload.get("output", "")),
                                }
                            ],
                            "isError": False,
                        },
                    }
                )
            )
    return entries


def import_sessions(source: str, out_dir: Path) -> ImportStats:
    """Import every session from one agent's store."""
    if source == "claude":
        root = Path.home() / ".claude" / "projects"
        convert = _convert_claude
    elif source == "codex":
        root = Path.home() / ".codex" / "sessions"
        convert = _convert_codex
    else:
        raise ValueError(f"unknown source: {source}")
    if not root.is_dir():
        return ImportStats(source, 0, 0, str(out_dir))

    target_root = out_dir / source
    target_root.mkdir(parents=True, exist_ok=True)
    seen = 0
    written = 0
    for session_file in sorted(root.rglob("*.jsonl")):
        seen += 1
        entries = convert(session_file)
        if not entries:
            continue
        target = target_root / f"{session_file.stem}.jsonl"
        target.write_text("\n".join(entries) + "\n")
        written += 1
    return ImportStats(source, seen, written, str(target_root))
