"""Import Claude Code and Codex sessions into the omp-gym session shape.

The output files match the omp session schema, so the exporter and
the clustering commands treat imported sessions like native ones.
Every output file opens with a session header that stamps the
source agent ("claude" or "codex"); a parsed trajectory reads it
back as Trajectory.source.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportStats:
    """What one import produced."""

    source: str
    files_seen: int
    files_written: int
    out_dir: str
    results_without_error_signal: int
    sessions_without_cwd: int


@dataclass(frozen=True)
class ConvertedSession:
    """One converted session plus the measurement facts about it."""

    entries: list[str]
    results_without_error_signal: int
    cwd: str | None


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


def _convert_claude(session_file: Path) -> ConvertedSession:
    """Convert one Claude Code session file.

    Claude records carry a top-level ``cwd`` field, which supplies the
    session header. Tool results define ``is_error`` with an absent key
    meaning success, so every result carries an explicit error signal.
    """
    tool_names: dict[str, str] = {}
    entries: list[str] = []
    cwd: str | None = None
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if cwd is None and isinstance(record.get("cwd"), str):
            cwd = record["cwd"]
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
                        "arguments": arguments if isinstance(arguments, dict) else {},
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
    return ConvertedSession(entries, 0, cwd)


_EXIT_CODE_LINE = re.compile(r"^Process exited with code (\d+)\s*$", re.MULTILINE)


def _codex_output_text(output: object) -> str:
    """Flatten one function_call_output payload into plain text."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(
            str(part.get("text", "")) for part in output if isinstance(part, dict)
        )
    if output is None:
        return ""
    return str(output)


def _parse_signal_dict(parsed: dict[str, object]) -> bool | None:
    """Read an error signal from one decoded output object."""
    exit_code = parsed.get("exit_code")
    if isinstance(exit_code, int):
        return exit_code != 0
    if "error" in parsed:
        return bool(parsed["error"])
    timed_out = parsed.get("timed_out")
    if isinstance(timed_out, bool):
        return timed_out
    return None


def _codex_error_signal(text: str) -> bool | None:
    """Extract the real error signal from one Codex tool output.

    Rollouts record failure in three observed shapes: a header line
    ``Process exited with code N``, a JSON object with ``exit_code``,
    ``error``, or ``timed_out`` fields, or that same JSON object after
    an ``Output:`` header line. Return None when no shape matches.
    """
    match = _EXIT_CODE_LINE.search(text)
    if match:
        return int(match.group(1)) != 0
    stripped = text.strip()
    if not stripped.startswith("{"):
        _, sep, tail = text.partition("\nOutput:\n")
        if not sep:
            return None
        stripped = tail.strip()
        if not stripped.startswith("{"):
            return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _parse_signal_dict(parsed)


def _convert_codex(session_file: Path) -> ConvertedSession:
    """Convert one Codex rollout file."""
    entries: list[str] = []
    cwd: str | None = None
    results_without_signal = 0
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        payload = record.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if record.get("type") in ("session_meta", "turn_context"):
            if cwd is None and isinstance(payload.get("cwd"), str):
                cwd = payload["cwd"]
            continue
        if record.get("type") != "response_item":
            continue
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
            text = _codex_output_text(payload.get("output"))
            signal = _codex_error_signal(text)
            if signal is None:
                results_without_signal += 1
            entries.append(
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": str(payload.get("call_id", "")),
                            "toolName": "tool",
                            "content": [{"type": "text", "text": text}],
                            "isError": bool(signal),
                        },
                    }
                )
            )
    return ConvertedSession(entries, results_without_signal, cwd)


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
        return ImportStats(source, 0, 0, str(out_dir), 0, 0)

    target_root = out_dir / source
    target_root.mkdir(parents=True, exist_ok=True)
    seen = 0
    written = 0
    results_without_signal = 0
    sessions_without_cwd = 0
    for session_file in sorted(root.rglob("*.jsonl")):
        seen += 1
        converted = convert(session_file)
        if not converted.entries:
            continue
        results_without_signal += converted.results_without_error_signal
        # Every imported session carries a header that names its
        # source agent, so a parsed trajectory can tell an imported
        # session from a native omp one. The cwd rides along when
        # the source format records it.
        header: dict[str, str] = {"type": "session", "source": source}
        if converted.cwd is None:
            sessions_without_cwd += 1
        else:
            header["cwd"] = converted.cwd
        lines = [json.dumps(header), *converted.entries]
        target = target_root / f"{session_file.stem}.jsonl"
        target.write_text("\n".join(lines) + "\n")
        written += 1
    return ImportStats(
        source,
        seen,
        written,
        str(target_root),
        results_without_signal,
        sessions_without_cwd,
    )
