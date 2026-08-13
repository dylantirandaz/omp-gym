"""Cluster harvested sessions and episodes by failure mode.

Rule-based, rule-visible: each cluster is a named detector over
the session trace, so a user can see exactly why an episode sits
in a cluster. Counts come with example artifact paths that open
in the dashboard's transcript view.
"""

import json
import re
import time
from pathlib import Path

from .mint import _CORRECTION
from .trajectory import (
    AssistantStep,
    ToolResultStep,
    UserStep,
    parse_session,
)

_EDIT_FAIL = re.compile(r"not found|stale|anchor|old_string", re.IGNORECASE)
_GAVE_UP = re.compile(
    r"\b(cannot (complete|finish)|unable to|i'?m sorry|beyond my)\b",
    re.IGNORECASE,
)


def _classify_session(session_file: Path) -> dict[str, int]:
    """Count failure-mode hits in one session."""
    counts = {
        "tool_errors": 0,
        "edit_mismatches": 0,
        "user_corrections": 0,
        "gave_up": 0,
        "provider_errors": 0,
    }
    trajectory = parse_session(session_file)
    for step in trajectory.steps:
        if isinstance(step, ToolResultStep):
            if step.is_error:
                counts["tool_errors"] += 1
                if _EDIT_FAIL.search(step.text):
                    counts["edit_mismatches"] += 1
        elif isinstance(step, UserStep):
            if not step.text.startswith("<tool_response"):
                counts["user_corrections"] += len(
                    _CORRECTION.findall(step.text)
                )
        elif isinstance(step, AssistantStep):
            if step.tool_calls == () and _GAVE_UP.search(step.text):
                counts["gave_up"] += 1
    for line in session_file.read_text().splitlines():
        if '"stopReason": "error"' in line or '"stopReason":"error"' in line:
            counts["provider_errors"] += 1
    return counts


def compute_clusters(
    sessions_root: Path,
    runs_dir: Path,
    out_dir: Path,
) -> dict:
    """Cluster all known sessions and scored episodes."""
    clusters: dict[str, dict] = {}

    def add(mode: str, source: str, hits: int) -> None:
        entry = clusters.setdefault(mode, {"count": 0, "examples": []})
        entry["count"] += hits
        if len(entry["examples"]) < 5 and hits:
            entry["examples"].append(source)

    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        counts = _classify_session(session_file)
        for mode, hits in counts.items():
            add(mode, str(session_file), hits)

    for episode_file in sorted(runs_dir.glob("*/episode.json")):
        record = json.loads(episode_file.read_text())
        reward = float(record["reward"])
        if reward >= 1.0:
            continue
        episode = record["episode_dir"]
        add("failed_episode", episode, 1)
        session_file = Path(record["session_file"])
        if session_file.is_file():
            counts = _classify_session(session_file)
            for mode, hits in counts.items():
                add(mode, episode, hits)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "clusters.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clusters": dict(
            sorted(
                clusters.items(), key=lambda item: -item[1]["count"]
            )
        ),
    }
    artifact.write_text(json.dumps(payload, indent=2))
    return payload
