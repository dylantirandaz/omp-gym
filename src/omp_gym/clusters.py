"""Count keyword-frequency signals in sessions and episodes.

Rule-based, rule-visible: each signal is a named regular
expression over the session trace. A user can open the pattern and
see exactly which keywords put a session in a bucket. The counts
are keyword hits, not verified failure diagnoses. Each count comes
with example artifact paths that open in the dashboard's
transcript view.
"""

import json
import re
import time
from pathlib import Path

from .trajectory import (
    AssistantStep,
    ToolResultStep,
    UserStep,
    parse_session,
)

# Anchored to the edit tool's real error strings, as recorded in
# session data under runs/. A generic "not found" or "anchor" in a
# tool result is not an edit failure.
_EDIT_FAIL = re.compile(
    r"payload line has no preceding hunk header"
    r"|input must begin with \"\[PATH#HASH\]\""
    r"|input header must be \[PATH\] or \[PATH#TAG\]"
    r"|`(?:PUT|CUT) [^`]+` rejected"
    r"|stale (?:tag|snapshot)",
    re.IGNORECASE,
)

# Second-person or imperative correction phrasing. A bare "wrong"
# or "stop" in a task statement is not a correction. mint.py gates
# its correction signal on this pattern; keep it importable.
_CORRECTION = re.compile(
    r"\bno,? that'?s (?:wrong|not right|not it)\b"
    r"|\bthat'?s (?:wrong|incorrect|not right)\b"
    r"|\byou(?:'re| are) wrong\b"
    r"|\byou broke\b"
    r"|\bstill (?:fails?|failed|failing|broken)\b"
    r"|\b(?:doesn'?t|does not|didn'?t|did not) work\b"
    r"|\bnot what i (?:asked|wanted|meant|said)\b"
    r"|\brevert (?:that|this|it|the)\b"
    r"|\bundo that\b"
    r"|\bstop (?:doing|changing|editing|touching|adding|rewriting)\b",
    re.IGNORECASE,
)

# Phrases of abandonment. An apology followed by a fix is not a
# giving-up signal.
_GAVE_UP = re.compile(
    r"\b(?:cannot|can'?t) (?:proceed|complete|finish|continue)\b"
    r"|\bunable to (?:proceed|complete|finish|continue)\b"
    r"|\bgiv(?:e|ing) up\b"
    r"|\bbeyond my (?:abilit|capabilit)",
    re.IGNORECASE,
)


def _classify_session(session_file: Path) -> dict[str, int]:
    """Count keyword-signal hits in one session."""
    counts = {
        "tool_error_results": 0,
        "edit_error_keywords": 0,
        "correction_keywords": 0,
        "abandonment_keywords": 0,
        "provider_error_lines": 0,
    }
    trajectory = parse_session(session_file)
    for step in trajectory.steps:
        if isinstance(step, ToolResultStep):
            if step.is_error:
                counts["tool_error_results"] += 1
                if _EDIT_FAIL.search(step.text):
                    counts["edit_error_keywords"] += 1
        elif isinstance(step, UserStep):
            if not step.text.startswith("<tool_response"):
                counts["correction_keywords"] += len(
                    _CORRECTION.findall(step.text)
                )
        elif isinstance(step, AssistantStep):
            if step.tool_calls == () and _GAVE_UP.search(step.text):
                counts["abandonment_keywords"] += 1
    for line in session_file.read_text().splitlines():
        if '"stopReason": "error"' in line or '"stopReason":"error"' in line:
            counts["provider_error_lines"] += 1
    return counts


def compute_clusters(
    sessions_root: Path,
    runs_dir: Path,
    out_dir: Path,
) -> dict:
    """Count keyword signals over all known sessions and episodes."""
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
