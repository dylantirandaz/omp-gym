"""Local web dashboard over the experiment ledger.

Read-only: the dashboard shows what the platform did. Adapters and
their loss curves, the models leaderboard with cost per pass, the
episode browser with full transcripts, and the ledger timeline.
One stdlib HTTP server, one embedded page, no framework.
"""

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .ledger import DEFAULT_LEDGER, read_ledger
from .report import _model_stats
from .trajectory import AssistantStep, ToolResultStep, UserStep, parse_session

from .page import DASHBOARD_PAGE


def _episode_index(runs_dir: Path) -> list[dict[str, object]]:
    """List all recorded episodes, newest first."""
    episodes = []
    for episode_file in sorted(
        runs_dir.glob("*/episode.json"), reverse=True
    ):
        record = json.loads(episode_file.read_text())
        episodes.append(
            {
                "episode": Path(record["episode_dir"]).name,
                "task": record["task"],
                "model": record["model"],
                "reward": record["reward"],
                "duration_seconds": record["duration_seconds"],
            }
        )
    return episodes


def _transcript(runs_dir: Path, episode: str) -> list[dict[str, object]]:
    """Render one episode's session as display steps."""
    episode_dir = runs_dir / episode
    record_path = episode_dir / "episode.json"
    if not record_path.is_file():
        return []
    record = json.loads(record_path.read_text())
    trajectory = parse_session(Path(record["session_file"]))
    steps: list[dict[str, object]] = []
    for step in trajectory.steps:
        match step:
            case UserStep():
                steps.append({"role": "user", "text": step.text})
            case AssistantStep():
                steps.append(
                    {
                        "role": "assistant",
                        "text": step.text,
                        "tool_calls": [
                            {"name": call.name, "arguments": call.arguments}
                            for call in step.tool_calls
                        ],
                    }
                )
            case ToolResultStep():
                steps.append(
                    {
                        "role": "tool",
                        "text": step.text[:2000],
                        "tool_name": step.tool_name,
                        "is_error": step.is_error,
                    }
                )
    return steps


def make_handler(ledger_path: Path, runs_dir: Path):
    """Build the request handler bound to ledger and runs paths."""

    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send_json(self, payload: object) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_html(DASHBOARD_PAGE)
            elif parsed.path == "/api/summary":
                entries, torn = read_ledger(ledger_path)
                train_entries = [
                    entry for entry in entries if entry.kind == "train"
                ]
                self._send_json(
                    {
                        "adapters": [
                            {
                                "adapter": entry.config.get("adapter"),
                                "model": entry.config.get("model"),
                                "method": entry.config.get("method"),
                                "metrics": entry.metrics,
                                "timestamp": entry.timestamp,
                            }
                            for entry in train_entries
                        ],
                        "models": [
                            asdict(stat)
                            for stat in _model_stats(entries)
                        ],
                        "entries": len(entries),
                        "torn": torn,
                    }
                )
            elif parsed.path == "/api/episodes":
                self._send_json(_episode_index(runs_dir))
            elif parsed.path == "/api/transcript":
                episode = query.get("episode", [""])[0]
                if "/" in episode or ".." in episode:
                    self._send_json([])
                    return
                self._send_json(_transcript(runs_dir, episode))
            elif parsed.path == "/api/timeline":
                entries, _ = read_ledger(ledger_path)
                self._send_json(
                    [
                        {
                            "kind": entry.kind,
                            "timestamp": entry.timestamp,
                            "config": entry.config,
                        }
                        for entry in entries[-25:]
                    ]
                )
            else:
                self.send_response(404)
                self.end_headers()

    return DashboardHandler


def run_ui(
    port: int, ledger_path: Path, runs_dir: Path
) -> None:
    """Serve the dashboard until interrupted."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(ledger_path, runs_dir)
    )
    print(f"omp-gym dashboard: http://127.0.0.1:{port}")
    print(f"ledger: {ledger_path}  runs: {runs_dir}")
    print("stop with Ctrl+C")
    server.serve_forever()
