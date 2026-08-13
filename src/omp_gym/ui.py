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

from .ledger import read_ledger
from .report import _model_stats
from .trajectory import AssistantStep, ToolResultStep, UserStep, parse_session

from .page import DASHBOARD_PAGE

_LENS_CACHE: dict[str, object] = {}


def _live_lens(prompt: str, model_id: str, adapter_dir: Path | None, top_k: int) -> dict:
    """Compute a logit lens for an arbitrary prompt.

    The model and tokenizer load once per server process and stay
    cached; adapter weights apply on first load too.
    """
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    from .inspect import _layer_predictions

    cache_key = f"{model_id}::{adapter_dir}"
    if cache_key not in _LENS_CACHE:
        model, tokenizer = mlx_load(model_id)
        if adapter_dir is not None:
            weights = adapter_dir / "adapters.safetensors"
            if weights.is_file():
                model.load_weights(str(weights), strict=False)
        _LENS_CACHE[cache_key] = (model, tokenizer)
    model, tokenizer = _LENS_CACHE[cache_key]

    ids = mx.array(tokenizer.encode(prompt))[None]
    per_layer = _layer_predictions(model, ids, top_k)
    top_by_layer = []
    for top in per_layer:
        token_ids = top[0, -1, :].tolist()
        top_by_layer.append(
            [tokenizer.decode([int(t)]) for t in reversed(token_ids)]
        )
    return {
        "prompt": prompt,
        "model": model_id,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "layers": len(top_by_layer),
        "top_by_layer": top_by_layer,
    }


def _latest_artifact(experiments_dir: Path, prefix: str) -> dict | None:
    """Read the newest experiment artifact with a given prefix."""
    candidates = sorted(experiments_dir.glob(f"{prefix}-*.json"))
    if not candidates:
        single = experiments_dir / f"{prefix}.json"
        if single.is_file():
            return json.loads(single.read_text())
        return None
    return json.loads(candidates[-1].read_text())


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
            elif parsed.path == "/api/lens":
                prompt = query.get("prompt", [""])[0]
                if not prompt:
                    self._send_json({"error": "no prompt"})
                    return
                model_id = query.get("model", [None])[0]
                adapter_raw = query.get("adapter", [None])[0]
                try:
                    self._send_json(
                        _live_lens(
                            prompt,
                            model_id or "mlx-community/Qwen2.5-3B-Instruct-4bit",
                            Path(adapter_raw) if adapter_raw else None,
                            3,
                        )
                    )
                except Exception as error:
                    self._send_json({"error": str(error)})
            elif parsed.path == "/api/training":
                entries, _ = read_ledger(ledger_path)
                series = []
                for entry in entries:
                    if entry.kind != "train":
                        continue
                    metrics = entry.metrics
                    first = metrics.get("first_train_loss")
                    last = metrics.get("last_train_loss")
                    if first is None or last is None:
                        continue
                    series.append(
                        {
                            "adapter": entry.config.get("adapter", "?"),
                            "method": entry.config.get("method", "sft"),
                            "first": first,
                            "last": last,
                            "first_val": metrics.get("first_val_loss"),
                            "last_val": metrics.get("last_val_loss"),
                            "iters": metrics.get("iterations"),
                            "series": metrics.get("train_series"),
                        }
                    )
                rl_entries = [
                    entry for entry in entries if entry.kind == "rl"
                ]
                rl_series = [
                    {
                        "adapter": entry.config.get("adapter", "?"),
                        "mean_reward_first": entry.metrics.get(
                            "mean_reward_first"
                        ),
                        "mean_reward_last": entry.metrics.get(
                            "mean_reward_last"
                        ),
                        "rounds": entry.metrics.get("rounds", []),
                    }
                    for entry in rl_entries
                ]
                self._send_json({"train": series, "rl": rl_series})
            elif parsed.path == "/api/clusters":
                clusters_path = Path("experiments/clusters.json")
                if clusters_path.is_file():
                    self._send_json(
                        json.loads(clusters_path.read_text())
                    )
                else:
                    self._send_json({"clusters": {}})
            elif parsed.path == "/api/inspect":
                self._send_json(
                    {
                        "lens": _latest_artifact(
                            Path("experiments"), "lens"
                        ),
                        "sae": _latest_artifact(
                            Path("experiments"), "sae"
                        ),
                    }
                )
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
