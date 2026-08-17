"""Local web dashboard over the experiment ledger.

Read-only: the dashboard shows what the platform did. Adapters and
their loss curves, the models leaderboard with cost per pass, the
episode browser with full transcripts, and the ledger timeline.
One stdlib HTTP server, one embedded page, no framework.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import DEFAULT_MODEL, default_model
from .ledger import read_ledger
from .page import DASHBOARD_PAGE
from .report import _model_stats
from .trajectory import AssistantStep, ToolResultStep, UserStep, parse_session

_LENS_CACHE: dict[str, object] = {}
_SAE_CACHE: dict[str, object] = {}
_GPU_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

_TRAIN_LINE = re.compile(
    r"Iter (\d+): Train loss ([\d.]+).*?"
    r"It/sec ([\d.]+), Tokens/sec ([\d.]+).*?"
    r"Peak mem ([\d.]+) GB"
)
_VAL_LINE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")


_MODEL_CACHE_LIMIT = 2


def _model_allowed(model_id: str) -> bool:
    """Only the configured model or a local model directory may load.

    The dashboard binds localhost, but any local page can fire GET
    requests at it. An arbitrary model id would trigger a network
    download of an attacker-chosen repository.
    """
    if model_id in (default_model(), DEFAULT_MODEL):
        return True
    candidate = Path(model_id)
    return candidate.is_dir() and candidate.resolve().is_relative_to(
        Path.cwd().resolve()
    )


def _adapter_allowed(adapter_dir: Path) -> bool:
    """Only adapters inside the workspace may load."""
    return (
        adapter_dir.resolve().is_relative_to(Path.cwd().resolve())
        and (adapter_dir / "adapters.safetensors").is_file()
    )


def _load_model(model_id: str, adapter_dir: Path | None):
    """Load one model once per process. Apply LoRA layers on load.

    The cache keeps the two newest entries; every entry holds a
    full model on a machine with limited unified memory.
    """
    from mlx_lm import load as mlx_load

    cache_key = f"{model_id}::{adapter_dir}"
    if cache_key not in _LENS_CACHE:
        if adapter_dir is None:
            _LENS_CACHE[cache_key] = mlx_load(model_id)
        else:
            weights = adapter_dir / "adapters.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(f"no adapter weights at {weights}")
            _LENS_CACHE[cache_key] = mlx_load(model_id, adapter_path=str(adapter_dir))
        while len(_LENS_CACHE) > _MODEL_CACHE_LIMIT:
            _LENS_CACHE.pop(next(iter(_LENS_CACHE)))
    return _LENS_CACHE[cache_key]


def _live_lens(
    prompt: str, model_id: str, adapter_dir: Path | None, top_k: int
) -> dict:
    """Compute a logit lens for an arbitrary prompt."""
    import mlx.core as mx

    from .inspect import _layer_predictions

    model, tokenizer = _load_model(model_id, adapter_dir)
    ids = mx.array(tokenizer.encode(prompt)[:256])[None]
    per_layer = _layer_predictions(model, ids, top_k)
    top_by_layer = []
    for top in per_layer:
        token_ids = top[0, -1, :].tolist()
        top_by_layer.append([tokenizer.decode([int(t)]) for t in reversed(token_ids)])
    return {
        "prompt": prompt,
        "model": model_id,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "layers": len(top_by_layer),
        "top_by_layer": top_by_layer,
    }


def _lens_diff(prompt: str, model_id: str, adapter_dir: Path, top_k: int) -> dict:
    """Show the base lens and the adapter lens side by side."""
    base = _live_lens(prompt, model_id, None, top_k)
    tuned = _live_lens(prompt, model_id, adapter_dir, top_k)
    diverges = [
        base_tokens[0] != tuned_tokens[0]
        for base_tokens, tuned_tokens in zip(
            base["top_by_layer"], tuned["top_by_layer"], strict=True
        )
    ]
    return {
        "prompt": prompt,
        "model": model_id,
        "adapter": str(adapter_dir),
        "base": base["top_by_layer"],
        "adapter_top": tuned["top_by_layer"],
        "diverges": diverges,
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


def _train_reports(runs_dir: Path) -> list[dict[str, object]]:
    """Collect train reports from adapters and fetched run trees."""
    report_paths = [
        *Path("adapters").glob("*/train_report.json"),
        *runs_dir.rglob("train_report.json"),
    ]
    reports: list[dict[str, object]] = []
    for report_path in report_paths:
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload["report_path"] = str(report_path)
        payload["modified"] = report_path.stat().st_mtime
        reports.append(payload)
    reports.sort(key=lambda report: report["modified"], reverse=True)
    return reports[:12]


def _live_training(log_path: Path) -> dict[str, object] | None:
    """Parse an mlx-lm training log that a job mirror appends to."""
    if not log_path.is_file():
        return None
    text = log_path.read_text()[-200_000:]
    iterations: list[int] = []
    losses: list[float] = []
    tokens_per_second: float | None = None
    peak_memory_gb: float | None = None
    for match in _TRAIN_LINE.finditer(text):
        iterations.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        tokens_per_second = float(match.group(4))
        peak_memory_gb = float(match.group(5))
    validations = [
        {"iteration": int(match.group(1)), "loss": float(match.group(2))}
        for match in _VAL_LINE.finditer(text)
    ]
    return {
        "path": str(log_path),
        "age_seconds": round(time.time() - log_path.stat().st_mtime, 1),
        "iterations": iterations,
        "losses": losses,
        "validations": validations,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_memory_gb,
    }


def _holdout_names() -> set[str]:
    """The names of the sealed holdout tasks."""
    root = Path("holdout-tasks")
    if not root.is_dir():
        return set()
    return {entry.name for entry in root.iterdir() if (entry / "task.toml").is_file()}


def _merge_cell(
    cells: dict[str, dict[str, int]],
    task: str,
    reward: float,
    error: object,
) -> None:
    """Fold one bench row into a task cell."""
    cell = cells.setdefault(task, {"passes": 0, "trials": 0, "errors": 0})
    if error:
        cell["errors"] += 1
        return
    cell["trials"] += 1
    if reward >= 1.0:
        cell["passes"] += 1


def _ledger_matrix_rows(ledger_path: Path) -> list[dict[str, object]]:
    """Bench rows from the ledger, labeled by the preceding serve."""
    entries, _ = read_ledger(ledger_path)
    serve_label: str | None = None
    rows: list[dict[str, object]] = []
    for entry in entries:
        if entry.kind == "serve":
            model_id = entry.config.get("model_id")
            serve_label = str(model_id) if model_id else None
            continue
        if entry.kind != "bench":
            continue
        grouped: dict[str, dict[str, dict[str, int]]] = {}
        for row in entry.metrics.get("rows", []):
            if not isinstance(row, dict):
                continue
            model = str(row.get("model", "?"))
            label = model
            if model.startswith("omp-gym/") and serve_label:
                label = serve_label
            _merge_cell(
                grouped.setdefault(label, {}),
                str(row.get("task", "?")),
                float(row.get("reward") or 0.0),
                row.get("error"),
            )
        for label, cells in grouped.items():
            rows.append({"label": label, "when": entry.timestamp, "cells": cells})
    return rows


def _file_matrix_rows() -> list[dict[str, object]]:
    """Bench rows from fetched or local bench row files."""
    candidates = list(Path(".").glob("bench-*.jsonl"))
    holdout_results = Path("holdout-results")
    if holdout_results.is_dir():
        candidates.extend(holdout_results.rglob("*report*.jsonl"))
    rows: list[dict[str, object]] = []
    for rows_path in candidates:
        try:
            lines = rows_path.read_text().splitlines()
        except OSError:
            continue
        cells_by_model: dict[str, dict[str, dict[str, int]]] = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                cells_by_model = {}
                break
            if not isinstance(row, dict) or "task" not in row:
                cells_by_model = {}
                break
            _merge_cell(
                cells_by_model.setdefault(str(row.get("model", "?")), {}),
                str(row["task"]),
                float(row.get("reward") or 0.0),
                row.get("error"),
            )
        if not cells_by_model:
            continue
        source = (
            rows_path.stem.replace("-report", "").replace("report", "").strip("-")
        ) or rows_path.parent.name
        when = time.strftime(
            "%Y-%m-%dT%H:%M", time.localtime(rows_path.stat().st_mtime)
        )
        for model, cells in cells_by_model.items():
            label = f"{model.split('/')[-1]} · {source}"
            rows.append({"label": label, "when": when, "cells": cells})
    return rows


def _bench_matrix(ledger_path: Path) -> dict[str, object]:
    """The run x task grid across ledger and fetched benches."""
    rows = _ledger_matrix_rows(ledger_path) + _file_matrix_rows()
    rows.sort(key=lambda row: str(row["when"]), reverse=True)
    holdout = _holdout_names()
    tasks: set[str] = set()
    for row in rows:
        cells = row["cells"]
        if not isinstance(cells, dict):
            raise TypeError("bench matrix cells must be a dictionary")
        tasks.update(cells.keys())
    train_tasks = sorted(tasks - holdout)
    holdout_tasks = sorted(tasks & holdout)
    return {
        "tasks": train_tasks + holdout_tasks,
        "holdout": holdout_tasks,
        "rows": rows[:40],
    }


def _latest_sae() -> dict | None:
    """The newest SAE artifact that still has its weights file."""
    artifact = _latest_artifact(Path("experiments"), "sae")
    if artifact is None:
        return None
    weights = artifact.get("weights")
    if not weights or not Path(str(weights)).is_file():
        return None
    return artifact


def _sae_weights(weights_path: str):
    """Load SAE weights once per process, keeping the two newest."""
    import mlx.core as mx

    if weights_path not in _SAE_CACHE:
        _SAE_CACHE[weights_path] = mx.load(weights_path)
        while len(_SAE_CACHE) > _MODEL_CACHE_LIMIT:
            _SAE_CACHE.pop(next(iter(_SAE_CACHE)))
    return _SAE_CACHE[weights_path]


def _sae_tokens(prompt: str) -> dict:
    """Per-token SAE feature activations for one prompt."""
    import mlx.core as mx

    from .sae import _captured_forward

    meta = _latest_sae()
    if meta is None:
        return {"error": "no SAE artifact; run omp-gym sae first"}
    adapter = meta.get("adapter")
    model, tokenizer = _load_model(
        str(meta["model"]), Path(str(adapter)) if adapter else None
    )
    ids = tokenizer.encode(prompt)[:128]
    if not ids:
        return {"error": "empty prompt"}
    weights = _sae_weights(str(meta["weights"]))
    hidden = _captured_forward(model, mx.array(ids)[None], int(meta["layer"]))
    z = mx.maximum(hidden @ weights["enc_w"].T + weights["enc_b"], 0)
    mx.eval(z)
    top_ids = mx.argsort(-z, axis=1)[:, :3].tolist()
    tokens = []
    for position, token_id in enumerate(ids):
        features = [
            {
                "id": feature_id,
                "activation": round(float(z[position, feature_id]), 3),
            }
            for feature_id in top_ids[position]
            if float(z[position, feature_id]) > 0
        ]
        tokens.append({"text": tokenizer.decode([token_id]), "features": features})
    peak = z.max(axis=0)
    mx.eval(peak)
    order = mx.argsort(-peak)[:8].tolist()
    top_features = [
        {"id": feature_id, "activation": round(float(peak[feature_id]), 3)}
        for feature_id in order
    ]
    return {
        "model": meta["model"],
        "adapter": adapter,
        "layer": meta["layer"],
        "tokens": tokens,
        "top_features": top_features,
    }


def _sae_steer(prompt: str, feature: int, alpha: float) -> dict:
    """Steered against unsteered short completions for one prompt."""
    import mlx.core as mx

    from .steer import _generate

    meta = _latest_sae()
    if meta is None:
        return {"error": "no SAE artifact; run omp-gym sae first"}
    weights = _sae_weights(str(meta["weights"]))
    feature_count = int(weights["dec_w"].shape[1])
    if not 0 <= feature < feature_count:
        return {"error": f"feature must be in 0..{feature_count - 1}"}
    adapter = meta.get("adapter")
    model, tokenizer = _load_model(
        str(meta["model"]), Path(str(adapter)) if adapter else None
    )
    direction = weights["dec_w"][:, feature]
    prompt_ids = mx.array(tokenizer.encode(prompt)[:128])
    unsteered = _generate(model, tokenizer, prompt_ids, direction, 0.0, 48)
    steered = _generate(model, tokenizer, prompt_ids, direction, alpha, 48)
    return {
        "feature": feature,
        "alpha": alpha,
        "unsteered": unsteered,
        "steered": steered,
    }


def _episode_index(runs_dir: Path) -> list[dict[str, object]]:
    """List all recorded episodes, newest first."""
    episodes = []
    for episode_file in sorted(runs_dir.glob("*/episode.json"), reverse=True):
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


def _episode_record(runs_dir: Path, episode: str) -> dict[str, object] | None:
    """The episode record plus the tail of its test output."""
    record_path = runs_dir / episode / "episode.json"
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    test_log = runs_dir / episode / "test_output.log"
    record["test_output"] = test_log.read_text()[-2000:] if test_log.is_file() else ""
    return record


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

        def _host_allowed(self) -> bool:
            """Reject DNS-rebound requests with a foreign Host header."""
            host = self.headers.get("Host", "").rsplit(":", 1)[0]
            return host in ("127.0.0.1", "localhost")

        def do_GET(self) -> None:
            if not self._host_allowed():
                self.send_response(403)
                self.end_headers()
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_html(DASHBOARD_PAGE)
            elif parsed.path == "/api/summary":
                entries, torn = read_ledger(ledger_path)
                train_entries = [entry for entry in entries if entry.kind == "train"]
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
                        "models": [asdict(stat) for stat in _model_stats(entries)],
                        "entries": len(entries),
                        "torn": torn,
                    }
                )
            elif parsed.path == "/api/episodes":
                self._send_json(_episode_index(runs_dir))
            elif parsed.path == "/api/transcript":
                episode = query.get("episode", [""])[0]
                if "/" in episode or ".." in episode:
                    self._send_json({"record": None, "steps": []})
                    return
                self._send_json(
                    {
                        "record": _episode_record(runs_dir, episode),
                        "steps": _transcript(runs_dir, episode),
                    }
                )
            elif parsed.path == "/api/monitor":
                self._send_json(
                    {
                        "live": _live_training(runs_dir / "live-train.log"),
                        "reports": _train_reports(runs_dir),
                    }
                )
            elif parsed.path == "/api/matrix":
                self._send_json(_bench_matrix(ledger_path))
            elif parsed.path == "/api/lens":
                prompt = query.get("prompt", [""])[0]
                if not prompt:
                    self._send_json({"error": "no prompt"})
                    return
                model_id = query.get("model", [None])[0] or default_model()
                adapter_raw = query.get("adapter", [None])[0]
                if not _model_allowed(model_id):
                    self._send_json({"error": "model not allowed; set it in gym.toml"})
                    return
                if adapter_raw and not _adapter_allowed(Path(adapter_raw)):
                    self._send_json({"error": "adapter must be a workspace directory"})
                    return
                try:
                    self._send_json(
                        _GPU_WORKER.submit(
                            _live_lens,
                            prompt,
                            model_id,
                            Path(adapter_raw) if adapter_raw else None,
                            3,
                        ).result()
                    )
                except Exception as error:
                    self._send_json({"error": str(error)})
            elif parsed.path == "/api/lensdiff":
                prompt = query.get("prompt", [""])[0]
                adapter_raw = query.get("adapter", [""])[0]
                if not prompt or not adapter_raw:
                    self._send_json({"error": "prompt and adapter are required"})
                    return
                model_id = query.get("model", [None])[0] or default_model()
                if not _model_allowed(model_id):
                    self._send_json({"error": "model not allowed; set it in gym.toml"})
                    return
                if not _adapter_allowed(Path(adapter_raw)):
                    self._send_json({"error": "adapter must be a workspace directory"})
                    return
                try:
                    self._send_json(
                        _GPU_WORKER.submit(
                            _lens_diff, prompt, model_id, Path(adapter_raw), 3
                        ).result()
                    )
                except Exception as error:
                    self._send_json({"error": str(error)})
            elif parsed.path == "/api/sae/tokens":
                prompt = query.get("prompt", [""])[0]
                if not prompt:
                    self._send_json({"error": "no prompt"})
                    return
                try:
                    self._send_json(_GPU_WORKER.submit(_sae_tokens, prompt).result())
                except Exception as error:
                    self._send_json({"error": str(error)})
            elif parsed.path == "/api/sae/steer":
                prompt = query.get("prompt", [""])[0]
                feature_raw = query.get("feature", [""])[0]
                alpha_raw = query.get("alpha", ["2"])[0]
                if not prompt or not feature_raw:
                    self._send_json({"error": "prompt and feature are required"})
                    return
                try:
                    feature = int(feature_raw)
                    alpha = float(alpha_raw)
                except ValueError:
                    self._send_json({"error": "feature and alpha must be numbers"})
                    return
                try:
                    self._send_json(
                        _GPU_WORKER.submit(_sae_steer, prompt, feature, alpha).result()
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
                rl_entries = [entry for entry in entries if entry.kind == "rl"]
                rl_series = [
                    {
                        "adapter": entry.config.get("adapter", "?"),
                        "mean_reward_first": entry.metrics.get("mean_reward_first"),
                        "mean_reward_last": entry.metrics.get("mean_reward_last"),
                        "rounds": entry.metrics.get("rounds", []),
                    }
                    for entry in rl_entries
                ]
                self._send_json({"train": series, "rl": rl_series})
            elif parsed.path == "/api/clusters":
                clusters_path = Path("experiments/clusters.json")
                if clusters_path.is_file():
                    self._send_json(json.loads(clusters_path.read_text()))
                else:
                    self._send_json({"clusters": {}})
            elif parsed.path == "/api/inspect":
                self._send_json(
                    {
                        "lens": _latest_artifact(Path("experiments"), "lens"),
                        "sae": _latest_artifact(Path("experiments"), "sae"),
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


def run_ui(port: int, ledger_path: Path, runs_dir: Path) -> None:
    """Serve the dashboard until interrupted."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(ledger_path, runs_dir)
    )
    print(f"omp-gym dashboard: http://127.0.0.1:{port}")
    print(f"ledger: {ledger_path}  runs: {runs_dir}")
    print("stop with Ctrl+C")
    server.serve_forever()
