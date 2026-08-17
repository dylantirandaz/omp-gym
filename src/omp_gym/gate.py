"""Memorization gate for LoRA adapters.

The gate compares one adapter against its base model. It runs two
checks. The first check captures residual activations for a fixed
probe set and maps them through an SAE. Feature profiles that move
far from the base profile give a high drift score. The second check
generates short completions with the adapter and counts training
task tokens that the prompt did not ask for. The two scores make one
memorization score. A score at or above the threshold flags the
adapter and the CLI exits with code 1.

The gate is a first-pass detector. It rests on hand-picked leak
markers and one threshold tuned on a small set of measured runs.
It is not a general memorization test: an adapter that leaks
training data without these exact markers passes the leak check,
and drift only sees the one probed layer.

The gate also reads the adapter's own train_report.json when one
exists and replays the training-time memorization-shaped curve
check (memorization_shaped_run) on it. The curve verdict is
recorded in the artifact next to the thresholds used; the flag
itself stays a drift-plus-leakage decision.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from .sae import SAE_LAYER, _captured_forward, train_sae
except ModuleNotFoundError:
    # Off-Mac machines have no mlx. The pure helpers and the
    # constants stay importable; run_gate fails at preflight.
    mx = generate = load = make_sampler = None
    SAE_LAYER = _captured_forward = train_sae = None

from .export import SYSTEM_PROMPT, TASK_PROMPT_PREFIX
from .preflight import require_metal_gpu
from .task import TaskLoadError, load_task

TOP_K = 32
PROBE_TOKEN_CAP = 512
MAX_GEN_TOKENS = 128
LEAK_CAP = 3.0
DRIFT_WEIGHT = 0.4
LEAK_WEIGHT = 0.6
# Threshold rationale, from measured runs on the Apple M3:
# - Control (base vs base) measures exactly 0.0.
# - Weighted drift of the subtle rank-8 LoRA v15 measures 0.0097.
#   Drift alone reaches 0.10 only when the layer-12 profiles move
#   far on most probes.
# - One probe completion with two pasted training-code hits adds
#   0.6 * (2/3) / 6 = 0.0667. One stray single-hit probe adds only
#   0.0333 and does not flag.
# The margin of 0.10 needs repeated verbatim leakage or extreme
# drift. Adapter v15 measured above it; the control measured 0.0.
MEMORIZATION_THRESHOLD = 0.10

# Markers are training-task tokens. The first group holds task
# names. The second group holds verbatim code from the slug task
# solutions in the training data. Adapter v15 pasted that code
# into unrelated tasks without the task name, so name markers
# alone do not catch it. Bare common words such as "csv" or
# "slug" are excluded: they appear in ordinary completions and
# would count false leaks.
LEAK_MARKERS = (
    "slugify",
    "fizzbuzz",
    "FizzBuzz",
    "total_column",
    "[^a-z0-9]",
    ".strip('-')",
)

NEUTRAL_PROMPTS = (
    (
        "neutral-reverse",
        "Write a Python function that reverses a string. Add a short docstring.",
    ),
    (
        "neutral-search",
        "Explain how binary search works on a sorted list. Keep the answer short.",
    ),
    (
        "neutral-fib",
        "Write a Python function that returns the n-th Fibonacci "
        "number. Use iteration, not recursion.",
    ),
)


class GateError(SystemExit):
    """Raised when the gate cannot run."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"gate failed: {reason}")


def memorization_shaped_run(
    losses: list[float],
    val_losses: list[float],
    *,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
) -> str | None:
    """A reason when the loss curves show memorization, else None.

    A memorization-shaped run drops the train loss by more than
    train_drop_pct while the val loss drops by less than
    val_drop_ratio of the train drop. Such a run learned the train
    set, not the task. This is the single home of the check:
    train.py gates the training run on it, run_gate replays it on
    the adapter's train report.
    """
    if not losses or not val_losses:
        return None
    if losses[0] <= 0 or val_losses[0] <= 0:
        return None
    train_drop = (losses[0] - losses[-1]) / losses[0]
    val_drop = (val_losses[0] - val_losses[-1]) / val_losses[0]
    if train_drop > train_drop_pct and val_drop < train_drop * val_drop_ratio:
        return (
            "memorization-shaped run: train loss dropped "
            f"{train_drop:.1%} but val loss dropped only "
            f"{val_drop:.1%}; the adapter learned the train "
            "set, not the task"
        )
    return None


@dataclass(frozen=True)
class Probe:
    """One named prompt for activation capture."""

    name: str
    source: str
    prompt: str
    runner: str


def _task_probes(root: Path, source: str) -> list[Probe]:
    """Load the prompt of every task directory under one root."""
    probes: list[Probe] = []
    for config_path in sorted(root.glob("*/task.toml")):
        task = load_task(config_path.parent)
        if isinstance(task, TaskLoadError):
            raise GateError(f"bad task {task.path}: {task.reason}")
        probes.append(Probe(task.name, source, task.prompt, task.test_command[0]))
    if not probes:
        raise GateError(f"no tasks found under {root}")
    return probes


def _collect_probes(holdout_dir: Path, tasks_dir: Path) -> list[Probe]:
    """Build the full probe set: holdout, training, neutral."""
    probes = _task_probes(holdout_dir, "holdout")
    probes.extend(_task_probes(tasks_dir, "training"))
    probes.extend(
        Probe(name, "neutral", prompt, "python3") for name, prompt in NEUTRAL_PROMPTS
    )
    return probes


def _find_compatible_sae(out_dir: Path, model_id: str) -> Path | None:
    """Find the newest SAE weights trained for this base model."""
    for artifact in sorted(out_dir.glob("sae-*.json"), reverse=True):
        try:
            meta = json.loads(artifact.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(meta, dict):
            continue
        weights = meta.get("weights")
        if meta.get("model") != model_id or not isinstance(weights, str):
            continue
        if meta.get("adapter"):
            continue
        weights_path = Path(weights)
        if weights_path.is_file():
            return weights_path
    return None


def _ensure_sae(
    out_dir: Path,
    model_id: str,
    sae_weights: Path | None,
    data_dir: Path,
) -> tuple[Path, str]:
    """Return usable SAE weights and a provenance note."""
    if sae_weights is not None:
        if not sae_weights.is_file():
            raise GateError(f"no SAE weights at {sae_weights}")
        return sae_weights, f"given on the command line: {sae_weights}"
    found = _find_compatible_sae(out_dir, model_id)
    if found is not None:
        return found, f"reused, trained earlier for {model_id}: {found}"
    if not (data_dir / "train.jsonl").is_file():
        raise GateError(f"no compatible SAE and no {data_dir}/train.jsonl to train one")
    print(f"no compatible SAE for {model_id}; training a fresh one")
    metrics = train_sae(data_dir, model_id, None, out_dir)
    gc.collect()
    mx.clear_cache()
    return (
        Path(metrics["weights"]),
        f"trained fresh for {model_id} in this run: {metrics['weights']}",
    )


def _train_curve_verdict(
    adapter_dir: Path | None,
    *,
    train_drop_pct: float,
    val_drop_ratio: float,
) -> str | None:
    """Replay the curve check on the adapter's own train report.

    An adapter directory written by a completed training run holds
    train_report.json with the first and last train/val losses.
    When the file is absent or unreadable the verdict is None: no
    evidence, not a pass.
    """
    if adapter_dir is None:
        return None
    report_path = adapter_dir / "train_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    first_train = report.get("first_train_loss")
    last_train = report.get("last_train_loss")
    if not isinstance(first_train, (int, float)) or not isinstance(
        last_train, (int, float)
    ):
        return None
    first_val = report.get("first_val_loss")
    last_val = report.get("last_val_loss")
    val_losses = (
        [float(first_val), float(last_val)]
        if isinstance(first_val, (int, float)) and isinstance(last_val, (int, float))
        else []
    )
    return memorization_shaped_run(
        [float(first_train), float(last_train)],
        val_losses,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )


def _feature_profile(model, enc_w, enc_b, ids: list[int]) -> mx.array:
    """Mean SAE feature activation over the tokens of one prompt."""
    hidden = _captured_forward(model, mx.array(ids)[None], SAE_LAYER)
    z = mx.maximum(hidden[1:] @ enc_w.T + enc_b, 0)
    profile = z.mean(axis=0)
    mx.eval(profile)
    return profile


def _capture_profiles(
    model, tokenizer, enc_w, enc_b, probes: list[Probe]
) -> list[mx.array]:
    """Compute one feature profile per probe prompt."""
    profiles = []
    for probe in probes:
        ids = tokenizer.encode(probe.prompt)[:PROBE_TOKEN_CAP]
        if len(ids) < 8:
            raise GateError(f"probe {probe.name} is too short")
        profiles.append(_feature_profile(model, enc_w, enc_b, ids))
    return profiles


def _probe_metrics(base: mx.array, other: mx.array) -> tuple[float, float]:
    """Top-k overlap and cosine shift between two profiles."""
    base_top = set(mx.argsort(-base)[:TOP_K].tolist())
    other_top = set(mx.argsort(-other)[:TOP_K].tolist())
    overlap = len(base_top & other_top) / TOP_K
    norm = float(mx.linalg.norm(base)) * float(mx.linalg.norm(other))
    if norm == 0.0:
        raise GateError("a probe produced an all-zero feature profile")
    cosine = float((base * other).sum()) / norm
    shift = min(1.0, max(0.0, 1.0 - cosine))
    return overlap, shift


def _count_leaks(prompt: str, text: str) -> dict[str, int]:
    """Count marker tokens in the text that the prompt did not use."""
    lowered_prompt = prompt.lower()
    counts: dict[str, int] = {}
    for marker in LEAK_MARKERS:
        if marker.lower() in lowered_prompt:
            continue
        count = text.count(marker)
        if marker == "slug":
            count -= text.count("slugify")
        if count > 0:
            counts[marker] = count
    return counts


def _generate_text(model, tokenizer, prompt: str) -> str:
    """Greedy short completion in the training chat format.

    The known leaks appear in episodes that run with the agent
    system prompt and the task prefix. The probe uses the same
    format so that it measures the deployed regime.
    """
    user_turn = f"{TASK_PROMPT_PREFIX}\n\n{prompt}"
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    return generate(
        model,
        tokenizer,
        prompt=chat,
        max_tokens=MAX_GEN_TOKENS,
        sampler=make_sampler(temp=0.0),
    )


def _leakage_records(model, tokenizer, probes: list[Probe]) -> list[dict]:
    """Generate completions and score marker leakage.

    The training tasks are Python tasks. The probe prefers Python
    holdout tasks so that pasted training code is comparable.
    """
    holdout = [p for p in probes if p.source == "holdout"]
    holdout.sort(key=lambda p: not p.runner.startswith("python"))
    holdout = holdout[:3]
    neutral = [p for p in probes if p.source == "neutral"][:3]
    records = []
    for probe in holdout + neutral:
        text = _generate_text(model, tokenizer, probe.prompt)
        leaks = _count_leaks(probe.prompt, text)
        total = sum(leaks.values())
        records.append(
            {
                "name": probe.name,
                "source": probe.source,
                "leaks": leaks,
                "leak_total": total,
                "leak_score": round(min(1.0, total / LEAK_CAP), 4),
                "completion_excerpt": text[:400],
            }
        )
    return records


def run_gate(
    model_id: str,
    adapter_dir: Path | None,
    sae_weights: Path | None,
    out_dir: Path,
    holdout_dir: Path,
    tasks_dir: Path,
    data_dir: Path,
    *,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
) -> dict:
    """Run both checks, write the artifact, return the verdict."""
    gpu = require_metal_gpu()
    if adapter_dir is not None and not (adapter_dir / "adapters.safetensors").is_file():
        raise GateError(f"no adapter at {adapter_dir}")
    probes = _collect_probes(holdout_dir, tasks_dir)

    sae_path, provenance = _ensure_sae(out_dir, model_id, sae_weights, data_dir)
    weights = mx.load(str(sae_path))
    if "enc_w" not in weights or "enc_b" not in weights:
        raise GateError(f"{sae_path} has no encoder weights")
    enc_w = weights["enc_w"]
    enc_b = weights["enc_b"]

    model, tokenizer = load(model_id)
    # Quantized embeddings pack their weight, so measure the
    # embedding output width instead of the stored shape.
    hidden_dim = int(model.model.embed_tokens(mx.array([[0]])).shape[-1])
    if enc_w.shape[1] != hidden_dim:
        raise GateError(
            f"SAE expects dim {enc_w.shape[1]}, model has {hidden_dim}; "
            "train an SAE for this base model"
        )
    base_profiles = _capture_profiles(model, tokenizer, enc_w, enc_b, probes)

    if adapter_dir is None:
        # Control mode: a second base pass stands in for the adapter.
        adapted_profiles = _capture_profiles(model, tokenizer, enc_w, enc_b, probes)
        generations = _leakage_records(model, tokenizer, probes)
    else:
        del model
        gc.collect()
        mx.clear_cache()
        model, tokenizer = load(model_id, adapter_path=str(adapter_dir))
        adapted_profiles = _capture_profiles(model, tokenizer, enc_w, enc_b, probes)
        generations = _leakage_records(model, tokenizer, probes)

    probe_rows = []
    for probe, base_p, adapted_p in zip(
        probes, base_profiles, adapted_profiles, strict=True
    ):
        overlap, shift = _probe_metrics(base_p, adapted_p)
        probe_rows.append(
            {
                "name": probe.name,
                "source": probe.source,
                "top_k_overlap": round(overlap, 4),
                "activation_shift": round(shift, 4),
                "drift": round(0.5 * (1.0 - overlap) + 0.5 * shift, 4),
            }
        )

    drift_score = sum(row["drift"] for row in probe_rows) / len(probe_rows)
    leakage_score = sum(record["leak_score"] for record in generations) / len(
        generations
    )
    memorization = DRIFT_WEIGHT * drift_score + LEAK_WEIGHT * leakage_score
    flagged = memorization >= MEMORIZATION_THRESHOLD
    curve_verdict = _train_curve_verdict(
        adapter_dir,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"gate-{stamp}.json"
    payload = {
        "model": model_id,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "device": gpu.device_name,
        "sae_weights": str(sae_path),
        "sae_provenance": provenance,
        "sae_layer": SAE_LAYER,
        "top_k": TOP_K,
        "max_gen_tokens": MAX_GEN_TOKENS,
        "leak_markers": list(LEAK_MARKERS),
        "probes": probe_rows,
        "generations": generations,
        "drift_score": round(drift_score, 4),
        "leakage_score": round(leakage_score, 4),
        "drift_weight": DRIFT_WEIGHT,
        "leak_weight": LEAK_WEIGHT,
        "memorization_score": round(memorization, 4),
        "threshold": MEMORIZATION_THRESHOLD,
        "flagged": flagged,
        "train_drop_pct": train_drop_pct,
        "val_drop_ratio": val_drop_ratio,
        "train_curve_memorization": curve_verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact": str(artifact),
    }
    artifact.write_text(json.dumps(payload, indent=2))
    print(f"gate artifact: {artifact}")
    print(
        f"drift {drift_score:.4f}  leakage {leakage_score:.4f}  "
        f"memorization {memorization:.4f}  "
        f"threshold {MEMORIZATION_THRESHOLD}  flagged {flagged}"
    )
    return payload
