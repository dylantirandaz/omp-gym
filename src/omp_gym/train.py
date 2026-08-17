"""LoRA training on the Metal GPU through mlx-lm.

The preflight gate runs first. Training that does not lower the
train loss is reported as a failure, not as a success. The trainer
runs as a scrubbed subprocess with a hard wall-clock deadline: at
the deadline the whole process group is killed and the run fails.
The adapter artifact is verified after the run — fresh, nonempty,
parseable as safetensors, and holding LoRA tensors — and the
resolved model revision is pinned in the report.
"""

import json
import math
import os
import queue
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dpo import DPO_BETA, MAX_SEQ
from .isolation import scrub_secret_environment
from .layers import LayerSelection, layer_config_fields, mlx_num_layers
from .preflight import require_metal_gpu

# One loss number as printed by the trainer: plain decimals,
# scientific notation, inf, and NaN all parse as floats. Case and
# the separator (space or colon) vary between trainer versions.
_LOSS_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf(?:inity)?|nan)"
_LOSS_PATTERN = re.compile(rf"\btrain loss[:\s]+{_LOSS_NUMBER}", re.IGNORECASE)
_VAL_LOSS_PATTERN = re.compile(rf"\bval loss[:\s]+{_LOSS_NUMBER}", re.IGNORECASE)

# The trainer subprocess keeps these variables even though some of
# them look like credentials: HF_TOKEN authorizes gated model
# downloads and HF_HOME locates the local snapshot cache.
_TRAINER_ENV_KEEP = ("PATH", "HOME", "TMPDIR", "LANG", "HF_HOME", "HF_TOKEN")


@dataclass(frozen=True)
class TrainReport:
    """Verified facts about one completed training run."""

    model: str
    data_dir: str
    adapter_dir: str
    iterations: int
    first_train_loss: float
    last_train_loss: float
    first_val_loss: float | None
    last_val_loss: float | None
    device_name: str
    train_series: list[float] | None = None
    model_revision: str | None = None
    stale_adapter_files: list[str] = field(default_factory=list)
    train_drop_pct: float = 0.15
    val_drop_ratio: float = 0.25
    beta: float | None = None
    seed: int | None = None
    batch_size: int | None = None
    grad_clip: float | None = None
    fixed_first_loss: float | None = None
    fixed_last_loss: float | None = None
    val_accuracy: float | None = None
    topology_source: str | None = None


class TrainError(SystemExit):
    """Raised when training did not run or did not learn."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"training failed: {reason}")


def _validate_loss_curves(
    losses: list[float],
    val_losses: list[float],
    *,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
) -> None:
    """Stop when the loss curves show no real learning.

    The train loss must go down. When val losses exist, the last
    val loss must not be higher than the first: a rise means the
    adapter fit the train set and got worse on held-out data.
    A memorization-shaped run also fails; the shape check itself
    lives in gate.memorization_shaped_run, the single home for it.
    """
    for name, series in (("train", losses), ("validation", val_losses)):
        for value in series:
            if not math.isfinite(value):
                kind = "NaN" if math.isnan(value) else "infinite"
                raise TrainError(f"{name} loss is {kind}: {value}")
    if len(losses) < 2:
        raise TrainError("no loss reports found in training output")
    if losses[-1] >= losses[0]:
        raise TrainError(f"train loss did not go down: {losses[0]} -> {losses[-1]}")
    if val_losses and val_losses[-1] > val_losses[0]:
        raise TrainError(f"val loss went up: {val_losses[0]} -> {val_losses[-1]}")
    from .gate import memorization_shaped_run

    shaped = memorization_shaped_run(
        losses,
        val_losses,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )
    if shaped is not None:
        raise TrainError(shaped)


def _trainer_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for the trainer subprocess.

    Secret-shaped variables stay out. PATH, HOME, TMPDIR, LANG,
    HF_HOME, and HF_TOKEN survive even where their names match a
    credential pattern, because the trainer cannot locate the
    interpreter, the snapshot cache, or a gated model without them.
    """
    source = os.environ if environ is None else environ
    return scrub_secret_environment(source, keep=_TRAINER_ENV_KEEP)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Kill the trainer's whole process group, then the process.

    The trainer spawns its own children (tokenizer workers,
    compilers). Killing only the leader would leave them running
    against a run that already failed.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (AttributeError, OSError):
        process.kill()


def _stream_trainer(
    command: list[str],
    *,
    max_seconds: float,
    env: dict[str, str] | None = None,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[float], list[float]]:
    """Run a trainer, echo output, and collect loss curves.

    The trainer runs in its own process group. When it runs past
    max_seconds the whole group is killed and the run fails; a
    trainer wedged without producing output cannot run forever.
    """
    print("+", " ".join(command))
    process = popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    deadline = clock() + max_seconds
    lines: queue.Queue[str | None] = queue.Queue()

    def drain() -> None:
        stdout = process.stdout
        if stdout is None:
            lines.put(None)
            return
        for line in stdout:
            lines.put(line)
        lines.put(None)

    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()

    losses: list[float] = []
    val_losses: list[float] = []
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            _terminate_process_group(process)
            process.wait()
            drain_thread.join(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
            raise TrainError(
                f"trainer exceeded the {max_seconds:g}s limit; "
                "killed the trainer process group"
            )
        try:
            line = lines.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue
        if line is None:
            break
        print(line, end="")
        found = _LOSS_PATTERN.search(line)
        if found:
            losses.append(float(found.group(1)))
        val_found = _VAL_LOSS_PATTERN.search(line)
        if val_found:
            val_losses.append(float(val_found.group(1)))
    exit_code = process.wait()
    if process.stdout is not None:
        process.stdout.close()

    if exit_code != 0:
        raise TrainError(f"trainer exited with {exit_code}")
    _validate_loss_curves(
        losses,
        val_losses,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )
    return losses, val_losses


_LORA_KEY_PATTERN = re.compile(r"lora_[ab]")


def _read_safetensors_header(adapter_file: Path) -> dict | None:
    """Parse the safetensors JSON header; None when unreadable.

    The format is an 8-byte little-endian header length followed by
    a JSON object mapping tensor names to their descriptors. A
    truncated or garbage write fails the parse instead of passing
    as a finished adapter.
    """
    try:
        size = adapter_file.stat().st_size
        with adapter_file.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) < 8:
                return None
            (length,) = struct.unpack("<Q", length_bytes)
            if length > size - 8:
                return None
            header = json.loads(stream.read(length))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return header if isinstance(header, dict) else None


def _require_fresh_adapter(adapter_dir: Path, started_at: float) -> list[str]:
    """Stop when the adapter artifact is not a verified new adapter.

    The artifact must be fresh (mtime at or after the run start),
    nonempty, parseable as safetensors, and holding at least one
    LoRA-shaped tensor key. Returns the names of the other files in
    the adapter directory that predate the run, so the report can
    list leftovers from earlier runs.
    """
    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.is_file():
        raise TrainError(f"{adapter_file} was not written")
    if adapter_file.stat().st_mtime < started_at:
        raise TrainError(
            f"{adapter_file} is a stale artifact from an earlier run; "
            "the trainer did not write a new adapter"
        )
    if adapter_file.stat().st_size == 0:
        raise TrainError(f"{adapter_file} is empty; the trainer did not finish")
    header = _read_safetensors_header(adapter_file)
    if header is None:
        raise TrainError(
            f"{adapter_file} has no parseable safetensors header; "
            "the trainer did not finish writing it"
        )
    if not any(_LORA_KEY_PATTERN.search(key) for key in header):
        raise TrainError(
            f"{adapter_file} holds no LoRA tensors; "
            "whatever wrote it was not the LoRA trainer"
        )
    return sorted(
        path.name
        for path in adapter_dir.iterdir()
        if path.is_file()
        and path.name not in ("adapters.safetensors", "train_report.json")
        and path.stat().st_mtime < started_at
    )


def _resolve_model_revision(model_id: str, hub_dir: Path | None = None) -> str | None:
    """The HF snapshot commit hash for a cached model id, when known.

    The Hugging Face cache keeps refs/<name> files holding the
    commit hash each ref points at, and snapshots/<hash>/ holding
    the files. A ref is trusted only when its snapshot directory
    exists. A local path, an uncached id, or an unreadable cache
    gives None: the revision is unknown, not wrong.
    """
    if Path(model_id).exists():
        return None
    if hub_dir is None:
        hf_home = os.environ.get("HF_HOME")
        hub_dir = (
            Path(hf_home) / "hub"
            if hf_home
            else Path.home() / ".cache" / "huggingface" / "hub"
        )
    refs = hub_dir / ("models--" + model_id.replace("/", "--")) / "refs"
    if not refs.is_dir():
        return None
    for ref in sorted(refs.iterdir()):
        try:
            revision = ref.read_text().strip()
        except OSError:
            continue
        if revision and (refs.parent / "snapshots" / revision).is_dir():
            return revision
    return None


def _finish_report(
    model: str,
    data_dir: Path,
    iterations: int,
    adapter_dir: Path,
    losses: list[float],
    val_losses: list[float],
    device_name: str,
    started_at: float,
    *,
    model_revision: str | None = None,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
) -> TrainReport:
    """Validate the adapter artifact and write the train report."""
    stale_files = _require_fresh_adapter(adapter_dir, started_at)
    report = TrainReport(
        model=model,
        data_dir=str(data_dir),
        adapter_dir=str(adapter_dir),
        iterations=iterations,
        first_train_loss=losses[0],
        last_train_loss=losses[-1],
        first_val_loss=val_losses[0] if val_losses else None,
        last_val_loss=val_losses[-1] if val_losses else None,
        device_name=device_name,
        train_series=losses[:: max(1, len(losses) // 100)],
        model_revision=model_revision,
        stale_adapter_files=stale_files,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )
    (adapter_dir / "train_report.json").write_text(json.dumps(asdict(report), indent=2))
    print(
        f"training ok: loss {report.first_train_loss} -> "
        f"{report.last_train_loss} on {report.device_name}"
    )
    return report


def _validate_training_parameters(
    iterations: int,
    batch_size: int,
    max_seq_length: int,
    learning_rate: float,
    max_train_seconds: int,
    train_drop_pct: float,
    val_drop_ratio: float,
) -> None:
    """Reject settings that cannot produce a valid SFT run."""
    if iterations < 2:
        raise TrainError("iterations must be at least 2")
    if batch_size < 1:
        raise TrainError("batch size must be positive")
    if max_seq_length < 1:
        raise TrainError("maximum sequence length must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise TrainError("learning rate must be finite and positive")
    if max_train_seconds < 1:
        raise TrainError("maximum training time must be positive")
    if not math.isfinite(train_drop_pct) or not 0.0 <= train_drop_pct <= 1.0:
        raise TrainError("train drop threshold must be from 0 through 1")
    if not math.isfinite(val_drop_ratio) or not 0.0 <= val_drop_ratio <= 1.0:
        raise TrainError("validation drop ratio must be from 0 through 1")


def _require_data(data_dir: Path) -> None:
    """Stop when the training data is missing or empty."""
    train_file = data_dir / "train.jsonl"
    if not train_file.is_file() or not train_file.read_text().strip():
        raise TrainError(f"{train_file} is missing or empty")


def _record_layer_selection(adapter_dir: Path, selection: LayerSelection) -> None:
    """Add the explicit layer selection to the mlx-lm adapter config."""
    config_path = adapter_dir / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TrainError(
            f"cannot read adapter config {config_path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise TrainError(f"adapter config {config_path} is not an object")
    config.update(layer_config_fields(selection))
    config_path.write_text(json.dumps(config, indent=2))


def run_training(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    max_seq_length: int,
    num_layers: LayerSelection,
    learning_rate: float,
    resume_adapter: Path | None = None,
    max_train_seconds: int = 14400,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
) -> TrainReport:
    """Run one real SFT LoRA pass and validate the loss curve.

    With resume_adapter, the pass continues from an existing
    adapter — the way to close a coverage gap without relearning
    what the base adapter already knows. The trainer gets a
    scrubbed environment and a hard wall-clock deadline of
    max_train_seconds; at the deadline its process group is killed
    and the run fails.
    """
    _validate_training_parameters(
        iterations,
        batch_size,
        max_seq_length,
        learning_rate,
        max_train_seconds,
        train_drop_pct,
        val_drop_ratio,
    )
    gpu = require_metal_gpu()
    _require_data(data_dir)
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--model",
        model,
        "--data",
        str(data_dir),
        "--iters",
        str(iterations),
        "--adapter-path",
        str(adapter_dir),
        "--batch-size",
        str(batch_size),
        "--max-seq-length",
        str(max_seq_length),
        "--steps-per-report",
        "1",
        "--num-layers",
        str(mlx_num_layers(num_layers)),
        "--learning-rate",
        str(learning_rate),
        "--mask-prompt",
    ]
    if resume_adapter is not None:
        if not resume_adapter.is_file():
            raise TrainError(f"resume adapter {resume_adapter} is missing")
        command.extend(["--resume-adapter-file", str(resume_adapter)])
    started_at = time.time()
    model_revision = _resolve_model_revision(model)
    losses, val_losses = _stream_trainer(
        command,
        max_seconds=float(max_train_seconds),
        env=_trainer_environment(),
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )
    _record_layer_selection(adapter_dir, num_layers)
    return _finish_report(
        model,
        data_dir,
        iterations,
        adapter_dir,
        losses,
        val_losses,
        gpu.device_name,
        started_at,
        model_revision=model_revision,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )


def run_dpo_training(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    learning_rate: float,
    resume_adapter: Path | None,
    *,
    max_seq_length: int = MAX_SEQ,
    seed: int = 0,
    beta: float = DPO_BETA,
    grad_clip: float = 1.0,
    train_drop_pct: float = 0.15,
    val_drop_ratio: float = 0.25,
) -> TrainReport:
    """Run one real DPO pass on preference pairs.

    The community trainer deadlocked on this stack, so the sigmoid
    DPO loss runs natively in MLX (see dpo.py). The resume adapter
    is required: it seeds the policy and defines the frozen
    reference, which makes this the standard SFT-then-DPO chain.
    Each optimizer step uses up to batch_size pairs and one
    mean batch gradient. max_seq_length caps padded pair length.
    """
    from .dpo import train_dpo

    gpu = require_metal_gpu()
    _require_data(data_dir)
    if resume_adapter is None or not resume_adapter.is_file():
        raise TrainError("dpo requires --resume-adapter pointing at an SFT adapter")
    started_at = time.time()
    model_revision = _resolve_model_revision(model)
    metrics = train_dpo(
        data_dir=data_dir,
        model_id=model,
        iterations=iterations,
        adapter_dir=adapter_dir,
        resume_adapter=resume_adapter,
        learning_rate=learning_rate,
        device_name=gpu.device_name,
        batch_size=batch_size,
        seed=seed,
        beta=beta,
        grad_clip=grad_clip,
        max_seq=max_seq_length,
    )
    stale_files = _require_fresh_adapter(adapter_dir, started_at)
    # The decision signal is the loss on the fixed probe set; the
    # per-batch train scalars are noisy under batching and shuffle
    # and stay in the report for the record only.
    _validate_loss_curves(
        [metrics["fixed_first_loss"], metrics["fixed_last_loss"]],
        [metrics["first_val_loss"], metrics["last_val_loss"]]
        if metrics["first_val_loss"] is not None
        else [],
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
    )
    report = TrainReport(
        model=model,
        data_dir=str(data_dir),
        adapter_dir=str(adapter_dir),
        iterations=iterations,
        first_train_loss=metrics["first_train_loss"],
        last_train_loss=metrics["last_train_loss"],
        first_val_loss=metrics["first_val_loss"],
        last_val_loss=metrics["last_val_loss"],
        device_name=gpu.device_name,
        model_revision=model_revision,
        stale_adapter_files=stale_files,
        train_drop_pct=train_drop_pct,
        val_drop_ratio=val_drop_ratio,
        beta=metrics.get("beta"),
        seed=metrics.get("seed"),
        batch_size=metrics.get("batch_size", batch_size),
        grad_clip=metrics.get("grad_clip"),
        fixed_first_loss=metrics.get("fixed_first_loss"),
        fixed_last_loss=metrics.get("fixed_last_loss"),
        val_accuracy=metrics.get("val_accuracy"),
        topology_source=metrics.get("topology_source"),
    )
    (adapter_dir / "train_report.json").write_text(json.dumps(asdict(report), indent=2))
    return report
