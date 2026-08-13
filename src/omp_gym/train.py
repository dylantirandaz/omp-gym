"""LoRA training on the Metal GPU through mlx-lm.

The preflight gate runs first. Training that does not lower the
train loss is reported as a failure, not as a success.
"""

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .preflight import require_metal_gpu

_LOSS_PATTERN = re.compile(r"Train loss ([0-9.]+)")
_VAL_LOSS_PATTERN = re.compile(r"Val loss ([0-9.]+)")


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


class TrainError(SystemExit):
    """Raised when training did not run or did not learn."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"training failed: {reason}")


def _stream_trainer(
    command: list[str],
) -> tuple[list[float], list[float]]:
    """Run a trainer, echo output, and collect loss curves."""
    print("+", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    losses: list[float] = []
    val_losses: list[float] = []
    nan_lines = 0
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if "loss nan" in line.lower():
            nan_lines += 1
        found = _LOSS_PATTERN.search(line)
        if found:
            losses.append(float(found.group(1)))
        val_found = _VAL_LOSS_PATTERN.search(line)
        if val_found:
            val_losses.append(float(val_found.group(1)))
    exit_code = process.wait()

    if exit_code != 0:
        raise TrainError(f"trainer exited with {exit_code}")
    if len(losses) < 2:
        raise TrainError("no loss reports found in training output")
    if nan_lines:
        raise TrainError(
            f"{nan_lines} loss reports were NaN; "
            "a sample lost its completion to truncation"
        )
    if losses[-1] >= losses[0]:
        raise TrainError(
            f"train loss did not go down: {losses[0]} -> {losses[-1]}"
        )
    return losses, val_losses


def _finish_report(
    model: str,
    data_dir: Path,
    iterations: int,
    adapter_dir: Path,
    losses: list[float],
    val_losses: list[float],
    device_name: str,
) -> TrainReport:
    """Validate the adapter artifact and write the train report."""
    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.is_file():
        raise TrainError(f"{adapter_file} was not written")
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
    )
    (adapter_dir / "train_report.json").write_text(
        json.dumps(asdict(report), indent=2)
    )
    print(
        f"training ok: loss {report.first_train_loss} -> "
        f"{report.last_train_loss} on {report.device_name}"
    )
    return report


def _require_data(data_dir: Path) -> None:
    """Stop when the training data is missing or empty."""
    train_file = data_dir / "train.jsonl"
    if not train_file.is_file() or not train_file.read_text().strip():
        raise TrainError(f"{train_file} is missing or empty")


def run_training(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    max_seq_length: int,
) -> TrainReport:
    """Run one real SFT LoRA pass and validate the loss curve."""
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
        "--mask-prompt",
    ]
    losses, val_losses = _stream_trainer(command)
    return _finish_report(
        model,
        data_dir,
        iterations,
        adapter_dir,
        losses,
        val_losses,
        gpu.device_name,
    )


def run_dpo_training(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    resume_adapter: Path | None,
) -> TrainReport:
    """Run one real DPO pass on preference pairs.

    With --resume-adapter the DPO pass continues from an SFT
    adapter, which is the standard SFT-then-DPO chain.
    """
    gpu = require_metal_gpu()
    _require_data(data_dir)
    command = [
        sys.executable,
        "-m",
        "mlx_lm_lora",
        "train",
        "--train",
        "--train-mode",
        "dpo",
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
        "--steps-per-report",
        "1",
        "--steps-per-eval",
        str(iterations),
        "--val-batches",
        "4",
        "--save-every",
        str(iterations),
        "--learning-rate",
        "5e-6",
    ]
    if resume_adapter is not None:
        if not resume_adapter.is_file():
            raise TrainError(f"resume adapter {resume_adapter} missing")
        command.extend(["--resume-adapter-file", str(resume_adapter)])
    losses, val_losses = _stream_trainer(command)
    return _finish_report(
        model,
        data_dir,
        iterations,
        adapter_dir,
        losses,
        val_losses,
        gpu.device_name,
    )
