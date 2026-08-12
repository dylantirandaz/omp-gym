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


@dataclass(frozen=True)
class TrainReport:
    """Verified facts about one completed training run."""

    model: str
    data_dir: str
    adapter_dir: str
    iterations: int
    first_train_loss: float
    last_train_loss: float
    device_name: str


class TrainError(SystemExit):
    """Raised when training did not run or did not learn."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"training failed: {reason}")


def run_training(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    max_seq_length: int,
) -> TrainReport:
    """Run one real LoRA training pass and validate the loss curve."""
    gpu = require_metal_gpu()

    train_file = data_dir / "train.jsonl"
    if not train_file.is_file() or not train_file.read_text().strip():
        raise TrainError(f"{train_file} is missing or empty")

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
    ]
    print("+", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    losses: list[float] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        found = _LOSS_PATTERN.search(line)
        if found:
            losses.append(float(found.group(1)))
    exit_code = process.wait()

    if exit_code != 0:
        raise TrainError(f"mlx_lm lora exited with {exit_code}")
    if len(losses) < 2:
        raise TrainError("no loss reports found in training output")
    if losses[-1] >= losses[0]:
        raise TrainError(
            f"train loss did not go down: {losses[0]} -> {losses[-1]}"
        )
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
        device_name=gpu.device_name,
    )
    (adapter_dir / "train_report.json").write_text(
        json.dumps(asdict(report), indent=2)
    )
    print(
        f"training ok: loss {report.first_train_loss} -> "
        f"{report.last_train_loss} on {report.device_name}"
    )
    return report
