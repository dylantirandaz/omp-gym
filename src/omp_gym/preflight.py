"""Fail-fast check for the Apple silicon GPU (Metal).

Training must run on the Metal GPU. This module is the single
hardware gate. It stops the program when the GPU is not usable.
Preflight success is not proof that training works.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuReport:
    """Verified facts about the selected accelerator."""

    backend: str
    device_name: str
    architecture: str
    memory_bytes: int


class PreflightError(SystemExit):
    """Raised when the intended accelerator is not usable."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"preflight failed: {reason}")


def require_metal_gpu() -> GpuReport:
    """Verify that the Metal GPU is selected and does real work.

    The check runs one matrix product on the GPU and validates the
    result. A wrong device or a wrong result stops the program.
    The mlx import is local so that machines without mlx can
    import this module; they fail here, at the gate.
    """
    try:
        import mlx.core as mx
    except ModuleNotFoundError as error:
        raise PreflightError(
            "mlx is not installed; training needs Apple silicon"
        ) from error

    device = mx.default_device()
    if device.type != mx.DeviceType.gpu:
        raise PreflightError(
            f"default device is {device}, expected the Metal GPU; "
            "do not fall back to CPU"
        )
    if not mx.is_available(mx.gpu):
        raise PreflightError("Metal GPU backend is not available")

    ones = mx.ones((256, 256))
    product = mx.matmul(ones, ones)
    mx.eval(product)
    corner = product[0, 0].item()
    if corner != 256.0:
        raise PreflightError(
            f"GPU matmul returned {corner}, expected 256.0"
        )

    info = mx.device_info()
    report = GpuReport(
        backend="mlx/metal",
        device_name=str(info["device_name"]),
        architecture=str(info["architecture"]),
        memory_bytes=int(info["max_recommended_working_set_size"]),
    )
    print(
        f"preflight ok: backend={report.backend} "
        f"device={report.device_name} arch={report.architecture} "
        f"memory={report.memory_bytes // (1024 * 1024)} MiB"
    )
    return report
