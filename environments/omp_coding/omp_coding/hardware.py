"""Verify the required Apple Metal execution device."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal


@dataclass(frozen=True)
class MetalFailure:
    reason: str


@dataclass(frozen=True)
class MetalReport:
    backend: Literal["metal"]
    logical_device: Literal["gpu:0"]
    device_name: str
    architecture: str
    memory_bytes: int
    mlx_version: str
    dtype: Literal["float32"]
    check_value: float


def metal_preflight() -> MetalReport | MetalFailure:
    """Run one checked matrix operation on the Apple Metal GPU."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return MetalFailure("execution requires Apple silicon on macOS")
    try:
        import mlx.core as mx

        mlx_version = version("mlx")
    except (ImportError, PackageNotFoundError) as error:
        return MetalFailure(f"MLX is not installed: {error}")
    if not mx.metal.is_available():
        return MetalFailure("the Metal GPU is not available")
    mx.set_default_device(mx.gpu)
    if str(mx.default_device()) != "Device(gpu, 0)":
        return MetalFailure(
            f"the default MLX device is not gpu:0: {mx.default_device()}"
        )
    left = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    right = mx.array([[2.0, 0.0], [1.0, 2.0]], dtype=mx.float32)
    result = mx.matmul(left, right)
    mx.eval(result)
    check_value = float(result[1, 1].item())
    if check_value != 8.0:
        return MetalFailure(f"the Metal operation returned {check_value}, expected 8.0")
    device = mx.device_info()
    return MetalReport(
        backend="metal",
        logical_device="gpu:0",
        device_name=str(device.get("device_name", "unknown")),
        architecture=str(device.get("architecture", "unknown")),
        memory_bytes=int(device.get("memory_size", 0)),
        mlx_version=mlx_version,
        dtype="float32",
        check_value=check_value,
    )
