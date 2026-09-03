"""Prime Verifiers v1 environment for OMP coding tasks.

The Verifiers classes live in :mod:`omp_coding.environment` and import the
Prime runtime stack, which is available only on Linux and macOS. They are
exported lazily so the offline tools (session reader, task minter, benchmark
aggregation) stay importable everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import OmpEnv, OmpHarness, OmpTaskset

__all__ = ["OmpTaskset", "OmpHarness", "OmpEnv"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import environment

        return getattr(environment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
