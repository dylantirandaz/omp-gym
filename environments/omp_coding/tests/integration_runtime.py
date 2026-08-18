"""Run the real OMP install and sealed verifier path in Verifiers Docker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Literal

from omp_coding import (
    COMMAND_USER_ID,
    OMP_BINARY,
    OMP_VERSION,
    OmpEnvConfig,
    OmpHarness,
    OmpHarnessConfig,
    OmpTask,
    OmpTaskset,
    OmpTasksetConfig,
    _clean_candidate,
    _invoke_candidate,
    _stage_candidate_support,
)
from omp_coding.runtime import RuntimeFailure
from omp_coding.verifier import run_verifier_cases
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.utils.compile import resolve_runtime_config


async def run() -> dict[str, object]:
    taskset = OmpTaskset(OmpTasksetConfig(id="omp-coding"))
    task = next(task for task in taskset if task.data.name == "fizzbuzz-fix")
    if not isinstance(task, OmpTask):
        raise TypeError("the taskset did not yield an OmpTask")
    environment = OmpEnvConfig(taskset=OmpTasksetConfig(id="omp-coding"))
    runtime_config = resolve_runtime_config(environment.agent.runtime, task)
    harness = OmpHarness(OmpHarnessConfig(id="omp-coding"))

    async with provision_runtime(runtime_config) as runtime:
        await runtime.prepare_setup()
        await harness.setup(runtime)
        version = await runtime.run([OMP_BINARY, "--version"], {})
        architecture = await runtime.run(["uname", "-m"], {})
        if version.exit_code != 0 or version.stdout.strip() != f"omp/{OMP_VERSION}":
            raise RuntimeError("the pinned OMP binary did not run")
        if architecture.exit_code != 0 or architecture.stdout.strip() != "aarch64":
            raise RuntimeError("the runtime is not arm64")

        await task.setup(runtime)
        await _stage_candidate_support(runtime)
        await runtime.prepare_execution([])
        scratch = await runtime.run(
            [
                "setpriv",
                f"--reuid={COMMAND_USER_ID}",
                f"--regid={COMMAND_USER_ID}",
                "--clear-groups",
                "--no-new-privs",
                "sh",
                "-c",
                "printf state > /dev/shm/omp-candidate-state && "
                "printf state > /var/tmp/omp-candidate-state",
            ],
            {},
        )
        if scratch.exit_code != 0:
            raise RuntimeError("candidate scratch setup failed")
        clean_failure = await _clean_candidate(runtime)
        if clean_failure is not None:
            raise RuntimeError(clean_failure.reason)
        scratch_probe = await runtime.run(
            [
                "sh",
                "-c",
                "test ! -e /dev/shm/omp-candidate-state && "
                "test ! -e /var/tmp/omp-candidate-state",
            ],
            {},
        )
        if scratch_probe.exit_code != 0:
            raise RuntimeError("candidate scratch state survived cleanup")

        async def invoke(
            language: Literal["python", "node"],
            request: Mapping[str, object],
            timeout_seconds: float,
        ) -> Mapping[str, object] | RuntimeFailure:
            return await _invoke_candidate(
                runtime,
                language,
                request,
                timeout_seconds,
            )

        result = await run_verifier_cases(
            task.suite,
            invoke,
            timeout_seconds=float(task.spec.max_time_seconds),
        )

    if result.status != "failed":
        raise RuntimeError(f"unexpected baseline status: {result.status}")
    if (result.passed_cases, result.total_cases) != (7, 10):
        raise RuntimeError(
            "the fizzbuzz baseline no longer has the expected sealed score"
        )
    return {
        "runtime": "verifiers-docker",
        "image": runtime_config.image,
        "architecture": architecture.stdout.strip(),
        "omp_version": version.stdout.strip(),
        "task_id": task.data.task_id,
        "passed_cases": result.passed_cases,
        "total_cases": result.total_cases,
        "scratch_reset": True,
        "reward": result.reward,
    }


def main() -> int:
    print(json.dumps(asyncio.run(run()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
