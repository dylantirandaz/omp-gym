"""Prime Verifiers v1 taskset, harness, and environment for OMP coding tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import shlex
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from aiohttp import web
from verifiers.v1.configs.env import TimeoutConfig as EnvTimeoutConfig
from verifiers.v1.interception.tunnel import PrimeTunnel
from verifiers.v1.runtimes import RuntimeConfig, provision_runtime
from verifiers.v1.utils.artifacts import collect, restore
from verifiers.v1.utils.compile import resolve_runtime_config

from .runtime import MAX_COMMAND_OUTPUT_BYTES, RUNTIME_IMAGE, SAFE_PATH, RuntimeFailure
from .task import TaskLoadError, TaskSpec, load_task_suite
from .verifier import (
    VerifierLoadFailure,
    VerifierResult,
    VerifierSuite,
    load_verifier_suite,
    run_verifier_cases,
)

logger = logging.getLogger(__name__)

WORKDIR = "/workspace"
POLICY_PROVIDER = "omp-coding"
POLICY_KEY_VARIABLE = "OMP_CODING_POLICY_KEY"
MODEL_CONTEXT_WINDOW = 32_768
DEFAULT_TOTAL_TOKEN_LIMIT = 65_536
DEFAULT_OUTPUT_TOKEN_LIMIT = 4_096

OMP_VERSION = "17.2.15"
OMP_VERSION_OUTPUT = f"omp/{OMP_VERSION}"
OMP_BINARY = f"/opt/omp-gym/omp-{OMP_VERSION}"
OMP_DOWNLOAD_URL = (
    f"https://github.com/can1357/oh-my-pi/releases/download/v{OMP_VERSION}/"
    "omp-linux-arm64"
)
OMP_SHA256 = "36507ba3d98332f52649d22009ead86f154ab007cb169d68690fa2b0111769ad"

PYTHON_WORKER = "/opt/omp-gym-python-candidate.py"
NODE_WORKER = "/opt/omp-gym-node-candidate.mjs"
CLEANER = "/opt/omp-gym-container-tool.py"
RUNTIME_HARNESS = "/opt/omp-gym-runtime-harness.py"
TOOL_REQUEST = "/run/omp-gym-tool.json"
AGENT_USER_ID = 1
COMMAND_USER_ID = 65534
MAX_TEST_INVOCATIONS = 20
MAX_TOOL_REQUEST_BYTES = 5 * 1024 * 1024
PROCESS_PIPE_DRAIN_SECONDS = 1.0
RPC_SUPPORT_FILES = (
    "__init__.py",
    "client.py",
    "host_tools.py",
    "host_uris.py",
    "protocol.py",
)
CLEAN_REQUEST = "/run/omp-gym-clean.json"
CASE_REQUEST = "/run/omp-gym-case.json"


def _default_tasks_dir() -> Path:
    return Path(__file__).resolve().parent / "tasks"


def _support_file(name: str) -> bytes:
    path = Path(__file__).resolve().with_name(name)
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"candidate support file is not readable: {path}: {error}"
        ) from error


def _environment_support_file(name: str) -> bytes:
    path = Path(__file__).resolve().with_name(name)
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"harness support file is not readable: {path}: {error}"
        ) from error


def _rpc_support_file(name: str) -> bytes:
    path = Path(__file__).resolve().parent / "_vendor" / "omp_rpc" / name
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"OMP RPC support file is not readable: {path}: {error}"
        ) from error


def _task_system_prompt(task: TaskSpec) -> str:
    editable = ", ".join(task.editable_files)
    return "\n".join(
        (
            f"You work in an isolated Linux workspace at {WORKDIR}.",
            f"Change only these files: {editable}.",
            "The sealed verifier cases are not visible in the workspace.",
            "Use the OMP tools to inspect the code, make the change, and run applicable public checks.",
            "Finish only when the requested behavior is correct or when you can state a specific blocker.",
        )
    )


class OmpTaskData(vf.TaskData):
    """Public and versioned data for one sealed coding task."""

    task_id: str
    task_revision: int
    task_digest: str
    family: str
    split: Literal["train", "validation", "holdout"]
    runtime: Literal["python", "node"]
    runtime_version: str
    token_budget: int
    expected_cases: int
    editable_files: tuple[str, ...]
    context_files: tuple[str, ...]
    source: str
    source_revision: str
    license: str
    seed: int


class OmpTask(vf.Task[OmpTaskData]):
    """Materialize public files and collect only declared editable files."""

    NEEDS_CONTAINER = True

    def __init__(
        self,
        data: OmpTaskData,
        config: vf.TaskConfig | None = None,
        *,
        spec: TaskSpec | None = None,
        suite: VerifierSuite | None = None,
    ) -> None:
        super().__init__(data, config)
        self._spec = spec
        self._suite = suite

    @property
    def spec(self) -> TaskSpec:
        if self._spec is None:
            raise RuntimeError("task source metadata is not available")
        return self._spec

    @property
    def suite(self) -> VerifierSuite:
        if self._suite is None:
            raise RuntimeError("sealed verifier suite is not available")
        return self._suite

    async def setup(self, runtime: vf.Runtime) -> None:
        spec = self.spec
        for relative in spec.context_files:
            source = spec.workspace.joinpath(*Path(relative).parts)
            try:
                content = source.read_bytes()
            except OSError as error:
                raise RuntimeError(
                    f"public task file is not readable: {relative}: {error}"
                ) from error
            await runtime.write(f"{WORKDIR}/{relative}", content)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for relative in self.data.editable_files:
            path = f"{WORKDIR}/{relative}"
            check = await runtime.run(
                [
                    "sh",
                    "-c",
                    f"test -f {shlex.quote(path)} && test ! -L {shlex.quote(path)}",
                ],
                {},
            )
            if check.exit_code != 0:
                raise RuntimeError(
                    f"editable file is missing or is not a plain file: {relative}"
                )
        trace.state.artifacts = await collect(runtime, self.data.artifacts)

    @vf.stop
    async def task_token_budget_reached(self, trace: vf.Trace) -> bool:
        usage = trace.usage
        return usage is not None and usage.total_tokens >= self.data.token_budget


class OmpTasksetConfig(vf.TasksetConfig):
    """Select one fixed package task split."""

    split: Literal["train", "validation", "holdout"] = "train"


class OmpTaskset(vf.Taskset[OmpTask, OmpTasksetConfig]):
    """Yield typed OMP tasks with private verifier state outside TaskData."""

    def load(self) -> list[OmpTask]:
        loaded = load_task_suite(_default_tasks_dir())
        if isinstance(loaded, TaskLoadError):
            raise ValueError(loaded.reason)
        selected = [task for task in loaded if task.split == self.config.split]
        if not selected:
            raise ValueError(f"task split has no tasks: {self.config.split}")

        tasks: list[OmpTask] = []
        for index, spec in enumerate(selected):
            suite = load_verifier_suite(
                spec.verifier_path,
                runtime=spec.runtime,
                expected_cases=spec.expected_cases,
            )
            if isinstance(suite, VerifierLoadFailure):
                raise ValueError(f"task {spec.task_id}: {suite.reason}")
            resources = vf.TaskResources(
                cpu=spec.environment.cpus,
                memory=spec.environment.memory_bytes / (1024**3),
                disk=1.0,
            )
            timeout = vf.TaskTimeout(
                setup=300.0,
                agent=float(spec.max_time_seconds),
                finalize=60.0,
                scoring=float(spec.max_time_seconds),
            )
            data = OmpTaskData(
                idx=index,
                name=spec.name,
                description=f"{spec.family} coding task",
                prompt=spec.prompt,
                system_prompt=_task_system_prompt(spec),
                image=spec.environment.image,
                workdir=WORKDIR,
                network_allow=[],
                network_block=[],
                artifacts=[
                    vf.Artifact(source=f"{WORKDIR}/{relative}")
                    for relative in spec.editable_files
                ],
                timeout=timeout,
                resources=resources,
                task_id=spec.task_id,
                task_revision=spec.task_revision,
                task_digest=spec.task_digest,
                family=spec.family,
                split=spec.split,
                runtime=spec.runtime,
                runtime_version=spec.runtime_version,
                token_budget=spec.token_budget,
                expected_cases=spec.expected_cases,
                editable_files=spec.editable_files,
                context_files=spec.context_files,
                source=spec.source,
                source_revision=spec.source_revision,
                license=spec.license,
                seed=spec.seed,
            )
            tasks.append(OmpTask(data, self.config.task, spec=spec, suite=suite))
        return tasks


class OmpHarnessConfig(vf.HarnessConfig):
    """Configuration for the pinned OMP release."""

    version: Literal["17.2.15"] = OMP_VERSION


class OmpHarness(vf.Harness[OmpHarnessConfig]):
    """Run OMP in the policy runtime through the Verifiers interception endpoint."""

    APPENDS_SYSTEM_PROMPT = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.write(
            RUNTIME_HARNESS,
            _environment_support_file("_runtime_harness.py"),
        )
        await runtime.write(CLEANER, _support_file("_container_tool.py"))
        for name in RPC_SUPPORT_FILES:
            await runtime.write(f"/opt/omp_rpc/{name}", _rpc_support_file(name))
        script = f"""
set -eu
if [ "$(uname -m)" != "aarch64" ]; then
    echo "OMP {OMP_VERSION} requires the arm64 task runtime" >&2
    exit 1
fi
command -v python3 >/dev/null
command -v setpriv >/dev/null
mkdir -p {shlex.quote(str(Path(OMP_BINARY).parent))}
if [ -x {shlex.quote(OMP_BINARY)} ] && printf '%s  %s\n' {shlex.quote(OMP_SHA256)} {shlex.quote(OMP_BINARY)} | sha256sum -c - >/dev/null 2>&1; then
    {shlex.quote(OMP_BINARY)} --version
    exit 0
fi
temporary={shlex.quote(OMP_BINARY)}.download
trap 'rm -f "$temporary"' EXIT
curl --fail --location --silent --show-error --connect-timeout 10 --max-time 300 \
    {shlex.quote(OMP_DOWNLOAD_URL)} --output "$temporary"
printf '%s  %s\n' {shlex.quote(OMP_SHA256)} "$temporary" | sha256sum -c -
chmod 0755 "$temporary"
mv -f "$temporary" {shlex.quote(OMP_BINARY)}
version="$({shlex.quote(OMP_BINARY)} --version)"
if [ "$version" != {shlex.quote(OMP_VERSION_OUTPUT)} ]; then
    echo "unexpected OMP version: $version" >&2
    exit 1
fi
printf '%s\n' "$version"
"""
        logger.info("omp: ensuring pinned release %s is installed", self.config.version)
        result = await runtime.run(["sh", "-c", script], {})
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise RuntimeError(f"OMP install failed: {detail or 'no output'}")

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> vf.ProgramResult:
        if mcp_urls:
            raise ValueError("the OMP harness does not accept external MCP tools")
        if self.config.disabled_tools:
            raise ValueError("the OMP harness tool contract cannot be changed")
        if not isinstance(data, OmpTaskData):
            raise TypeError(
                f"OmpHarness requires OmpTaskData, got {type(data).__name__}"
            )
        system_prompt, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("the OMP harness requires a text prompt")
        output_tokens = (
            DEFAULT_OUTPUT_TOKEN_LIMIT
            if ctx.sampling.max_tokens is None
            else ctx.sampling.max_tokens
        )
        if not 1 <= output_tokens <= MODEL_CONTEXT_WINDOW:
            raise ValueError("the model output token limit is invalid")

        task = _task_for_data(data)
        agent_dir = f"/run/omp-gym/agent-{trace.id}"
        harness_config_path = f"{agent_dir}/harness.json"
        reasoning = ctx.sampling.reasoning_effort not in {
            None,
            "none",
        } or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                POLICY_PROVIDER: {
                    "baseUrl": endpoint,
                    "apiKey": POLICY_KEY_VARIABLE,
                    "auth": "apiKey",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": ctx.model,
                            "name": f"Verifiers policy {ctx.model}",
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                            "contextWindow": MODEL_CONTEXT_WINDOW,
                            "maxTokens": output_tokens,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }
        tool_prompt = "\n".join(
            (
                system_prompt or "",
                "Use only sandbox_read, sandbox_write, sandbox_edit, sandbox_exec, and run_tests.",
                f"You can change only these files: {', '.join(data.editable_files)}.",
                f"Public read-only files: {', '.join(data.context_files)}.",
                "Inspect and run the public test files before you call run_tests.",
                "No verifier source is visible. Use run_tests for behavioral feedback.",
                "Call run_tests after the change.",
                f"The total model token budget is {data.token_budget} tokens.",
            )
        ).strip()
        await runtime.write(
            f"{agent_dir}/models.yml",
            json.dumps(models, ensure_ascii=False, separators=(",", ":")).encode(),
        )
        await runtime.write(
            harness_config_path,
            json.dumps(
                {
                    "executable": OMP_BINARY,
                    "provider": POLICY_PROVIDER,
                    "model": ctx.model,
                    "prompt": prompt,
                    "system_prompt": tool_prompt,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode(),
        )
        editable_permissions = "; ".join(
            f"chmod a=rw -- {shlex.quote(f'{WORKDIR}/{path}')}"
            for path in data.editable_files
        )
        prepared = await runtime.run(
            [
                "sh",
                "-c",
                (
                    "set -eu; "
                    "mkdir -p /tmp/omp-gym-command-home; "
                    f"chown {COMMAND_USER_ID}:{COMMAND_USER_ID} /tmp/omp-gym-command-home; "
                    f"chmod -R a=rX -- {shlex.quote(WORKDIR)}; "
                    f"{editable_permissions}; "
                    f"chown -R {AGENT_USER_ID}:{AGENT_USER_ID} -- {shlex.quote(agent_dir)}; "
                    f"chmod 0700 -- {shlex.quote(agent_dir)}"
                ),
            ],
            {},
        )
        if prepared.exit_code != 0:
            detail = (prepared.stderr or prepared.stdout).strip()[-1000:]
            raise RuntimeError(f"failed to prepare the OMP runtime: {detail}")

        command = [
            "setpriv",
            f"--reuid={AGENT_USER_ID}",
            f"--regid={AGENT_USER_ID}",
            "--clear-groups",
            "--no-new-privs",
            "python3",
            "-I",
            RUNTIME_HARNESS,
            harness_config_path,
        ]
        trace.info["omp_version"] = self.config.version
        trace.info["omp_tool_contract"] = "rpc-host-v1"
        async with _tool_route(runtime, task) as tool_route:
            await runtime.prepare_execution([endpoint, tool_route.url])
            environment = {
                **self.config.resolved_env,
                POLICY_KEY_VARIABLE: secret,
                "HOME": agent_dir,
                "PI_CODING_AGENT_DIR": agent_dir,
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
                "OMP_CODING_TOOL_URL": tool_route.url,
            }
            return await runtime.run_program(command, environment)

    async def cleanup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        result = await runtime.run(["rm", "-rf", f"/run/omp-gym/agent-{trace.id}"], {})
        if result.exit_code != 0:
            raise RuntimeError("failed to remove the OMP rollout home")


@dataclass(frozen=True)
class _ProcessCapture:
    exit_code: int
    stdout: str
    stderr: str


@dataclass
class _OutputBudget:
    remaining_bytes: int
    overflow: asyncio.Event

    def consume(self, chunk: bytes) -> bytes:
        captured = chunk[: self.remaining_bytes]
        self.remaining_bytes -= len(captured)
        if len(captured) != len(chunk):
            self.overflow.set()
        return captured


async def _read_bounded(
    stream: AsyncIterator[bytes],
    output: bytearray,
    budget: _OutputBudget,
) -> None:
    async for chunk in stream:
        output.extend(budget.consume(chunk))


async def _capture_process(
    runtime: vf.Runtime,
    command: list[str],
    timeout_seconds: float,
) -> _ProcessCapture | RuntimeFailure:
    if not runtime.supports_live_processes:
        return RuntimeFailure(
            "unavailable", "candidate runtime does not support live processes"
        )
    process = await runtime.open_process(
        ["sh", "-c", 'sleep 0.1; exec "$@"', "omp-capture", *command],
        {},
    )
    output = bytearray()
    stderr_bytes = bytearray()
    budget = _OutputBudget(MAX_COMMAND_OUTPUT_BYTES, asyncio.Event())
    stdout_task = asyncio.create_task(_read_bounded(process.stdout, output, budget))
    stderr_task = asyncio.create_task(
        _read_bounded(process.stderr, stderr_bytes, budget)
    )
    wait_task = asyncio.create_task(process.wait())
    overflow_task = asyncio.create_task(budget.overflow.wait())
    try:
        done, _ = await asyncio.wait(
            {wait_task, overflow_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if overflow_task in done and budget.overflow.is_set():
            await process.kill()
            await wait_task
            return RuntimeFailure(
                "output_limit",
                "candidate worker exceeded its output limit",
                output.decode(errors="replace"),
                stderr_bytes.decode(errors="replace"),
            )
        if wait_task not in done:
            await process.kill()
            await wait_task
            return RuntimeFailure(
                "timeout",
                "candidate worker exceeded its time limit",
                output.decode(errors="replace"),
                stderr_bytes.decode(errors="replace"),
            )
        readers_done, _ = await asyncio.wait(
            {stdout_task, stderr_task},
            timeout=PROCESS_PIPE_DRAIN_SECONDS,
        )
        if len(readers_done) != 2:
            return RuntimeFailure(
                "process_leak",
                "candidate descendant kept an output pipe open",
                output.decode(errors="replace"),
                stderr_bytes.decode(errors="replace"),
            )
        await asyncio.gather(stdout_task, stderr_task)
        if budget.overflow.is_set():
            return RuntimeFailure(
                "output_limit",
                "candidate worker exceeded its output limit",
                output.decode(errors="replace"),
                stderr_bytes.decode(errors="replace"),
            )
        return _ProcessCapture(
            wait_task.result(),
            output.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
        )
    finally:
        overflow_task.cancel()
        if not wait_task.done():
            await process.kill()
        for reader_task in (stdout_task, stderr_task):
            if not reader_task.done():
                reader_task.cancel()
        await asyncio.gather(
            stdout_task,
            stderr_task,
            wait_task,
            overflow_task,
            return_exceptions=True,
        )


async def _clean_candidate(runtime: vf.Runtime) -> RuntimeFailure | None:
    await runtime.write(CLEAN_REQUEST, b'{"action":"clean"}')
    result = await runtime.run(["python3", "-I", CLEANER, CLEAN_REQUEST], {})
    if result.exit_code == 0:
        return None
    return RuntimeFailure(
        "command_failed",
        "candidate cleanup did not complete",
        result.stdout,
        result.stderr,
    )


async def _invoke_candidate(
    runtime: vf.Runtime,
    language: Literal["python", "node"],
    request: Mapping[str, object],
    timeout_seconds: float,
) -> Mapping[str, object] | RuntimeFailure:
    cleaned = await _clean_candidate(runtime)
    if cleaned is not None:
        return cleaned
    await runtime.write(
        CASE_REQUEST,
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(),
    )
    worker = (
        ["python3", "-I", PYTHON_WORKER, CASE_REQUEST]
        if language == "python"
        else ["node", "--disable-proto=throw", NODE_WORKER, CASE_REQUEST]
    )
    command = [
        "setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--no-new-privs",
        *worker,
    ]
    captured = await _capture_process(runtime, command, timeout_seconds)
    cleanup = await _clean_candidate(runtime)
    await runtime.run(["rm", "-f", CASE_REQUEST], {})
    if isinstance(captured, RuntimeFailure):
        return captured
    if cleanup is not None:
        return cleanup
    if captured.exit_code != 0:
        return RuntimeFailure(
            "command_failed",
            "candidate worker did not complete",
            captured.stdout,
            captured.stderr,
        )
    lines = [line for line in captured.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return RuntimeFailure(
            "protocol_error",
            "candidate worker wrote unexpected output",
            captured.stdout,
            captured.stderr,
        )
    try:
        reply = json.loads(lines[0])
    except (json.JSONDecodeError, RecursionError):
        return RuntimeFailure(
            "protocol_error",
            "candidate worker returned invalid JSON",
            captured.stdout,
            captured.stderr,
        )
    if (
        not isinstance(reply, dict)
        or reply.get("schema_version") != 1
        or reply.get("status") not in {"ok", "error"}
    ):
        return RuntimeFailure(
            "protocol_error", "candidate worker returned an invalid reply"
        )
    return reply


async def _stage_candidate_support(runtime: vf.Runtime) -> None:
    await runtime.write(PYTHON_WORKER, _support_file("_python_candidate.py"))
    await runtime.write(NODE_WORKER, _support_file("_node_candidate.mjs"))
    await runtime.write(CLEANER, _support_file("_container_tool.py"))
    created = await runtime.run(["mkdir", "-p", "/home/solver", "/run"], {})
    if created.exit_code != 0:
        raise RuntimeError("failed to create candidate runtime directories")


def _task_for_data(data: OmpTaskData) -> OmpTask:
    taskset = OmpTaskset(
        OmpTasksetConfig(
            id="omp-coding",
            split=data.split,
        )
    )
    matches = [task for task in taskset if task.data.task_id == data.task_id]
    if len(matches) != 1:
        raise RuntimeError(f"task data does not select one task: {data.task_id}")
    task = matches[0]
    if (
        task.data.task_revision != data.task_revision
        or task.data.task_digest != data.task_digest
    ):
        raise RuntimeError(
            f"task data does not match the installed task: {data.task_id}"
        )
    return task


async def _verify_artifacts(
    task: OmpTask,
    artifacts: dict[str, bytes | None],
    runtime_config: RuntimeConfig,
) -> VerifierResult:
    async with asyncio.timeout(task.data.timeout.scoring):
        async with provision_runtime(runtime_config) as candidate_runtime:
            await candidate_runtime.prepare_setup()
            await task.setup(candidate_runtime)
            await restore(candidate_runtime, artifacts)
            await _stage_candidate_support(candidate_runtime)
            await candidate_runtime.prepare_execution([])

            async def invoke(
                language: Literal["python", "node"],
                request: Mapping[str, object],
                timeout_seconds: float,
            ) -> Mapping[str, object] | RuntimeFailure:
                return await _invoke_candidate(
                    candidate_runtime,
                    language,
                    request,
                    timeout_seconds,
                )

            return await run_verifier_cases(
                task.suite,
                invoke,
                timeout_seconds=float(task.spec.max_time_seconds),
            )


def _artifact_digest(artifacts: Mapping[str, bytes | None]) -> str:
    digest = hashlib.sha256()
    for path, content in sorted(artifacts.items()):
        encoded_path = path.encode()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if content is None:
            digest.update(b"N")
            continue
        digest.update(b"B")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


async def _runtime_tool(
    runtime: vf.Runtime,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    await runtime.write(
        TOOL_REQUEST,
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(),
    )
    try:
        captured = await _capture_process(
            runtime,
            ["python3", "-I", CLEANER, TOOL_REQUEST],
            30.0,
        )
    finally:
        await runtime.run(["rm", "-f", TOOL_REQUEST], {})
    if isinstance(captured, RuntimeFailure):
        raise RuntimeError(captured.reason)
    lines = [line for line in captured.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("workspace tool returned unexpected output")
    try:
        reply = json.loads(lines[0])
    except (json.JSONDecodeError, RecursionError) as error:
        raise RuntimeError(f"workspace tool returned invalid JSON: {error}") from error
    if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
        raise RuntimeError("workspace tool returned an invalid reply")
    if captured.exit_code == 0 and reply["ok"] is not True:
        raise RuntimeError("workspace tool returned an invalid status")
    return reply


def _required_tool_string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _tool_integer(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be from {minimum} to {maximum}")
    return value


@dataclass(frozen=True)
class _ToolRoute:
    url: str


class _ToolController:
    def __init__(self, runtime: vf.Runtime, task: OmpTask) -> None:
        self.runtime = runtime
        self.task = task
        self.lock = asyncio.Lock()
        self.test_invocations = 0
        self.test_cache: dict[str, VerifierResult] = {}

    async def dispatch(
        self,
        action: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        async with self.lock:
            if action == "read":
                return await self._file_action(
                    action,
                    arguments,
                    allowed=frozenset({"path", "line", "limit"}),
                )
            if action == "write":
                return await self._editable_action(
                    action,
                    arguments,
                    allowed=frozenset({"path", "content"}),
                )
            if action == "edit":
                return await self._editable_action(
                    action,
                    arguments,
                    allowed=frozenset({"path", "old_text", "new_text"}),
                )
            if action == "exec":
                return await self._exec(arguments)
            if action == "tests":
                return await self._tests(arguments)
            raise ValueError(f"unknown tool action: {action}")

    async def _file_action(
        self,
        action: Literal["read", "write", "edit"],
        arguments: Mapping[str, object],
        *,
        allowed: frozenset[str],
    ) -> Mapping[str, object]:
        unknown = set(arguments).difference(allowed)
        if unknown:
            raise ValueError(f"{action} has unknown arguments: {sorted(unknown)}")
        reply = await _runtime_tool(self.runtime, {"action": action, **arguments})
        if reply["ok"] is not True:
            reason = reply.get("error")
            raise RuntimeError(
                reason if isinstance(reason, str) else f"{action} failed"
            )
        return reply

    async def _editable_action(
        self,
        action: Literal["write", "edit"],
        arguments: Mapping[str, object],
        *,
        allowed: frozenset[str],
    ) -> Mapping[str, object]:
        path = _required_tool_string(arguments, "path")
        if path not in self.task.data.editable_files:
            raise ValueError("path is not editable")
        return await self._file_action(action, arguments, allowed=allowed)

    async def _exec(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        unknown = set(arguments).difference({"command", "timeout_seconds"})
        if unknown:
            raise ValueError(f"exec has unknown arguments: {sorted(unknown)}")
        command = _required_tool_string(arguments, "command")
        timeout_seconds = _tool_integer(
            arguments,
            "timeout_seconds",
            default=10,
            minimum=1,
            maximum=30,
        )
        captured = await _capture_process(
            self.runtime,
            [
                "setpriv",
                f"--reuid={COMMAND_USER_ID}",
                f"--regid={COMMAND_USER_ID}",
                "--clear-groups",
                "--no-new-privs",
                "env",
                "-i",
                "HOME=/tmp/omp-gym-command-home",
                f"PATH={SAFE_PATH}",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "PYTHONNOUSERSITE=1",
                "PYTHONDONTWRITEBYTECODE=1",
                "NODE_OPTIONS=--disable-proto=throw",
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                command,
            ],
            float(timeout_seconds),
        )
        reap = await _runtime_tool(self.runtime, {"action": "reap"})
        if reap["ok"] is not True:
            raise RuntimeError("sandbox processes did not stop")
        if isinstance(captured, RuntimeFailure):
            raise RuntimeError(captured.reason)
        return {
            "ok": True,
            "text": json.dumps(
                {
                    "returncode": captured.exit_code,
                    "stdout": captured.stdout,
                    "stderr": captured.stderr,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    async def _tests(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        if arguments:
            raise ValueError("run_tests does not accept arguments")
        if self.test_invocations >= MAX_TEST_INVOCATIONS:
            raise RuntimeError("run_tests invocation limit reached")
        self.test_invocations += 1
        artifacts = await collect(self.runtime, self.task.data.artifacts)
        digest = _artifact_digest(artifacts)
        result = self.test_cache.get(digest)
        cached = result is not None
        if result is None:
            result = await _verify_artifacts(
                self.task,
                artifacts,
                self.runtime.config,
            )
            self.test_cache[digest] = result
        return {
            "ok": True,
            "text": json.dumps(
                {
                    "schema_version": 1,
                    "status": result.status,
                    "passed_cases": result.passed_cases,
                    "total_cases": result.total_cases,
                    "cached": cached,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }


@asynccontextmanager
async def _tool_route(
    runtime: vf.Runtime,
    task: OmpTask,
) -> AsyncIterator[_ToolRoute]:
    controller = _ToolController(runtime, task)
    token = secrets.token_urlsafe(32)

    async def handle(request: web.Request) -> web.Response:
        raw = await request.read()
        if len(raw) > MAX_TOOL_REQUEST_BYTES:
            return web.json_response(
                {"ok": False, "error": "tool request is too large"},
                status=413,
            )
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.json_response(
                {"ok": False, "error": "tool request is invalid JSON"},
                status=400,
            )
        if not isinstance(document, dict):
            return web.json_response(
                {"ok": False, "error": "tool request must be an object"},
                status=400,
            )
        action = document.get("action")
        arguments = document.get("arguments")
        if not isinstance(action, str) or not isinstance(arguments, dict):
            return web.json_response(
                {"ok": False, "error": "tool request fields are invalid"},
                status=400,
            )
        try:
            reply = await controller.dispatch(action, arguments)
        except (RuntimeError, ValueError) as error:
            return web.json_response({"ok": False, "error": str(error)})
        return web.json_response(reply)

    application = web.Application(client_max_size=MAX_TOOL_REQUEST_BYTES)
    application.router.add_post(f"/tool/{token}", handle)
    runner = web.AppRunner(application)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.setblocking(False)
    await runner.setup()
    try:
        site = web.SockSite(runner, server_socket)
        await site.start()
        port = server_socket.getsockname()[1]
        local_url = f"http://127.0.0.1:{port}/tool/{token}"
        if runtime.is_local:
            yield _ToolRoute(url=runtime.host_url(local_url))
        else:
            async with PrimeTunnel().expose(port) as public_url:
                yield _ToolRoute(url=f"{public_url}/tool/{token}")
    finally:
        await runner.cleanup()


class OmpEnvConfig(vf.EnvConfig):
    """One trainable OMP agent and one fresh sealed grading runtime."""

    agent: vf.AgentConfig = vf.AgentConfig(
        runtime=vf.DockerConfig(
            image=RUNTIME_IMAGE,
            workdir=WORKDIR,
            allow=[],
        ),
        max_total_tokens=DEFAULT_TOTAL_TOKEN_LIMIT,
    )
    timeout: EnvTimeoutConfig = EnvTimeoutConfig(finalize=1800.0)


class OmpEnv(vf.Env[OmpEnvConfig]):
    """Run OMP once, then grade declared edits in a fresh runtime."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, OmpTask):
            raise TypeError(f"OmpEnv requires OmpTask, got {type(task).__name__}")
        await agents.agent.run(task)

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        if not isinstance(task, OmpTask):
            raise TypeError(f"OmpEnv requires OmpTask, got {type(task).__name__}")
        if len(episode.traces) != 1:
            raise RuntimeError("an OMP episode must contain exactly one trace")
        solution = episode.traces[0]
        if not solution.ok:
            solution.record_reward("tests", 0.0)
            return
        usage = solution.usage
        if usage is None:
            solution.record_error(vf.TaskError("provider token usage is missing"))
            solution.record_reward("tests", 0.0)
            return
        if usage.total_tokens > task.data.token_budget:
            solution.record_error(
                vf.TaskError(
                    "token budget exceeded: "
                    f"{usage.total_tokens} > {task.data.token_budget}"
                )
            )
            solution.record_reward("tests", 0.0)
            return
        if not solution.state.artifacts:
            raise RuntimeError("the OMP rollout produced no editable-file artifacts")

        runtime_config: RuntimeConfig = resolve_runtime_config(
            self.config.agent.runtime,
            task,
        )
        result = await _verify_artifacts(
            task,
            solution.state.artifacts,
            runtime_config,
        )
        solution.record_reward("tests", result.reward)
        solution.record_metrics(
            {
                "passed_cases": float(result.passed_cases),
                "total_cases": float(result.total_cases),
                "verifier_seconds": result.duration_seconds,
            }
        )
        solution.info["verifier"] = {
            "status": result.status,
            "suite_digest": result.suite_digest,
            "failures": [
                {"case_id": item.case_id, "reason": item.reason}
                for item in result.failures
            ],
        }


__all__ = ["OmpTaskset", "OmpHarness", "OmpEnv"]
