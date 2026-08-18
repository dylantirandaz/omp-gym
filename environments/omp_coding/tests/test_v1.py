from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import unittest
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from omp_coding import OmpTaskset, OmpTasksetConfig, _capture_process
from omp_coding.evaluation import _fuse_checkpoint
from omp_coding.metrics import (
    MetricsFailure,
    TraceMetricsReport,
    compare_metrics,
    measure_traces,
)
from omp_coding.protocol import (
    PROBES,
    ProtocolContext,
    ProtocolFailure,
    ProtocolReport,
    load_protocol_context,
    protocol_gate_failure,
    run_protocol_gate,
)
from omp_coding.runtime import MAX_COMMAND_OUTPUT_BYTES, RuntimeFailure
from omp_coding.training import (
    REQUIRED_ACTION_KINDS,
    DatasetManifest,
    ExportFailure,
    MetalReport,
    TrainingFailure,
    TrainingReport,
    export_traces,
    load_dataset_manifest,
    train_adapter,
)
from omp_coding.verifier import VerifierSuite, run_verifier_cases


def _trace(split: str, trace_id: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "id": trace_id,
        "ok": passed,
        "stop_condition": "agent_completed",
        "rewards": {"tests": {"score": 1.0 if passed else 0.0, "weight": 1.0}},
        "task": {
            "data": {
                "split": split,
                "task_id": f"omp-gym/{trace_id}",
                "task_revision": 1,
                "task_digest": f"digest-{trace_id}",
                "prompt": "Fix it.",
                "system_prompt": "Use tools. §",
            }
        },
        "tools": [
            {
                "name": "sandbox_read",
                "description": "Read one file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
        ],
        "nodes": [
            {
                "message": {"role": "system", "content": "Use tools. §"},
                "sampled": False,
            },
            {
                "parent": 0,
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix it."}],
                },
                "sampled": False,
            },
            {
                "parent": 1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "sandbox_read",
                            "arguments": '{"path":"app.py"}',
                        }
                    ],
                },
                "sampled": True,
            },
            {
                "parent": 2,
                "message": {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "1:pass",
                },
                "sampled": False,
            },
            {
                "parent": 3,
                "message": {"role": "assistant", "content": "Done."},
                "sampled": True,
            },
        ],
        "calls": [
            {
                "finish_reason": "tool_calls",
                "sampling": {"temperature": 0.0, "max_tokens": 1024},
            },
            {
                "finish_reason": "stop",
                "sampling": {"temperature": 0.0, "max_tokens": 1024},
            },
        ],
        "metrics": {"passed_cases": 1 if passed else 0, "total_cases": 1},
    }


class _ChunkStream:
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        delay_seconds: float = 0.0,
        hold_open: bool = False,
    ) -> None:
        self._chunks = iter(chunks)
        self._delay_seconds = delay_seconds
        self._hold_open = asyncio.Event() if hold_open else None

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        try:
            return next(self._chunks)
        except StopIteration as error:
            if self._hold_open is not None:
                await self._hold_open.wait()
            raise StopAsyncIteration from error


class _FakeProcess:
    def __init__(
        self,
        stdout: AsyncIterator[bytes],
        stderr: AsyncIterator[bytes],
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.killed = False

    async def wait(self) -> int:
        return 0

    async def kill(self) -> None:
        self.killed = True


class _FakeRuntime:
    supports_live_processes = True

    def __init__(self, process: _FakeProcess) -> None:
        self._process = process

    async def open_process(
        self,
        command: list[str],
        environment: Mapping[str, str],
    ) -> _FakeProcess:
        return self._process


class ProcessCaptureTests(unittest.TestCase):
    def test_output_limit_is_checked_after_process_exit(self) -> None:
        process = _FakeProcess(
            _ChunkStream((b'{"status":"ok"}\n',)),
            _ChunkStream(
                (b"x" * (MAX_COMMAND_OUTPUT_BYTES + 1),),
                delay_seconds=0.01,
            ),
        )

        result = asyncio.run(
            _capture_process(_FakeRuntime(process), ["candidate"], 1.0)
        )

        self.assertIsInstance(result, RuntimeFailure)
        assert isinstance(result, RuntimeFailure)
        self.assertEqual(result.kind, "output_limit")

    def test_descendant_pipe_is_reported_without_hanging(self) -> None:
        process = _FakeProcess(
            _ChunkStream((b'{"status":"ok"}\n',), hold_open=True),
            _ChunkStream(()),
        )

        result = asyncio.run(
            _capture_process(_FakeRuntime(process), ["candidate"], 1.0)
        )

        self.assertIsInstance(result, RuntimeFailure)
        assert isinstance(result, RuntimeFailure)
        self.assertEqual(result.kind, "process_leak")


class TasksetTests(unittest.TestCase):
    def test_all_versioned_tasks_load_through_v1_tasksets(self) -> None:
        counts = {
            split: len(list(OmpTaskset(OmpTasksetConfig(id="omp-coding", split=split))))
            for split in ("train", "validation", "holdout")
        }

        self.assertEqual(counts, {"train": 10, "validation": 4, "holdout": 4})

    def test_generated_train_tasks_are_deterministic(self) -> None:
        config = OmpTasksetConfig(
            id="omp-coding",
            generated_tasks=6,
            generation_seed=17,
        )
        first = [
            task.data
            for task in OmpTaskset(config)
            if "/generated/" in task.data.task_id
        ]
        second = [
            task.data
            for task in OmpTaskset(config)
            if "/generated/" in task.data.task_id
        ]
        self.assertEqual(len(first), 6)
        self.assertEqual(
            [task.task_digest for task in first],
            [task.task_digest for task in second],
        )
        self.assertEqual(len({task.family for task in first}), 6)


class VerifierTests(unittest.TestCase):
    def test_async_verifier_uses_structured_candidate_result(self) -> None:
        suite = VerifierSuite(
            runtime="python",
            cases=(
                {
                    "id": "exact-value",
                    "operations": [],
                    "observe": ["value"],
                    "expected": {
                        "status": "ok",
                        "values": {"value": {"kind": "int", "value": "7"}},
                    },
                },
            ),
            digest="suite-digest",
        )

        async def invoke(
            runtime: str,
            request: dict[str, object],
            timeout_seconds: float,
        ) -> dict[str, object] | RuntimeFailure:
            self.assertEqual(runtime, "python")
            self.assertGreater(timeout_seconds, 0)
            self.assertEqual(request["schema_version"], 1)
            return {
                "schema_version": 1,
                "status": "ok",
                "values": {"value": {"kind": "int", "value": "7"}},
            }

        result = asyncio.run(run_verifier_cases(suite, invoke, timeout_seconds=1.0))

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.passed_cases, 1)


class _FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None,
        add_generation_prompt: bool = False,
        return_dict: Literal[False] = False,
    ) -> list[int]:
        content_size = 0
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("message content must be text")
            content_size += len(content)
        extra_tokens = 2 if add_generation_prompt else 4
        return [1] * (content_size + extra_tokens)


def _training_row(
    trace_id: str, user_text: str, assistant_text: str
) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "target_kinds": sorted(REQUIRED_ACTION_KINDS),
        "tools": [],
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def _metal_report() -> MetalReport:
    return MetalReport(
        backend="metal",
        logical_device="gpu:0",
        device_name="Apple M3",
        architecture="arm64",
        memory_bytes=1,
        mlx_version="0.32.0",
        dtype="float32",
        check_value=8.0,
    )


def _dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        schema_version=2,
        dataset_sha256="dataset",
        train_trajectories=1,
        valid_trajectories=1,
        action_counts=tuple((kind, 1) for kind in sorted(REQUIRED_ACTION_KINDS)),
        samples_per_trace_limit=1,
    )


def _protocol_context() -> ProtocolContext:
    tools = tuple(
        {
            "type": "function",
            "function": {
                "name": probe.expected_tool,
                "description": f"Test {probe.expected_tool}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string"}
                        for name, _value in probe.expected_arguments
                    },
                    "required": [name for name, _value in probe.expected_arguments],
                    "additionalProperties": False,
                },
            },
        }
        for probe in PROBES
    )
    return ProtocolContext(
        system_prompt="Use tools.",
        tools=tools,
        sha256="context",
    )


def _protocol_report() -> ProtocolReport:
    return ProtocolReport(
        model="local-model",
        parser="native",
        context_sha256="context",
        probes=(),
        valid_call_rate=1.0,
        parsed_call_rate=1.0,
        invalid_tool_rate=0.0,
        end_token_rate=1.0,
        loop_rate=0.0,
    )


def _metrics(
    *,
    reward: float,
    parsed_calls: int = 1,
    parsed_tool_call_rate: float = 0.75,
) -> TraceMetricsReport:
    return TraceMetricsReport(
        schema_version=1,
        parser="native",
        trace_files=1,
        traces=4,
        sampled_turns=16,
        parsed_turns=12,
        parsed_calls=parsed_calls,
        tool_call_attempts=max(parsed_calls, 1),
        invalid_calls=0,
        unavailable_tool_calls=0,
        repeated_call_turns=0,
        completed_model_calls=16,
        model_calls=16,
        length_limited_calls=0,
        passed_cases=int(reward * 40),
        total_cases=40,
        sealed_validation_reward=reward,
        parsed_tool_call_rate=parsed_tool_call_rate,
        invalid_tool_rate=0.0,
        end_token_rate=1.0,
        loop_rate=0.0,
        comparison_sha256="same-settings",
    )


class ProtocolTests(unittest.TestCase):
    def test_protocol_context_removes_variable_omp_metadata(self) -> None:
        tools = list(_protocol_context().tools)

        def row(model: str, date: str) -> dict[str, object]:
            system_prompt = (
                "<workstation>\n"
                "- OS: linux\n"
                f"- Model: {model}\n"
                "</workstation>\n"
                f"Today: {date}; current working directory: '/workspace'.\n"
                "\nYou work in an isolated Linux workspace at /workspace.\n"
                "Task-specific context.\n"
            )
            return {
                "messages": [{"role": "system", "content": system_prompt}],
                "tools": tools,
            }

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "train.jsonl").write_text(
                json.dumps(row("omp-coding/openai/gpt-4.1-mini", "2026-08-17")) + "\n"
            )
            (data / "valid.jsonl").write_text(
                json.dumps(row("omp-coding/default_model", "2026-08-18")) + "\n"
            )
            context = load_protocol_context(data)

        self.assertIsInstance(context, ProtocolContext)
        assert isinstance(context, ProtocolContext)
        self.assertNotIn("gpt-4.1-mini", context.system_prompt)
        self.assertNotIn("2026-08-17", context.system_prompt)
        self.assertIn(
            "This request checks the OMP tool protocol", context.system_prompt
        )

    def test_gate_accepts_only_native_openai_tool_calls(self) -> None:
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": f"call-{probe.name}",
                                    "function": {
                                        "name": probe.expected_tool,
                                        "arguments": json.dumps(
                                            dict(probe.expected_arguments)
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"completion_tokens": 8},
            }
            for probe in PROBES
        ]
        with patch("omp_coding.protocol._post_json", side_effect=responses):
            report = run_protocol_gate(
                base_url="http://127.0.0.1:1/v1",
                api_key="test",
                model="model",
                parser="native",
                context=_protocol_context(),
            )
        self.assertIsInstance(report, ProtocolReport)
        assert isinstance(report, ProtocolReport)
        self.assertIsNone(protocol_gate_failure(report))

    def test_gate_accepts_schema_valid_optional_arguments(self) -> None:
        context = _protocol_context()
        tools = json.loads(json.dumps(context.tools))
        read_function = tools[0]["function"]
        read_parameters = read_function["parameters"]
        read_parameters["properties"]["limit"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": 4000,
        }
        responses = []
        for probe in PROBES:
            arguments = dict(probe.expected_arguments)
            if probe.name == "read":
                arguments["limit"] = 4000
            responses.append(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": f"call-{probe.name}",
                                        "function": {
                                            "name": probe.expected_tool,
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"completion_tokens": 8},
                }
            )
        optional_context = ProtocolContext(
            system_prompt=context.system_prompt,
            tools=tuple(tools),
            sha256=context.sha256,
        )

        with patch("omp_coding.protocol._post_json", side_effect=responses):
            report = run_protocol_gate(
                base_url="http://127.0.0.1:1/v1",
                api_key="test",
                model="model",
                parser="native",
                context=optional_context,
            )

        self.assertIsInstance(report, ProtocolReport)
        assert isinstance(report, ProtocolReport)
        self.assertIsNone(protocol_gate_failure(report))

        text_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"name":"sandbox_read","arguments":{}}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 8},
        }
        with patch(
            "omp_coding.protocol._post_json",
            side_effect=[text_response for _probe in PROBES],
        ):
            text_report = run_protocol_gate(
                base_url="http://127.0.0.1:1/v1",
                api_key="test",
                model="model",
                parser="native",
                context=_protocol_context(),
            )
        self.assertIsInstance(text_report, ProtocolReport)
        assert isinstance(text_report, ProtocolReport)
        self.assertEqual(text_report.parsed_call_rate, 0.0)
        self.assertIsNotNone(protocol_gate_failure(text_report))

        malformed_responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": f"call-{probe.name}",
                                    "function": {
                                        "name": probe.expected_tool,
                                        "arguments": "{",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"completion_tokens": 8},
            }
            for probe in PROBES
        ]
        with patch(
            "omp_coding.protocol._post_json",
            side_effect=malformed_responses,
        ):
            malformed_report = run_protocol_gate(
                base_url="http://127.0.0.1:1/v1",
                api_key="test",
                model="model",
                parser="native",
                context=_protocol_context(),
            )
        self.assertIsInstance(malformed_report, ProtocolReport)
        assert isinstance(malformed_report, ProtocolReport)
        self.assertEqual(malformed_report.parsed_call_rate, 0.0)
        self.assertEqual(malformed_report.invalid_tool_rate, 1.0)
        self.assertIsNotNone(protocol_gate_failure(malformed_report))

    def test_gate_rejects_plain_http_remote_endpoint(self) -> None:
        report = run_protocol_gate(
            base_url="http://example.com/v1",
            api_key="secret",
            model="model",
            parser="native",
            context=_protocol_context(),
        )

        self.assertIsInstance(report, ProtocolFailure)
        assert isinstance(report, ProtocolFailure)
        self.assertIn("HTTPS", report.reason)


class TrainingTests(unittest.TestCase):
    def test_training_filters_long_samples_and_writes_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset"
            adapter = root / "adapter"
            data.mkdir()
            (data / "manifest.json").write_text("{}")
            short_row = _training_row("train-short", "Fix it.", "Done.")
            long_row = _training_row("train-long", "x" * 9000, "Done.")
            valid_row = _training_row("valid-short", "Fix it.", "Done.")
            (data / "train.jsonl").write_text(
                json.dumps(short_row) + "\n" + json.dumps(long_row) + "\n"
            )
            (data / "valid.jsonl").write_text(json.dumps(valid_row) + "\n")

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertNotIn("--test", command)
                self.assertEqual(command[1:3], ["-m", "omp_coding.masked_lora"])
                expected_values = {
                    "--model": "local-model",
                    "--num-layers": "1",
                    "--iters": "2",
                    "--learning-rate": "1e-05",
                    "--checkpoint-interval": "1",
                    "--max-seq-length": "8192",
                    "--seed": "0",
                }
                for flag, expected in expected_values.items():
                    with self.subTest(flag=flag):
                        self.assertEqual(
                            command[command.index(flag) + 1],
                            expected,
                        )
                staged_adapter = Path(command[command.index("--adapter-path") + 1])
                staged_adapter.mkdir()
                (staged_adapter / "adapters.safetensors").write_bytes(b"weights")
                (staged_adapter / "0000001_adapters.safetensors").write_bytes(
                    b"checkpoint"
                )
                (staged_adapter / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "fine_tune_type": "lora",
                            "num_layers": 1,
                            "lora_parameters": {"rank": 8},
                            "adapter_path": "/private/adapter",
                            "data": "/private/dataset",
                        }
                    )
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "Iter 1: Val loss 2.500\n"
                        "Iter 1: Train loss 2.000\n"
                        "Iter 2: Train loss 1.000\n"
                    ),
                )

            with (
                patch(
                    "omp_coding.training.MINIMUM_TRAIN_TRAJECTORIES",
                    1,
                ),
                patch(
                    "omp_coding.training.MINIMUM_VALIDATION_TRAJECTORIES",
                    1,
                ),
                patch(
                    "omp_coding.training.load_dataset_manifest",
                    return_value=_dataset_manifest(),
                ),
                patch(
                    "omp_coding.training.load_protocol_context",
                    return_value=_protocol_context(),
                ),
                patch(
                    "omp_coding.training.metal_preflight",
                    return_value=_metal_report(),
                ),
                patch(
                    "omp_coding.training.run_local_protocol_gate",
                    return_value=_protocol_report(),
                ),
                patch(
                    "omp_coding.training._load_training_tokenizer",
                    return_value=_FakeTokenizer(),
                ),
                patch("omp_coding.training.subprocess.run", side_effect=run),
            ):
                result = train_adapter(
                    data_dir=data,
                    model="local-model",
                    adapter_dir=adapter,
                    iterations=2,
                    checkpoint_interval=1,
                    max_sequence_length=8192,
                    number_of_layers=1,
                    learning_rate=1e-5,
                )
            self.assertTrue(adapter.is_dir())
            installed_config = json.loads((adapter / "adapter_config.json").read_text())
            self.assertNotIn("adapter_path", installed_config)
            self.assertNotIn("data", installed_config)
            self.assertEqual(installed_config["lora_parameters"], {"rank": 8})

        self.assertIsInstance(result, TrainingReport)
        assert isinstance(result, TrainingReport)
        self.assertEqual(result.train_samples, 1)
        self.assertEqual(result.valid_samples, 1)
        self.assertEqual(result.dropped_samples, 1)
        self.assertEqual(
            result.checkpoint_files,
            ("0000001_adapters.safetensors",),
        )
        self.assertEqual(result.protocol.valid_call_rate, 1.0)

    def test_training_rejects_non_finite_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset"
            adapter = root / "adapter"
            data.mkdir()
            (data / "manifest.json").write_text("{}")
            train_row = _training_row("train", "Fix it.", "Done.")
            valid_row = _training_row("valid", "Fix it.", "Done.")
            (data / "train.jsonl").write_text(json.dumps(train_row) + "\n")
            (data / "valid.jsonl").write_text(json.dumps(valid_row) + "\n")

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                staged_adapter = Path(command[command.index("--adapter-path") + 1])
                staged_adapter.mkdir()
                (staged_adapter / "adapters.safetensors").write_bytes(b"weights")
                (staged_adapter / "0000001_adapters.safetensors").write_bytes(
                    b"checkpoint"
                )
                (staged_adapter / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "fine_tune_type": "lora",
                            "num_layers": 1,
                            "lora_parameters": {"rank": 8},
                        }
                    )
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "Iter 1: Train loss 2.000\n"
                        "Iter 1: Val loss nan\n"
                        "Iter 2: Train loss 1.000\n"
                    ),
                )

            with (
                patch("omp_coding.training.MINIMUM_TRAIN_TRAJECTORIES", 1),
                patch("omp_coding.training.MINIMUM_VALIDATION_TRAJECTORIES", 1),
                patch(
                    "omp_coding.training.load_dataset_manifest",
                    return_value=_dataset_manifest(),
                ),
                patch(
                    "omp_coding.training.load_protocol_context",
                    return_value=_protocol_context(),
                ),
                patch(
                    "omp_coding.training.metal_preflight",
                    return_value=_metal_report(),
                ),
                patch(
                    "omp_coding.training.run_local_protocol_gate",
                    return_value=_protocol_report(),
                ),
                patch(
                    "omp_coding.training._load_training_tokenizer",
                    return_value=_FakeTokenizer(),
                ),
                patch("omp_coding.training.subprocess.run", side_effect=run),
            ):
                result = train_adapter(
                    data_dir=data,
                    model="local-model",
                    adapter_dir=adapter,
                    iterations=2,
                    checkpoint_interval=1,
                    max_sequence_length=8192,
                    number_of_layers=1,
                    learning_rate=1e-5,
                )
            self.assertFalse(adapter.exists())

        self.assertIsInstance(result, TrainingFailure)
        assert isinstance(result, TrainingFailure)
        self.assertIn("non-finite", result.reason)

    def test_training_rejects_zero_protocol_base_model(self) -> None:
        zero_protocol = ProtocolReport(
            model="local-model",
            parser="native",
            context_sha256="context",
            probes=(),
            valid_call_rate=0.0,
            parsed_call_rate=0.0,
            invalid_tool_rate=0.0,
            end_token_rate=1.0,
            loop_rate=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset"
            data.mkdir()
            (data / "manifest.json").write_text("{}")
            row = _training_row("trace", "Fix it.", "Done.")
            (data / "train.jsonl").write_text(json.dumps(row) + "\n")
            (data / "valid.jsonl").write_text(json.dumps(row) + "\n")
            with (
                patch("omp_coding.training.MINIMUM_TRAIN_TRAJECTORIES", 1),
                patch("omp_coding.training.MINIMUM_VALIDATION_TRAJECTORIES", 1),
                patch(
                    "omp_coding.training.load_dataset_manifest",
                    return_value=_dataset_manifest(),
                ),
                patch(
                    "omp_coding.training.load_protocol_context",
                    return_value=_protocol_context(),
                ),
                patch(
                    "omp_coding.training.metal_preflight",
                    return_value=_metal_report(),
                ),
                patch(
                    "omp_coding.training.run_local_protocol_gate",
                    return_value=zero_protocol,
                ),
                patch("omp_coding.training.subprocess.run") as training_run,
            ):
                result = train_adapter(
                    data_dir=data,
                    model="local-model",
                    adapter_dir=root / "adapter",
                    iterations=2,
                    checkpoint_interval=1,
                    max_sequence_length=8192,
                    number_of_layers=1,
                    learning_rate=1e-5,
                )
            training_run.assert_not_called()
        self.assertIsInstance(result, TrainingFailure)
        assert isinstance(result, TrainingFailure)
        self.assertIn("base model rejected", result.reason)


class MetricsTests(unittest.TestCase):
    def test_comparison_requires_reward_gain_and_parsed_calls(self) -> None:
        baseline = _metrics(reward=0.25)
        improved = compare_metrics(baseline, _metrics(reward=0.5))
        self.assertNotIsInstance(improved, MetricsFailure)
        self.assertEqual(improved.status, "improved")

        no_gain = compare_metrics(baseline, _metrics(reward=0.25))
        self.assertNotIsInstance(no_gain, MetricsFailure)
        self.assertEqual(no_gain.status, "rejected")
        self.assertIn("reward did not increase", no_gain.reasons[0])

        zero_calls = compare_metrics(
            baseline,
            _metrics(reward=0.5, parsed_calls=0),
        )
        self.assertNotIsInstance(zero_calls, MetricsFailure)
        self.assertEqual(zero_calls.status, "rejected")
        self.assertIn("candidate produced zero parsed tool calls", zero_calls.reasons)

    def test_trace_rates_ignore_final_turn_and_detect_consecutive_loop(self) -> None:
        trace = _trace("validation", "metric-trace")
        nodes = trace["nodes"]
        calls = trace["calls"]
        assert isinstance(nodes, list)
        assert isinstance(calls, list)
        nodes.insert(
            4,
            {
                "parent": 3,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "name": "sandbox_read",
                            "arguments": '{"path":"app.py"}',
                        }
                    ],
                },
                "sampled": True,
            },
        )
        calls.insert(
            1,
            {
                "finish_reason": "tool_calls",
                "sampling": {"temperature": 0.0, "max_tokens": 1024},
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            trace_file = Path(temporary) / "traces.jsonl"
            trace_file.write_text(json.dumps({"traces": [trace]}) + "\n")
            report = measure_traces([trace_file], parser="native")

        self.assertNotIsInstance(report, MetricsFailure)
        assert isinstance(report, TraceMetricsReport)
        self.assertEqual(report.parsed_tool_call_rate, 1.0)
        self.assertEqual(report.end_token_rate, 1.0)
        self.assertEqual(report.loop_rate, 0.5)


class TraceExportTests(unittest.TestCase):
    def test_export_preserves_sampled_turns_and_tool_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            incomplete = _trace("train", "incomplete-one")
            incomplete["stop_condition"] = "task_token_budget_reached"
            record = {
                "traces": [
                    _trace("train", "train-one"),
                    _trace("validation", "valid-one"),
                    _trace("train", "failed-one", passed=False),
                    incomplete,
                ]
            }
            traces.write_text(json.dumps(record) + "\n")
            output = root / "dataset"

            report = export_traces(
                [traces],
                output,
                minimum_train_trajectories=1,
                minimum_validation_trajectories=1,
                required_action_kinds=frozenset({"read", "final"}),
            )

            self.assertNotIsInstance(report, ExportFailure)
            assert not isinstance(report, ExportFailure)
            self.assertEqual(report.successful_traces, 2)
            self.assertEqual(report.train_samples, 1)
            self.assertEqual(report.valid_samples, 1)
            self.assertFalse((output / "test.jsonl").exists())
            self.assertTrue((output / "manifest.json").is_file())
            manifest = load_dataset_manifest(output)
            self.assertNotIsInstance(manifest, TrainingFailure)
            train_rows = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text().splitlines()
            ]
            first_call = train_rows[0]["messages"][2]["tool_calls"][0]
            self.assertEqual(first_call["id"], "call-1")
            self.assertEqual(first_call["function"]["arguments"], {"path": "app.py"})
            self.assertEqual(train_rows[0]["target_kinds"], ["final", "read"])
            final_roles = [message["role"] for message in train_rows[0]["messages"]]
            self.assertEqual(
                final_roles, ["system", "user", "assistant", "tool", "assistant"]
            )

    def test_export_keeps_repeated_action_turns_in_one_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_trace = _trace("train", "train-repeated")
            nodes = train_trace["nodes"]
            assert isinstance(nodes, list)
            nodes.insert(
                4,
                {
                    "parent": 3,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "name": "sandbox_read",
                                "arguments": '{"path":"app.py"}',
                            }
                        ],
                    },
                    "sampled": True,
                },
            )
            nodes.insert(
                5,
                {
                    "parent": 4,
                    "message": {
                        "role": "tool",
                        "tool_call_id": "call-2",
                        "content": "1:pass",
                    },
                    "sampled": False,
                },
            )
            final_node = nodes[6]
            assert isinstance(final_node, dict)
            final_node["parent"] = 5
            traces = root / "traces.jsonl"
            traces.write_text(
                json.dumps(
                    {
                        "traces": [
                            train_trace,
                            _trace("validation", "valid-one"),
                        ]
                    }
                )
                + "\n"
            )
            output = root / "dataset"

            report = export_traces(
                [traces],
                output,
                minimum_train_trajectories=1,
                minimum_validation_trajectories=1,
                required_action_kinds=frozenset({"read", "final"}),
            )

            self.assertNotIsInstance(report, ExportFailure)
            assert not isinstance(report, ExportFailure)
            self.assertEqual(report.train_trajectories, 1)
            self.assertEqual(report.train_samples, 1)
            train_row = json.loads((output / "train.jsonl").read_text())
            roles = [message["role"] for message in train_row["messages"]]
            self.assertEqual(
                roles,
                [
                    "system",
                    "user",
                    "assistant",
                    "tool",
                    "assistant",
                    "tool",
                    "assistant",
                ],
            )

    def test_export_rejects_holdout_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            traces.write_text(
                json.dumps({"traces": [_trace("holdout", "holdout-one")]}) + "\n"
            )

            result = export_traces(
                [traces],
                root / "dataset",
                minimum_train_trajectories=1,
                minimum_validation_trajectories=1,
                required_action_kinds=frozenset({"read", "final"}),
            )

        self.assertIsInstance(result, ExportFailure)
        assert isinstance(result, ExportFailure)
        self.assertIn("holdout", result.reason)

    def test_export_rejects_oversized_trace_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            traces.write_text("12345")
            with patch("omp_coding.training.MAX_TRACE_FILE_BYTES", 4):
                result = export_traces(
                    [traces],
                    root / "dataset",
                    minimum_train_trajectories=1,
                    minimum_validation_trajectories=1,
                    required_action_kinds=frozenset({"read", "final"}),
                )

        self.assertIsInstance(result, ExportFailure)
        assert isinstance(result, ExportFailure)
        self.assertIn("size limit", result.reason)

    def test_export_rejects_dataset_without_validation_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            traces.write_text(
                json.dumps({"traces": [_trace("train", "train-one")]}) + "\n"
            )

            result = export_traces(
                [traces],
                root / "dataset",
                minimum_train_trajectories=1,
                minimum_validation_trajectories=1,
                required_action_kinds=frozenset({"read", "final"}),
            )

        self.assertIsInstance(result, ExportFailure)
        assert isinstance(result, ExportFailure)
        self.assertIn("validation", result.reason)


class EvaluationTests(unittest.TestCase):
    def test_fusion_uses_resolved_local_model_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_source = root / "model"
            model_source.mkdir()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            weights = adapter / "checkpoint.safetensors"
            weights.write_bytes(b"checkpoint")
            output = root / "fused"
            staging = root / "staging"
            staging.mkdir()
            observed_command: list[str] = []

            def run_fusion(
                command: Sequence[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                observed_command.extend(command)
                output.mkdir()
                (output / "config.json").write_text("{}")
                (output / "model.safetensors").write_bytes(b"model")
                return subprocess.CompletedProcess(command, 0, "")

            with (
                patch(
                    "omp_coding.evaluation._resolve_model_source",
                    return_value=model_source.resolve(),
                ),
                patch("omp_coding.evaluation.subprocess.run", side_effect=run_fusion),
            ):
                result = _fuse_checkpoint(
                    model="remote/model",
                    adapter_dir=adapter,
                    weights=weights,
                    output_dir=output,
                    staging_dir=staging,
                )

        self.assertIsNone(result)
        model_flag = observed_command.index("--model")
        self.assertEqual(observed_command[model_flag + 1], str(model_source.resolve()))


if __name__ == "__main__":
    unittest.main()
