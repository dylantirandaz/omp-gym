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
from omp_coding.runtime import MAX_COMMAND_OUTPUT_BYTES, RuntimeFailure
from omp_coding.training import (
    ExportFailure,
    MetalReport,
    TrainingFailure,
    TrainingReport,
    export_traces,
    train_adapter,
)
from omp_coding.verifier import VerifierSuite, run_verifier_cases


def _trace(split: str, trace_id: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "id": trace_id,
        "ok": passed,
        "rewards": {"tests": {"score": 1.0 if passed else 0.0, "weight": 1.0}},
        "task": {"data": {"split": split, "task_id": f"omp-gym/{trace_id}"}},
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
                "message": {"role": "system", "content": "Use tools."},
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


def _training_row(user_text: str, assistant_text: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
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


class TrainingTests(unittest.TestCase):
    def test_training_filters_long_samples_and_accepts_no_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset"
            adapter = root / "adapter"
            data.mkdir()
            short_row = _training_row("Fix it.", "Done.")
            long_row = _training_row("x" * 300, "Done.")
            (data / "train.jsonl").write_text(
                json.dumps(short_row) + "\n" + json.dumps(long_row) + "\n"
            )
            (data / "valid.jsonl").write_text(json.dumps(short_row) + "\n")

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertNotIn("--test", command)
                staged_adapter = Path(command[command.index("--adapter-path") + 1])
                staged_adapter.mkdir()
                (staged_adapter / "adapters.safetensors").write_bytes(b"weights")
                (staged_adapter / "adapter_config.json").write_text("{}")
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
                    "omp_coding.training.metal_preflight",
                    return_value=_metal_report(),
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
                    max_sequence_length=256,
                    number_of_layers=1,
                    learning_rate=1e-5,
                )
            self.assertTrue(adapter.is_dir())
            installed_config = json.loads((adapter / "adapter_config.json").read_text())
            self.assertEqual(installed_config["adapter_path"], str(adapter.resolve()))
            self.assertEqual(installed_config["data"], str(data.resolve()))

        self.assertIsInstance(result, TrainingReport)
        assert isinstance(result, TrainingReport)
        self.assertEqual(result.train_samples, 1)
        self.assertEqual(result.valid_samples, 1)
        self.assertEqual(result.dropped_samples, 1)

    def test_training_rejects_non_finite_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset"
            adapter = root / "adapter"
            data.mkdir()
            row = _training_row("Fix it.", "Done.")
            (data / "train.jsonl").write_text(json.dumps(row) + "\n")
            (data / "valid.jsonl").write_text(json.dumps(row) + "\n")

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                staged_adapter = Path(command[command.index("--adapter-path") + 1])
                staged_adapter.mkdir()
                (staged_adapter / "adapters.safetensors").write_bytes(b"weights")
                (staged_adapter / "adapter_config.json").write_text("{}")
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
                patch(
                    "omp_coding.training.metal_preflight",
                    return_value=_metal_report(),
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
                    max_sequence_length=256,
                    number_of_layers=1,
                    learning_rate=1e-5,
                )
            self.assertFalse(adapter.exists())

        self.assertIsInstance(result, TrainingFailure)
        assert isinstance(result, TrainingFailure)
        self.assertIn("non-finite", result.reason)


class TraceExportTests(unittest.TestCase):
    def test_export_preserves_sampled_turns_and_tool_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            record = {
                "traces": [
                    _trace("train", "train-one"),
                    _trace("validation", "valid-one"),
                    _trace("train", "failed-one", passed=False),
                ]
            }
            traces.write_text(json.dumps(record) + "\n")
            output = root / "dataset"

            report = export_traces([traces], output)

            self.assertNotIsInstance(report, ExportFailure)
            assert not isinstance(report, ExportFailure)
            self.assertEqual(report.successful_traces, 2)
            self.assertEqual(report.train_samples, 2)
            self.assertEqual(report.valid_samples, 2)
            self.assertFalse((output / "test.jsonl").exists())
            train_rows = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text().splitlines()
            ]
            first_call = train_rows[0]["messages"][-1]["tool_calls"][0]
            self.assertEqual(first_call["id"], "call-1")
            self.assertEqual(first_call["function"]["arguments"], {"path": "app.py"})
            final_roles = [message["role"] for message in train_rows[1]["messages"]]
            self.assertEqual(
                final_roles, ["system", "user", "assistant", "tool", "assistant"]
            )

    def test_export_rejects_dataset_without_validation_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            traces.write_text(
                json.dumps({"traces": [_trace("train", "train-one")]}) + "\n"
            )

            result = export_traces([traces], root / "dataset")

        self.assertIsInstance(result, ExportFailure)
        assert isinstance(result, ExportFailure)
        self.assertIn("validation", result.reason)


if __name__ == "__main__":
    unittest.main()
