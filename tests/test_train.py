import io
import json
import math
import os
import struct
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from omp_gym.train import (
    TrainError,
    TrainReport,
    _finish_report,
    _record_layer_selection,
    _require_fresh_adapter,
    _resolve_model_revision,
    _stream_trainer,
    _trainer_environment,
    _validate_loss_curves,
    _validate_training_parameters,
)

_LORA_KEYS = ("model.layers.0.self_attn.q_proj.lora_a.weight",)


def _write_adapter(path: Path, keys: tuple[str, ...] = _LORA_KEYS) -> None:
    """Write a minimal but valid safetensors file."""
    header = {
        key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for key in keys
    }
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)


class _FakeProcess:
    """A Popen stand-in: canned stdout, instant exit, no signals.

    The pid is deliberately not a live process group; the killer is
    patched out in timeout tests and never runs otherwise.
    """

    pid = 2**22 + 4242

    def __init__(self, stdout, exit_code: int = 0) -> None:
        self.stdout = stdout
        self._exit_code = exit_code

    def wait(self) -> int:
        return self._exit_code

    def poll(self) -> int:
        return self._exit_code

    def kill(self) -> None:
        pass


def _fake_popen(lines, exit_code: int = 0, captured: dict | None = None):
    def popen(command, **kwargs):
        if captured is not None:
            captured["command"] = command
            captured.update(kwargs)
        return _FakeProcess(io.StringIO("".join(lines)), exit_code)

    return popen


class ValidateLossCurvesTests(unittest.TestCase):
    def test_accepts_an_improving_run(self) -> None:
        _validate_loss_curves([2.0, 1.5, 1.0], [2.0, 1.8, 1.6])

    def test_rejects_a_flat_train_loss(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([1.0, 1.0], [])
        self.assertIn("train loss did not go down", str(caught.exception))
        self.assertIn("1.0 -> 1.0", str(caught.exception))

    def test_rejects_a_rising_val_loss(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([2.0, 1.0], [1.2, 1.7])
        self.assertIn("val loss went up", str(caught.exception))
        self.assertIn("1.2", str(caught.exception))
        self.assertIn("1.7", str(caught.exception))

    def test_accepts_a_run_without_val_losses(self) -> None:
        _validate_loss_curves([2.0, 1.0], [])

    def test_flags_the_observed_memorization_run(self) -> None:
        # Measured run: train 2.058 -> 1.404 (-31.8%) while val
        # only moved 2.151 -> 2.001 (-7.0%).
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([2.058, 1.404], [2.151, 2.001])
        message = str(caught.exception)
        self.assertIn("memorization-shaped", message)
        self.assertIn("31.8%", message)
        self.assertIn("7.0%", message)

    def test_memorization_thresholds_are_parameters(self) -> None:
        # The same curves pass when the train-drop bar moves above
        # the observed 31.8% drop.
        _validate_loss_curves([2.058, 1.404], [2.151, 2.001], train_drop_pct=0.40)
        with self.assertRaises(TrainError):
            _validate_loss_curves([2.058, 1.404], [2.151, 2.001], train_drop_pct=0.15)

    def test_accepts_a_balanced_run(self) -> None:
        # Train -30% with val -25% is healthy generalization.
        _validate_loss_curves([2.0, 1.4], [2.0, 1.5])

    def test_accepts_a_small_train_drop_with_flat_val(self) -> None:
        # A 10% train drop is below the 15% bar; flat val passes.
        _validate_loss_curves([2.0, 1.8], [2.0, 2.0])

    def test_rejects_a_run_with_no_loss_reports(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([1.0], [])
        self.assertIn("no loss reports", str(caught.exception))


class ParameterValidationTests(unittest.TestCase):
    def test_valid_parameters_pass(self) -> None:
        _validate_training_parameters(2, 1, 2048, 1e-5, 60, 0.15, 0.25)

    def test_invalid_parameters_are_rejected(self) -> None:
        invalid = (
            (1, 1, 2048, 1e-5, 60, 0.15, 0.25),
            (2, 0, 2048, 1e-5, 60, 0.15, 0.25),
            (2, 1, 0, 1e-5, 60, 0.15, 0.25),
            (2, 1, 2048, math.nan, 60, 0.15, 0.25),
            (2, 1, 2048, 1e-5, 0, 0.15, 0.25),
            (2, 1, 2048, 1e-5, 60, -0.1, 0.25),
            (2, 1, 2048, 1e-5, 60, 0.15, math.inf),
        )
        for parameters in invalid:
            with self.assertRaises(TrainError):
                _validate_training_parameters(*parameters)


class StreamTrainerParsingTests(unittest.TestCase):
    def test_parses_scientific_notation_and_case_variations(self) -> None:
        with redirect_stdout(io.StringIO()):
            losses, val_losses = _stream_trainer(
                ["fake-trainer"],
                max_seconds=60,
                popen=_fake_popen(
                    [
                        "Iter 1: TRAIN LOSS: 2.0e0, Val Loss 2E0\n",
                        "Iter 2: train loss 1.0e-1, VAL LOSS: 1.5e0\n",
                    ]
                ),
            )
        self.assertEqual(losses, [2.0, 0.1])
        self.assertEqual(val_losses, [2.0, 1.5])

    def test_infinite_loss_is_rejected(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(TrainError) as caught:
                _stream_trainer(
                    ["fake-trainer"],
                    max_seconds=60,
                    popen=_fake_popen(
                        [
                            "Iter 1: Train loss inf\n",
                            "Iter 2: Train loss 1.0\n",
                        ]
                    ),
                )
        self.assertIn("infinite", str(caught.exception))

    def test_nan_is_detected_from_the_parsed_series(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(TrainError) as caught:
                _stream_trainer(
                    ["fake-trainer"],
                    max_seconds=60,
                    popen=_fake_popen(
                        [
                            "Iter 1: Train loss NaN\n",
                            "Iter 2: Train loss 1.0\n",
                        ]
                    ),
                )
        self.assertIn("NaN", str(caught.exception))

    def test_nan_in_a_val_loss_also_fails(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(TrainError) as caught:
                _stream_trainer(
                    ["fake-trainer"],
                    max_seconds=60,
                    popen=_fake_popen(
                        [
                            "Iter 1: Train loss 2.0, Val loss nan\n",
                            "Iter 2: Train loss 1.0, Val loss 1.5\n",
                        ]
                    ),
                )
        self.assertIn("NaN", str(caught.exception))

    def test_a_failing_exit_code_still_fails(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(TrainError) as caught:
                _stream_trainer(
                    ["fake-trainer"],
                    max_seconds=60,
                    popen=_fake_popen(["boom\n"], exit_code=3),
                )
        self.assertIn("exited with 3", str(caught.exception))


class TrainerTimeoutTests(unittest.TestCase):
    def test_deadline_kills_the_process_group(self) -> None:
        # The fake clock jumps past the deadline on the first loop
        # check, so the timeout path runs before any line is read.
        ticks = iter([0.0, 100.0])
        process = _FakeProcess(io.StringIO(""))
        with (
            redirect_stdout(io.StringIO()),
            mock.patch("omp_gym.train._terminate_process_group") as terminator,
            self.assertRaises(TrainError) as caught,
        ):
            _stream_trainer(
                ["fake-trainer"],
                max_seconds=3,
                popen=lambda command, **kwargs: process,
                clock=lambda: next(ticks),
            )
        terminator.assert_called_once_with(process)
        message = str(caught.exception)
        self.assertIn("exceeded", message)
        self.assertIn("killed", message)

    def test_deadline_kills_a_real_silent_trainer(self) -> None:
        # Host-only proof: a real python3 that prints nothing and
        # sleeps a minute must be killed at the one-second deadline.
        started = time.monotonic()
        with (
            redirect_stdout(io.StringIO()),
            self.assertRaises(TrainError),
        ):
            _stream_trainer(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                max_seconds=1,
            )
        self.assertLess(time.monotonic() - started, 15)


class TrainerEnvironmentTests(unittest.TestCase):
    def test_secret_names_are_dropped_but_trainer_needs_survive(self) -> None:
        environ = {
            "PATH": "/usr/bin",
            "HOME": "/home/x",
            "TMPDIR": tempfile.gettempdir(),
            "LANG": "en_US.UTF-8",
            "HF_HOME": "/hf",
            "HF_TOKEN": "hf-secret",
            "OPENROUTER_API_KEY": "sk-or-x",
            "AWS_SECRET_ACCESS_KEY": "aws",
            "OMP_GYM_SHIM_TOKEN": "tok",
            "EDITOR": "vim",
        }
        env = _trainer_environment(environ)
        for kept in ("PATH", "HOME", "TMPDIR", "LANG", "HF_HOME", "HF_TOKEN"):
            self.assertIn(kept, env)
        self.assertEqual(env["HF_TOKEN"], "hf-secret")
        for dropped in (
            "OPENROUTER_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "OMP_GYM_SHIM_TOKEN",
        ):
            self.assertNotIn(dropped, env)
        # Ordinary non-secret variables still pass through.
        self.assertEqual(env["EDITOR"], "vim")

    def test_stream_trainer_gets_the_env_and_its_own_group(self) -> None:
        captured: dict = {}
        env = {"PATH": "/usr/bin", "HF_TOKEN": "hf-secret"}
        with redirect_stdout(io.StringIO()):
            _stream_trainer(
                ["fake-trainer"],
                max_seconds=60,
                env=env,
                popen=_fake_popen(
                    ["Train loss 2.0\n", "Train loss 1.0\n"],
                    captured=captured,
                ),
            )
        self.assertIs(captured["env"], env)
        self.assertTrue(captured["start_new_session"])


class ModelRevisionTests(unittest.TestCase):
    def _hub(self, root: Path) -> Path:
        return root / "hub"

    def test_resolves_the_main_ref_when_the_snapshot_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = self._hub(Path(tmp))
            refs = hub / "models--org--name" / "refs"
            refs.mkdir(parents=True)
            (refs / "main").write_text("abc123\n")
            (hub / "models--org--name" / "snapshots" / "abc123").mkdir(parents=True)
            self.assertEqual(_resolve_model_revision("org/name", hub_dir=hub), "abc123")

    def test_a_ref_without_a_snapshot_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = self._hub(Path(tmp))
            refs = hub / "models--org--name" / "refs"
            refs.mkdir(parents=True)
            (refs / "main").write_text("ghost\n")
            self.assertIsNone(_resolve_model_revision("org/name", hub_dir=hub))

    def test_an_uncached_model_has_no_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                _resolve_model_revision("org/name", hub_dir=self._hub(Path(tmp)))
            )

    def test_a_local_path_has_no_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_resolve_model_revision(tmp))


class LayerMetadataTests(unittest.TestCase):
    def test_records_explicit_all_with_mlx_compatibility_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp)
            config_path = adapter_dir / "adapter_config.json"
            config_path.write_text(json.dumps({"fine_tune_type": "lora"}))

            _record_layer_selection(adapter_dir, "all")

            config = json.loads(config_path.read_text())
        self.assertEqual(config["layer_selection"], "all")
        self.assertEqual(config["num_layers"], 0)


class AdapterFreshnessTests(unittest.TestCase):
    def test_rejects_a_missing_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time())
        self.assertIn("was not written", str(caught.exception))

    def test_rejects_a_stale_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_file = Path(tmp) / "adapters.safetensors"
            adapter_file.write_bytes(b"weights")
            old = time.time() - 3600
            os.utime(adapter_file, (old, old))
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time())
        self.assertIn("stale artifact", str(caught.exception))

    def test_rejects_an_empty_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapters.safetensors").write_bytes(b"")
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time() - 60)
        self.assertIn("empty", str(caught.exception))

    def test_rejects_an_unparseable_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapters.safetensors").write_bytes(b"weights")
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time() - 60)
        self.assertIn("safetensors header", str(caught.exception))

    def test_rejects_an_adapter_without_lora_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_adapter(
                Path(tmp) / "adapters.safetensors",
                keys=("model.layers.0.self_attn.q_proj.weight",),
            )
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time() - 60)
        self.assertIn("no LoRA tensors", str(caught.exception))

    def test_accepts_a_fresh_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_adapter(Path(tmp) / "adapters.safetensors")
            stale = _require_fresh_adapter(Path(tmp), time.time() - 60)
            self.assertEqual(stale, [])

    def test_lists_stale_leftovers_but_not_fresh_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_adapter(root / "adapters.safetensors")
            old = time.time() - 3600
            for name in ("old-run.log", "adapter.bak"):
                leftover = root / name
                leftover.write_text("leftover\n")
                os.utime(leftover, (old, old))
            (root / "fresh-note.txt").write_text("new\n")
            stale = _require_fresh_adapter(root, time.time() - 60)
            self.assertEqual(stale, ["adapter.bak", "old-run.log"])


class FinishReportTests(unittest.TestCase):
    def _finish(self, adapter_dir: Path, started_at: float, **kwargs):
        return _finish_report(
            "test-model",
            Path("data"),
            2,
            adapter_dir,
            [2.0, 1.0],
            [],
            "test-gpu",
            started_at,
            **kwargs,
        )

    def test_rejects_a_stale_adapter_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_file = Path(tmp) / "adapters.safetensors"
            _write_adapter(adapter_file)
            old = time.time() - 3600
            os.utime(adapter_file, (old, old))
            with self.assertRaises(TrainError) as caught:
                self._finish(Path(tmp), time.time())
        self.assertIn("stale artifact", str(caught.exception))

    def test_accepts_a_fresh_adapter_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_adapter(Path(tmp) / "adapters.safetensors")
            report = self._finish(Path(tmp), time.time() - 60)
            self.assertEqual(report.last_train_loss, 1.0)
            report_file = Path(tmp) / "train_report.json"
            self.assertTrue(report_file.is_file())

    def test_report_records_revision_thresholds_and_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_adapter(root / "adapters.safetensors")
            leftover = root / "leftover.log"
            leftover.write_text("old\n")
            old = time.time() - 3600
            os.utime(leftover, (old, old))
            report = self._finish(
                root,
                time.time() - 60,
                model_revision="abc123",
                train_drop_pct=0.2,
                val_drop_ratio=0.3,
            )
            self.assertEqual(report.model_revision, "abc123")
            self.assertEqual(report.stale_adapter_files, ["leftover.log"])
            written = json.loads((root / "train_report.json").read_text())
            self.assertEqual(written["model_revision"], "abc123")
            self.assertEqual(written["stale_adapter_files"], ["leftover.log"])
            self.assertEqual(written["train_drop_pct"], 0.2)
            self.assertEqual(written["val_drop_ratio"], 0.3)


class LedgerConfigTests(unittest.TestCase):
    def test_train_ledger_entry_records_batch_config_and_revision(self) -> None:
        from omp_gym import cli

        report = TrainReport(
            model="m",
            data_dir="data",
            adapter_dir="adapters/x",
            iterations=3,
            first_train_loss=2.0,
            last_train_loss=1.0,
            first_val_loss=None,
            last_val_loss=None,
            device_name="test-gpu",
            model_revision="rev123",
        )
        with (
            mock.patch("omp_gym.train.run_training", return_value=report),
            mock.patch("omp_gym.cli.append_entry") as append,
        ):
            code = cli._cmd_train(
                data_dir=Path("data"),
                model="m",
                iterations=3,
                adapter_dir=Path("adapters/x"),
                batch_size=4,
                max_seq_length=1024,
                num_layers=8,
                learning_rate=1e-5,
                method="sft",
                resume_adapter=None,
                max_train_seconds=3600,
                train_drop_pct=0.2,
                val_drop_ratio=0.3,
                seed=7,
                dpo_beta=0.2,
                grad_clip=0.8,
            )
        self.assertEqual(code, 0)
        _, kwargs = append.call_args
        self.assertEqual(kwargs["kind"], "train")
        self.assertEqual(kwargs["config"]["batch_size"], 4)
        self.assertEqual(kwargs["config"]["max_seq_length"], 1024)
        self.assertEqual(kwargs["config"]["max_train_seconds"], 3600)
        self.assertEqual(kwargs["config"]["train_drop_pct"], 0.2)
        self.assertEqual(kwargs["config"]["val_drop_ratio"], 0.3)
        self.assertEqual(kwargs["metrics"]["model_revision"], "rev123")

    def test_dpo_cli_parameters_reach_the_native_trainer(self) -> None:
        from omp_gym import cli

        report = TrainReport(
            model="m",
            data_dir="data",
            adapter_dir="adapters/dpo",
            iterations=3,
            first_train_loss=2.0,
            last_train_loss=1.0,
            first_val_loss=None,
            last_val_loss=None,
            device_name="test-gpu",
        )
        with (
            mock.patch(
                "omp_gym.train.run_dpo_training", return_value=report
            ) as run_dpo,
            mock.patch("omp_gym.cli.append_entry"),
        ):
            code = cli._cmd_train(
                data_dir=Path("data"),
                model="m",
                iterations=3,
                adapter_dir=Path("adapters/dpo"),
                batch_size=4,
                max_seq_length=1536,
                num_layers=8,
                learning_rate=2e-5,
                method="dpo",
                resume_adapter=Path("adapters/sft.safetensors"),
                max_train_seconds=3600,
                train_drop_pct=0.2,
                val_drop_ratio=0.3,
                seed=7,
                dpo_beta=0.2,
                grad_clip=0.8,
            )
        self.assertEqual(code, 0)
        self.assertEqual(run_dpo.call_args.kwargs["batch_size"], 4)
        self.assertEqual(run_dpo.call_args.kwargs["max_seq_length"], 1536)
        self.assertEqual(run_dpo.call_args.kwargs["seed"], 7)
        self.assertEqual(run_dpo.call_args.kwargs["beta"], 0.2)
        self.assertEqual(run_dpo.call_args.kwargs["grad_clip"], 0.8)


if __name__ == "__main__":
    unittest.main()
