import io
import json
import math
import os
import socket
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from omp_gym import shim as shim_mod
from omp_gym.rl import (
    COLLAPSE_ABSOLUTE_FLOOR,
    COMPLETION_COLLAPSE_RATIO,
    AdapterTopology,
    RlError,
    _apply_grad_clip,
    _clip_scale,
    _completion_length_collapsed,
    _exceeds_seq_cap,
    _group_mean,
    _missing_lora_keys,
    _normalize_capture,
    _normalized_advantages,
    _policy_gradient_terms,
    _provider_lock,
    _read_adapter_topology,
    _register_provider,
    _require_finite_metric,
    _restore_provider_config,
    _run_evaluation,
    _select_reward,
    _snapshot_provider_config,
    _start_policy_server,
    _stop_process,
    _task_schedule,
    _validate_parameters,
    _window_captures,
    run_rl,
)
from omp_gym.runner import EpisodeFailure
from omp_gym.task import TaskSpec


class NormalizedAdvantagesTests(unittest.TestCase):
    def test_advantages_have_zero_mean(self) -> None:
        advantages = _normalized_advantages([1.0, 2.0, 3.0, 6.0])
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0)

    def test_advantages_have_unit_std(self) -> None:
        advantages = _normalized_advantages([0.0, 1.0])
        mean = sum(advantages) / len(advantages)
        variance = sum((value - mean) ** 2 for value in advantages) / len(advantages)
        self.assertAlmostEqual(variance**0.5, 1.0)

    def test_equal_rewards_give_zero_advantages(self) -> None:
        # The zero-std guard divides by 1.0 instead of 0.0.
        advantages = _normalized_advantages([0.5, 0.5, 0.5])
        self.assertEqual(advantages, [0.0, 0.0, 0.0])

    def test_higher_reward_gets_higher_advantage(self) -> None:
        low, high = _normalized_advantages([0.0, 1.0])
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_failures_count_as_zero_in_the_k_denominator(self) -> None:
        # A failed rollout stays in the group at reward 0.0, so the
        # baseline mean and variance use the requested group size K.
        advantages = _normalized_advantages([1.0, 0.0, 0.0])
        mean_reward = _group_mean([1.0, 0.0, 0.0])
        self.assertAlmostEqual(mean_reward, 1.0 / 3.0)
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0)
        # The zero-reward failures push the winner further from the
        # mean than a K=1 group would.
        self.assertGreater(advantages[0], 0.0)
        self.assertLess(advantages[1], 0.0)


class GroupMeanTests(unittest.TestCase):
    def test_zero_reward_failures_shift_the_mean(self) -> None:
        self.assertAlmostEqual(_group_mean([1.0, 0.0, 1.0, 0.0]), 0.5)

    def test_empty_group_is_zero(self) -> None:
        self.assertEqual(_group_mean([]), 0.0)


class CompletionLengthCollapseTests(unittest.TestCase):
    def test_ratio_is_the_documented_value(self) -> None:
        self.assertEqual(COMPLETION_COLLAPSE_RATIO, 0.3)

    def test_stops_when_length_falls_below_the_floor(self) -> None:
        self.assertTrue(_completion_length_collapsed(100.0, 25.0))

    def test_continues_above_the_floor(self) -> None:
        self.assertFalse(_completion_length_collapsed(100.0, 45.0))

    def test_the_floor_itself_does_not_stop(self) -> None:
        self.assertFalse(_completion_length_collapsed(100.0, 30.0))

    def test_first_iteration_below_absolute_floor_stops(self) -> None:
        self.assertTrue(
            _completion_length_collapsed(None, COLLAPSE_ABSOLUTE_FLOOR - 0.5)
        )

    def test_first_iteration_at_the_floor_survives(self) -> None:
        self.assertFalse(_completion_length_collapsed(None, COLLAPSE_ABSOLUTE_FLOOR))

    def test_zero_length_first_iteration_is_not_flagged(self) -> None:
        # A zero mean comes from division-by-empty guards, not from a
        # collapsed policy; the absolute floor needs a true length.
        self.assertFalse(_completion_length_collapsed(None, 0.0))

    def test_zero_reference_disables_the_ratio_check(self) -> None:
        self.assertFalse(_completion_length_collapsed(0.0, 0.0))

    def test_absolute_floor_is_documented(self) -> None:
        self.assertEqual(COLLAPSE_ABSOLUTE_FLOOR, 8.0)


class ParameterValidationTests(unittest.TestCase):
    def test_valid_parameters_pass(self) -> None:
        _validate_parameters(2, 1, 0.7, 0.0, 2048, 1.0, 0, 8810)

    def test_invalid_parameters_are_rejected(self) -> None:
        invalid = (
            (1, 1, 0.7, 0.0, 2048, 1.0, 0, 8810),
            (2, 0, 0.7, 0.0, 2048, 1.0, 0, 8810),
            (2, 1, math.nan, 0.0, 2048, 1.0, 0, 8810),
            (2, 1, 0.7, -0.1, 2048, 1.0, 0, 8810),
            (2, 1, 0.7, 0.0, 0, 1.0, 0, 8810),
            (2, 1, 0.7, 0.0, 2048, math.inf, 0, 8810),
            (2, 1, 0.7, 0.0, 2048, 1.0, -1, 8810),
            (2, 1, 0.7, 0.0, 2048, 1.0, 0, 70000),
        )
        for parameters in invalid:
            with self.assertRaises(RlError):
                _validate_parameters(*parameters)


class PolicyGradientTermsTests(unittest.TestCase):
    def test_every_captured_turn_is_summed(self) -> None:
        terms = _policy_gradient_terms([[-1.0, -2.0, -3.0]], [2.0])
        self.assertEqual(len(terms), 1)
        self.assertAlmostEqual(terms[0], 2.0 * (-6.0))

    def test_no_per_token_mean_normalization(self) -> None:
        summed = _policy_gradient_terms([[-1.0, -2.0]], [1.0])[0]
        self.assertAlmostEqual(summed, -3.0)
        # A per-token mean would halve the term; the objective sums.
        self.assertNotAlmostEqual(summed, -1.5)

    def test_one_term_per_rollout(self) -> None:
        terms = _policy_gradient_terms([[-1.0, -2.0], [-3.0]], [2.0, -0.5])
        self.assertEqual(len(terms), 2)
        self.assertAlmostEqual(terms[0], -6.0)
        self.assertAlmostEqual(terms[1], 1.5)


class GradientClipTests(unittest.TestCase):
    def test_below_cap_is_identity(self) -> None:
        self.assertEqual(_clip_scale(1.0, 2.0), 1.0)

    def test_above_cap_scales_to_the_cap(self) -> None:
        self.assertAlmostEqual(_clip_scale(10.0, 2.0), 0.2)

    def test_zero_norm_is_identity(self) -> None:
        self.assertEqual(_clip_scale(0.0, 2.0), 1.0)

    def test_nonpositive_cap_disables_clipping(self) -> None:
        self.assertEqual(_clip_scale(10.0, 0.0), 1.0)

    def test_apply_clip_returns_norm_and_scaled_tree(self) -> None:
        clipped, norm = _apply_grad_clip({"w": [3.0, 4.0]}, 2.5)
        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(clipped["w"][0], 1.5)
        self.assertAlmostEqual(clipped["w"][1], 2.0)

    def test_apply_clip_below_cap_leaves_values(self) -> None:
        clipped, norm = _apply_grad_clip({"w": [3.0, 4.0]}, 5.0)
        self.assertAlmostEqual(norm, 5.0)
        self.assertEqual(clipped["w"], [3.0, 4.0])


class FiniteMetricTests(unittest.TestCase):
    def test_nonfinite_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(RlError):
                _require_finite_metric(value, "loss")

    def test_finite_value_passes_through(self) -> None:
        self.assertEqual(_require_finite_metric(0.5, "loss"), 0.5)


class CaptureNormalizationTests(unittest.TestCase):
    def test_dict_passes_through(self) -> None:
        entry = {
            "messages": [{"role": "user", "content": "x"}],
            "text": "y",
            "request_id": "r-1",
            "finish_reason": "stop",
            "usage": {"total_tokens": 5},
            "model": "m",
        }
        self.assertIs(_normalize_capture(entry), entry)

    def test_non_dict_or_invalid_entry_is_dropped(self) -> None:
        self.assertIsNone(
            _normalize_capture(([{"role": "user", "content": "x"}], "legacy"))
        )
        self.assertIsNone(_normalize_capture("not a capture"))
        self.assertIsNone(_normalize_capture({"messages": [], "text": 3}))


class WindowCapturesTests(unittest.TestCase):
    def test_ordinal_slice_is_exact(self) -> None:
        captured = [
            {"messages": [], "text": "before"},
            {"messages": [], "text": "inside"},
            {"messages": [], "text": "after"},
        ]
        window = _window_captures(captured, 1, 2)
        self.assertEqual([entry["text"] for entry in window], ["inside"])

    def test_slice_boundary_math(self) -> None:
        # Two rollouts: entries 0-1 belong to the first, 2 to the
        # second. Ordinal slicing must never cross windows.
        captured = [
            {"messages": [], "text": "r1a"},
            {"messages": [], "text": "r1b"},
            {"messages": [], "text": "r2a"},
        ]
        first = _window_captures(captured, 0, 2)
        second = _window_captures(captured, 2, 3)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["text"], "r2a")

    def test_removed_tuple_shape_is_dropped(self) -> None:
        captured = [
            (["m"], "removed"),
            {"messages": ["m2"], "text": "current"},
            "garbage",
        ]
        window = _window_captures(captured, 0, 3)
        self.assertEqual(len(window), 1)
        self.assertEqual(window[0]["text"], "current")


class ImprovementRewardTests(unittest.TestCase):
    def _episode(self, **fields):
        return SimpleNamespace(**fields)

    def test_improvement_beats_partial_and_reward(self) -> None:
        episode = self._episode(reward_improvement=0.7, reward_partial=0.5, reward=0.3)
        self.assertAlmostEqual(_select_reward(episode), 0.7)

    def test_missing_improvement_falls_back_to_partial(self) -> None:
        episode = self._episode(reward_partial=0.4, reward=0.2)
        self.assertAlmostEqual(_select_reward(episode), 0.4)

    def test_none_improvement_falls_back_to_partial(self) -> None:
        episode = self._episode(reward_improvement=None, reward_partial=0.4, reward=0.2)
        self.assertAlmostEqual(_select_reward(episode), 0.4)

    def test_missing_improvement_and_partial_fallback_to_reward(self) -> None:
        episode = self._episode(reward=0.2)
        self.assertAlmostEqual(_select_reward(episode), 0.2)

    def test_all_missing_is_zero(self) -> None:
        self.assertEqual(_select_reward(self._episode()), 0.0)


class SequenceCapTests(unittest.TestCase):
    def test_within_cap_is_allowed(self) -> None:
        self.assertFalse(_exceeds_seq_cap([1, 2], [3, 4], 4))

    def test_over_cap_is_skipped(self) -> None:
        self.assertTrue(_exceeds_seq_cap([1, 2], [3, 4], 3))

    def test_exact_cap_is_allowed(self) -> None:
        self.assertFalse(_exceeds_seq_cap([1], [2, 3, 4], 4))


class AdapterTopologyTests(unittest.TestCase):
    def _adapter_dir(self, config) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        adapter = Path(temp.name)
        if config is not None:
            (adapter / "adapter_config.json").write_text(json.dumps(config))
        return adapter

    def test_derives_rank_scale_dropout_layers_from_config(self) -> None:
        adapter = self._adapter_dir(
            {
                "lora_parameters": {
                    "rank": 4,
                    "scale": 8.5,
                    "dropout": 0.1,
                },
                "num_layers": 2,
            }
        )
        topology = _read_adapter_topology(adapter)
        self.assertEqual(
            topology, AdapterTopology(rank=4, scale=8.5, dropout=0.1, layer_selection=2)
        )

    def test_explicit_all_layer_selection_is_preserved(self) -> None:
        adapter = self._adapter_dir(
            {
                "lora_parameters": {
                    "rank": 4,
                    "scale": 8.0,
                    "dropout": 0.0,
                },
                "num_layers": 0,
                "layer_selection": "all",
            }
        )
        topology = _read_adapter_topology(adapter)
        self.assertEqual(topology.layer_selection, "all")

    def test_missing_parameters_are_an_rl_error(self) -> None:
        adapter = self._adapter_dir({})
        with self.assertRaises(RlError) as caught:
            _read_adapter_topology(adapter)
        self.assertIn("lora_parameters", str(caught.exception))

    def test_missing_config_is_an_rl_error(self) -> None:
        with self.assertRaises(RlError):
            _read_adapter_topology(self._adapter_dir(None))

    def test_invalid_json_is_an_rl_error(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        adapter = Path(temp.name)
        (adapter / "adapter_config.json").write_text("{not json")
        with self.assertRaises(RlError):
            _read_adapter_topology(adapter)

    def test_missing_key_check_uses_converted_model_names(self) -> None:
        expected = [
            "model.layers.20.self_attn.q_proj.lora_a",
            "model.layers.20.self_attn.q_proj.lora_b",
        ]
        self.assertEqual(_missing_lora_keys(expected, expected), [])
        self.assertEqual(
            _missing_lora_keys([expected[0]], expected),
            [expected[1]],
        )
        self.assertEqual(_missing_lora_keys([], expected), sorted(expected))


class ProviderConfigTests(unittest.TestCase):
    def _tmp_models_yml(self, content: bytes | None) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "models.yml"
        if content is not None:
            path.write_bytes(content)
        return path

    def test_snapshot_then_restore_returns_original_bytes(self) -> None:
        path = self._tmp_models_yml(b"original-bytes")
        snapshot = _snapshot_provider_config(path)
        self.assertEqual(snapshot, b"original-bytes")
        path.write_bytes(b"changed-bytes")
        _restore_provider_config(path, snapshot)
        self.assertEqual(path.read_bytes(), b"original-bytes")

    def test_absent_snapshot_restore_removes_created_file(self) -> None:
        path = self._tmp_models_yml(None)
        self.assertIsNone(_snapshot_provider_config(path))
        path.write_bytes(b"created-by-run")
        _restore_provider_config(path, None)
        self.assertFalse(path.exists())

    def test_register_raises_on_foreign_provider_file(self) -> None:
        path = self._tmp_models_yml(b"foreign: true\n")
        with mock.patch("omp_gym.serve.ensure_provider", return_value=False):
            with self.assertRaises(RlError):
                _register_provider(path, 4040, "rl-x", "model")

    def test_register_accepts_owned_file_and_creates_lock(self) -> None:
        path = self._tmp_models_yml(None)
        with mock.patch("omp_gym.serve.ensure_provider", return_value=True):
            _register_provider(path, 4040, "rl-x", "model")
        self.assertTrue(path.with_name("models.yml.lock").exists())

    def test_lock_acquires_and_releases(self) -> None:
        path = self._tmp_models_yml(None)
        with _provider_lock(path):
            self.assertTrue(path.with_name("models.yml.lock").exists())


class TaskScheduleTests(unittest.TestCase):
    def test_same_seed_same_schedule(self) -> None:
        dirs = [Path(f"/task-{i}") for i in range(3)]
        first = _task_schedule(dirs, 8, 42)
        second = _task_schedule(dirs, 8, 42)
        self.assertEqual(first, second)

    def test_round_robin_covers_every_task_before_repeating(self) -> None:
        dirs = [Path(f"/task-{i}") for i in range(3)]
        schedule = _task_schedule(dirs, 7, 7)
        self.assertEqual(sorted(schedule[:3]), [0, 1, 2])
        for index in range(len(schedule)):
            self.assertEqual(schedule[index], schedule[index % len(dirs)])

    def test_single_task_always_index_zero(self) -> None:
        self.assertEqual(_task_schedule([Path("/only")], 4, 9), [0, 0, 0, 0])


class _FakeProc:
    """subprocess.Popen stand-in with controllable wait."""

    def __init__(self, fail_first_wait: bool = False) -> None:
        self.wait_calls = 0
        self.terminated = False
        self.killed = False
        self.fail_first_wait = fail_first_wait

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.fail_first_wait and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="server", timeout=timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


class StopProcessTests(unittest.TestCase):
    def test_clean_exit_avoids_kill(self) -> None:
        proc = _FakeProc()
        _stop_process(proc)
        self.assertTrue(proc.terminated)
        self.assertEqual(proc.wait_calls, 1)
        self.assertFalse(proc.killed)

    def test_timeout_escalates_to_kill_and_waits_again(self) -> None:
        proc = _FakeProc(fail_first_wait=True)
        _stop_process(proc)
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        self.assertEqual(proc.wait_calls, 2)


class _NullSock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self) -> None:
        pass


class PolicyServerEnvTests(unittest.TestCase):
    def test_sample_temperature_scoped_to_subprocess_and_shim(self) -> None:
        recorded = {}

        class FakeHttpd:
            def __init__(self, address, handler) -> None:
                recorded["address"] = address

            def serve_forever(self) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def server_close(self) -> None:
                pass

        class FakePopen:
            def __init__(self, args, env=None, stdout=None, stderr=None):
                recorded["args"] = args
                recorded["env"] = env

            def terminate(self) -> None:
                pass

            def wait(self, timeout=None):
                return 0

        def fake_make_handler(*args, **kwargs):
            recorded["handler_kwargs"] = kwargs
            return object

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        adapter = Path(temp.name)
        before = dict(os.environ)
        with (
            mock.patch.multiple("omp_gym.rl", ThreadingHTTPServer=FakeHttpd),
            mock.patch.object(subprocess, "Popen", FakePopen),
            mock.patch.object(
                socket,
                "create_connection",
                lambda address, timeout=1: _NullSock(),
            ),
            mock.patch.object(shim_mod, "make_handler", fake_make_handler),
        ):
            proc, httpd, chosen = _start_policy_server(
                "base-model",
                adapter,
                12345,
                capture=[],
                sample_temperature=0.25,
            )
        self.assertEqual(chosen, 12345)
        env = recorded["env"]
        self.assertIsNotNone(env)
        self.assertIsNot(env, os.environ)
        self.assertEqual(env["OMP_GYM_SAMPLE_TEMP"], "0.25")
        self.assertEqual(recorded["handler_kwargs"].get("sample_temp"), 0.25)
        self.assertEqual(dict(os.environ), before)
        # No temperature: the subprocess inherits the plain environment.
        with (
            mock.patch.multiple("omp_gym.rl", ThreadingHTTPServer=FakeHttpd),
            mock.patch.object(subprocess, "Popen", FakePopen),
            mock.patch.object(
                socket,
                "create_connection",
                lambda address, timeout=1: _NullSock(),
            ),
            mock.patch.object(shim_mod, "make_handler", fake_make_handler),
        ):
            _start_policy_server("base-model", adapter, 23456, capture=None)
        self.assertIsNone(recorded["env"])

    def test_real_shim_accepts_the_arbitrated_sample_temp(self) -> None:
        # Contract lock: rl.py passes sample_temp to make_handler; a
        # shim regression on that kwarg must fail here, not at train
        # time.
        handler = shim_mod.make_handler(1, None, capture=[], sample_temp=0.5)
        self.assertTrue(issubclass(handler, BaseHTTPRequestHandler))


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return [1, 2]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[3, 4])


class _FakeHttpd:
    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass


class EvaluationPassTests(unittest.TestCase):
    def test_only_verified_reward_counts_as_a_pass(self) -> None:
        outcomes = [
            SimpleNamespace(
                reward=0.0,
                test_exit_code=0,
                test_files_changed=False,
            ),
            SimpleNamespace(
                reward=1.0,
                test_exit_code=0,
                test_files_changed=False,
            ),
        ]
        with mock.patch.multiple(
            "omp_gym.rl",
            _start_policy_server=lambda *args, **kwargs: (
                _FakeProc(),
                _FakeHttpd(),
                4242,
            ),
            _register_provider=lambda *args, **kwargs: None,
            run_episode=mock.Mock(side_effect=outcomes),
            _stop_process=lambda process: None,
        ):
            result = _run_evaluation(
                base_model="base",
                out_adapter=Path("adapter"),
                port=4242,
                tasks=[SimpleNamespace()],
                eval_episodes=2,
                sample_temperature=0.7,
                models_yml=Path("models.yml"),
                provider_id="provider",
                runs_dir=Path("runs"),
                model_id="model",
            )
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["pass_rate"], 0.5)


class RunRlFakeTests(unittest.TestCase):
    """run_rl against fakes: no mlx model, no GPU, no real serving."""

    def _fixture(self) -> dict:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        workspace = root / "ws"
        workspace.mkdir()
        (workspace / "solution.py").write_text("x = 1\n")
        adapter_in = root / "adapter-in"
        adapter_in.mkdir()
        (adapter_in / "adapters.safetensors").touch()
        (adapter_in / "adapter_config.json").write_text(
            json.dumps(
                {
                    "lora_parameters": {
                        "rank": 4,
                        "scale": 8.0,
                        "dropout": 0.0,
                    },
                    "num_layers": 2,
                }
            )
        )
        models_yml = root / "home" / ".omp" / "agent" / "models.yml"
        models_yml.parent.mkdir(parents=True)
        models_yml.write_bytes(b"original-provider-bytes")
        return {
            "root": root,
            "workspace": workspace,
            "adapter_in": adapter_in,
            "out_adapter": root / "adapter-out",
            "runs_dir": root / "runs",
            "ledger": root / "ledger.jsonl",
            "models_yml": models_yml,
        }

    def _run(self, fixture, rollouts, **overrides):
        captures_by_rollout = [rollout.get("captures", []) for rollout in rollouts]
        outcomes = [rollout["outcome"] for rollout in rollouts]
        state = {
            "start_calls": [],
            "run_calls": [],
            "procs": [],
        }

        def fake_start(
            base_model, adapter, port, capture=None, sample_temperature=None
        ):
            state["start_calls"].append(
                {
                    "capture": capture,
                    "sample_temperature": sample_temperature,
                }
            )
            proc = _FakeProc()
            state["procs"].append(proc)
            return proc, _FakeHttpd(), 4242

        def fake_run_episode(task, runs, model, extra_env=None):
            index = len(state["run_calls"])
            state["run_calls"].append({"extra_env": extra_env})
            capture = state["start_calls"][-1]["capture"]
            if capture is not None:
                capture.extend(captures_by_rollout[index])
            return outcomes[index]

        env_before = dict(os.environ)
        stdout = io.StringIO()
        error = None
        with (
            mock.patch.multiple(
                "omp_gym.rl",
                require_metal_gpu=lambda: SimpleNamespace(device_name="fake-gpu"),
                Adam=None,
                load_task=lambda path: TaskSpec(
                    name=Path(path).name,
                    prompt="do the thing",
                    test_command=("python3", "-m", "unittest"),
                    tools="bash",
                    max_time="60",
                    workspace=fixture["workspace"],
                ),
                run_episode=fake_run_episode,
                _start_policy_server=fake_start,
                _load_tokenizer=lambda model: _FakeTokenizer(),
                _provider_config_path=lambda: fixture["models_yml"],
            ),
            mock.patch("omp_gym.serve.ensure_provider", return_value=True),
            redirect_stdout(stdout),
        ):
            try:
                run_rl(
                    task_dir=fixture["root"] / "task",
                    base_model="base-model",
                    adapter_dir=fixture["adapter_in"],
                    out_adapter=fixture["out_adapter"],
                    group_size=len(rollouts),
                    iterations=1,
                    port=4600,
                    runs_dir=fixture["runs_dir"],
                    ledger_path=fixture["ledger"],
                    sample_temperature=0.3,
                    seed=7,
                    **overrides,
                )
            except RlError as exc:
                error = exc
        return {
            "error": error,
            "state": state,
            "stdout": stdout.getvalue(),
            "env_delta": {
                key: value
                for key, value in dict(os.environ).items()
                if env_before.get(key) != value
            },
        }

    def _report(self, fixture) -> dict:
        return json.loads((fixture["out_adapter"] / "rl_report.json").read_text())

    def test_zero_variance_exits_nonzero_and_writes_no_success(self) -> None:
        fixture = self._fixture()

        def record():
            return SimpleNamespace(
                reward_improvement=0.7,
                reward_partial=0.6,
                reward=0.5,
                test_exit_code=0,
            )

        result = self._run(
            fixture,
            [
                {
                    "outcome": record(),
                    "captures": [
                        {
                            "messages": [{"role": "user", "content": "go"}],
                            "text": "done",
                            "request_id": "r-1",
                        }
                    ],
                },
                {
                    "outcome": record(),
                    "captures": [
                        {
                            "messages": [{"role": "user", "content": "go"}],
                            "text": "done",
                            "request_id": "r-2",
                        }
                    ],
                },
            ],
        )
        self.assertIsNotNone(result["error"])
        self.assertIn("no policy update", str(result["error"]))
        self.assertIn("zero advantage variance", str(result["error"]))
        self.assertNotIn("rl ok", result["stdout"])
        self.assertFalse((fixture["out_adapter"] / "adapters.safetensors").exists())
        self.assertFalse(fixture["ledger"].exists())
        report = self._report(fixture)
        self.assertEqual(report["updates"], 0)
        self.assertEqual(report["rounds"][0]["rewards"], [0.7, 0.7])
        self.assertEqual(report["rounds"][0]["n_failed"], 0)
        self.assertEqual(report["rounds"][0]["request_ids"], ["r-1", "r-2"])
        self.assertEqual(report["rounds"][0]["n_captured_turns"], 2)
        self.assertEqual(report["reproducibility"]["seed"], 7)
        # Provider bytes restored verbatim.
        self.assertEqual(
            fixture["models_yml"].read_bytes(),
            b"original-provider-bytes",
        )

    def test_failed_rollouts_stay_in_the_group_at_zero(self) -> None:
        fixture = self._fixture()

        def fail(reason):
            return EpisodeFailure(task="task", reason=reason)

        result = self._run(
            fixture,
            [
                {"outcome": fail("timed out")},
                {"outcome": fail("no session")},
            ],
        )
        self.assertIsNotNone(result["error"])
        self.assertIn("no captured assistant turns", str(result["error"]))
        report = self._report(fixture)
        self.assertEqual(report["rounds"][0]["rewards"], [0.0, 0.0])
        self.assertEqual(report["rounds"][0]["n_failed"], 2)

    def test_over_cap_turns_are_counted_and_skipped(self) -> None:
        fixture = self._fixture()

        def record():
            return SimpleNamespace(reward_improvement=0.5)

        result = self._run(
            fixture,
            [
                {
                    "outcome": record(),
                    "captures": [
                        {
                            "messages": [{"role": "user", "content": "go"}],
                            "text": "done",
                        }
                    ],
                },
                {
                    "outcome": record(),
                    "captures": [
                        {
                            "messages": [{"role": "user", "content": "go"}],
                            "text": "done",
                        }
                    ],
                },
            ],
            max_seq_len=3,
        )
        report = self._report(fixture)
        self.assertEqual(report["rounds"][0]["n_skipped_long"], 2)
        self.assertEqual(report["rounds"][0]["n_failed"], 2)
        self.assertEqual(report["n_skipped_long"], 2)
        self.assertIn("no captured assistant turns", str(result["error"]))

    def test_temperature_scoped_to_episode_env_and_shim_only(self) -> None:
        fixture = self._fixture()

        def record():
            return SimpleNamespace(reward_improvement=0.5)

        result = self._run(
            fixture,
            [
                {"outcome": record(), "captures": []},
                {"outcome": record(), "captures": []},
            ],
        )
        self.assertEqual(result["env_delta"], {})
        self.assertNotIn("OMP_GYM_SAMPLE_TEMP", dict(os.environ))
        calls = result["state"]["run_calls"]
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(call["extra_env"], {"OMP_GYM_SAMPLE_TEMP": "0.3"})
        self.assertEqual(result["state"]["start_calls"][0]["sample_temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
