import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

from omp_gym import dpo
from omp_gym.dpo import (
    DPO_BETA,
    LORA_CONFIG,
    DpoError,
    _batch_pad_length,
    _chosen_accuracy,
    _eos_token_id,
    _lora_topology,
    _mean_dpo_loss,
    _read_pairs,
    _require_finite,
    _shuffled_order,
    _steps_per_epoch,
    _val_steps,
    _validate_dpo_parameters,
    train_dpo,
)

mx = dpo.mx
nn = dpo.nn


def _pair(prompt: int, chosen: int, rejected: int):
    return ([0] * prompt, [0] * chosen, [0] * rejected)


class FakeTokenizer:
    """Tokenizer seam: fixed prompt ids, length-coded completions."""

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return [1, 2, 3]

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = [len(text)]

        return Encoded()


class EosTests(unittest.TestCase):
    def test_returns_the_eos_id(self) -> None:
        self.assertEqual(_eos_token_id(FakeTokenizer(7)), 7)

    def test_zero_is_a_valid_eos_id(self) -> None:
        self.assertEqual(_eos_token_id(FakeTokenizer(0)), 0)

    def test_missing_eos_is_a_clear_error(self) -> None:
        with self.assertRaises(DpoError) as caught:
            _eos_token_id(FakeTokenizer(None))
        self.assertIn("eos", str(caught.exception))

    def test_pairs_append_eos_to_both_sides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "chosen": "yes",
                        "rejected": "no",
                    }
                )
                + "\n"
            )
            pairs = _read_pairs(path, FakeTokenizer(7), 7)
        self.assertEqual(len(pairs), 1)
        prompt, chosen, rejected = pairs[0]
        self.assertEqual(prompt, [1, 2, 3])
        self.assertEqual(chosen, [3, 7])
        self.assertEqual(rejected, [2, 7])

    def test_system_prompt_records_also_get_eos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "system": "be terse",
                        "prompt": "hi",
                        "chosen": "ok",
                        "rejected": "fine",
                    }
                )
                + "\n"
            )
            pairs = _read_pairs(path, FakeTokenizer(9), 9)
        _, chosen, rejected = pairs[0]
        self.assertEqual(chosen[-1], 9)
        self.assertEqual(rejected[-1], 9)

    def test_non_object_pair_record_is_a_declared_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text("[]\n")
            with self.assertRaises(DpoError) as caught:
                _read_pairs(path, FakeTokenizer(9), 9)
        self.assertIn("JSON object", str(caught.exception))


class TopologyTests(unittest.TestCase):
    def _adapter(self, tmp: str, config) -> Path:
        (Path(tmp) / "adapter_config.json").write_text(
            config if isinstance(config, str) else json.dumps(config)
        )
        return Path(tmp) / "adapters.safetensors"

    def _good_config(self) -> dict:
        return {
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": 16,
                "scale": 32.0,
                "dropout": 0.05,
            },
            "num_layers": 24,
        }

    def test_topology_comes_from_the_adapter_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, self._good_config())
            lora_config, num_layers, source = _lora_topology(adapter)
        self.assertEqual(lora_config, {"rank": 16, "scale": 32.0, "dropout": 0.05})
        self.assertEqual(num_layers, 24)
        self.assertEqual(source.name, "adapter_config.json")

    def test_all_layer_marker_survives_topology_loading(self) -> None:
        config = self._good_config()
        config["num_layers"] = 0
        config["layer_selection"] = "all"
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, config)
            _, layer_selection, _ = _lora_topology(adapter)
        self.assertEqual(layer_selection, "all")

    def test_mismatched_layer_fields_are_rejected(self) -> None:
        config = self._good_config()
        config["layer_selection"] = "all"
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, config)
            with self.assertRaises(DpoError):
                _lora_topology(adapter)

    def test_missing_config_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DpoError) as caught:
                _lora_topology(Path(tmp) / "adapters.safetensors")
        self.assertIn("adapter_config.json", str(caught.exception))

    def test_malformed_config_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, "{not json")
            with self.assertRaises(DpoError):
                _lora_topology(adapter)

    def test_non_lora_adapter_is_rejected(self) -> None:
        config = self._good_config()
        config["fine_tune_type"] = "full"
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, config)
            with self.assertRaises(DpoError) as caught:
                _lora_topology(adapter)
        self.assertIn("lora", str(caught.exception))

    def test_missing_lora_parameters_is_rejected(self) -> None:
        config = self._good_config()
        del config["lora_parameters"]
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, config)
            with self.assertRaises(DpoError):
                _lora_topology(adapter)

    def test_inconsistent_rank_is_rejected(self) -> None:
        for bad_rank in ("8", 0, -4, None, True):
            config = self._good_config()
            config["lora_parameters"]["rank"] = bad_rank
            with tempfile.TemporaryDirectory() as tmp:
                adapter = self._adapter(tmp, config)
                with self.assertRaises(DpoError, msg=f"rank={bad_rank}"):
                    _lora_topology(adapter)

    def test_inconsistent_num_layers_is_rejected(self) -> None:
        config = self._good_config()
        del config["num_layers"]
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, config)
            with self.assertRaises(DpoError) as caught:
                _lora_topology(adapter)
        self.assertIn("num_layers", str(caught.exception))

    def test_missing_dropout_defaults_to_zero(self) -> None:
        config = self._good_config()
        del config["lora_parameters"]["dropout"]
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp, config)
            lora_config, _, _ = _lora_topology(adapter)
        self.assertEqual(lora_config["dropout"], 0.0)


class ShuffleTests(unittest.TestCase):
    def test_same_seed_and_epoch_give_the_same_order(self) -> None:
        self.assertEqual(_shuffled_order(64, 7, 0), _shuffled_order(64, 7, 0))

    def test_order_is_a_permutation(self) -> None:
        self.assertEqual(sorted(_shuffled_order(64, 7, 0)), list(range(64)))

    def test_later_epochs_get_a_new_order(self) -> None:
        self.assertNotEqual(_shuffled_order(64, 7, 0), _shuffled_order(64, 7, 1))

    def test_a_different_seed_changes_the_order(self) -> None:
        self.assertNotEqual(_shuffled_order(64, 7, 0), _shuffled_order(64, 8, 0))


class AccumulationTests(unittest.TestCase):
    def test_batch_two_over_four_pairs_is_two_steps(self) -> None:
        self.assertEqual(_steps_per_epoch(4, 2), 2)
        self.assertEqual(_steps_per_epoch(4, 1), 4)
        self.assertEqual(_steps_per_epoch(4, 4), 1)
        self.assertEqual(_steps_per_epoch(5, 2), 3)


@unittest.skipIf(mx is None, "mlx is not available")
class LossMathTests(unittest.TestCase):
    def setUp(self) -> None:
        mx.set_default_device(mx.cpu)
        # Constant per-position logits: token 1 scores 1.0, token 2
        # scores 0.5, so the chosen-minus-rejected margin is fixed.
        self.policy = self._policy([0.0, 1.0, 0.5])

    @staticmethod
    def _policy(logits):
        weights = mx.array(logits, dtype=mx.float32)

        class Policy:
            def __call__(self, ids):
                seq = ids.shape[1]
                return mx.broadcast_to(weights[None, None], (1, seq, weights.shape[0]))

        return Policy()

    def test_batch_loss_is_the_mean_of_the_pair_losses(self) -> None:
        # Accumulating two pairs in one batch must equal stepping on
        # each pair's mean contribution: grad(mean) == mean(grads).
        pair_a = ([0], [1], [2])
        pair_b = ([0], [2], [1])
        refs = [(0.0, 0.0)]
        loss_a = dpo._dpo_loss(self.policy, [pair_a], refs, 0.1, 32).item()
        loss_b = dpo._dpo_loss(self.policy, [pair_b], refs, 0.1, 32).item()
        batched = dpo._dpo_loss(self.policy, [pair_a, pair_b], refs * 2, 0.1, 32).item()
        self.assertAlmostEqual(batched, (loss_a + loss_b) / 2, places=6)

    def test_beta_scales_the_preference_logit(self) -> None:
        pair = ([0], [1], [2])
        refs = [(0.0, 0.0)]
        for beta in (0.05, 0.1, 0.3):
            loss = dpo._dpo_loss(self.policy, [pair], refs, beta, 32).item()
            # margin = logprob(1) - logprob(2) = 1.0 - 0.5
            expected = math.log1p(math.exp(-beta * 0.5))
            self.assertAlmostEqual(loss, expected, places=5)

    def test_beta_changes_the_loss(self) -> None:
        pair = ([0], [1], [2])
        refs = [(0.0, 0.0)]
        low = dpo._dpo_loss(self.policy, [pair], refs, 0.1, 32).item()
        high = dpo._dpo_loss(self.policy, [pair], refs, 0.2, 32).item()
        self.assertNotAlmostEqual(low, high, places=6)

    def test_chosen_accuracy_prefers_higher_logprob(self) -> None:
        good = ([0], [1], [2])
        bad = ([0], [2], [1])
        self.assertEqual(_chosen_accuracy(self.policy, [good], 64), 1.0)
        self.assertEqual(_chosen_accuracy(self.policy, [bad], 64), 0.0)
        self.assertEqual(_chosen_accuracy(self.policy, [good, bad], 64), 0.5)


class ParameterValidationTests(unittest.TestCase):
    def test_valid_parameters_pass(self) -> None:
        _validate_dpo_parameters(2, 1e-5, 1, 0.1, 1.0, 2048)

    def test_invalid_parameters_are_rejected(self) -> None:
        invalid = (
            (1, 1e-5, 1, 0.1, 1.0, 2048),
            (2, math.nan, 1, 0.1, 1.0, 2048),
            (2, 1e-5, 0, 0.1, 1.0, 2048),
            (2, 1e-5, 1, 0.0, 1.0, 2048),
            (2, 1e-5, 1, 0.1, -1.0, 2048),
            (2, 1e-5, 1, 0.1, 1.0, 0),
        )
        for parameters in invalid:
            with self.assertRaises(DpoError):
                _validate_dpo_parameters(*parameters)


class EmptyValidationTests(unittest.TestCase):
    def test_mean_loss_of_no_pairs_is_none(self) -> None:
        self.assertIsNone(_mean_dpo_loss(object(), [], [], 0.1, 64))

    def test_accuracy_of_no_pairs_is_none(self) -> None:
        self.assertIsNone(_chosen_accuracy(object(), [], 64))


class FiniteMetricTests(unittest.TestCase):
    def test_finite_values_pass_through(self) -> None:
        self.assertEqual(_require_finite(0.25, "loss"), 0.25)

    def test_nan_and_infinity_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(DpoError):
                _require_finite(value, "loss")


class FinalValidationTests(unittest.TestCase):
    def test_odd_iteration_counts_still_validate_at_the_end(self) -> None:
        self.assertIn(11, _val_steps(11))
        self.assertIn(5, _val_steps(11))

    def test_even_counts_validate_at_half_and_end(self) -> None:
        self.assertEqual(_val_steps(10), {5, 10})

    def test_tiny_counts_validate_every_step(self) -> None:
        self.assertEqual(_val_steps(2), {1, 2})
        self.assertEqual(_val_steps(1), {1})


class BatchPaddingTests(unittest.TestCase):
    def test_pads_to_a_multiple_of_32(self) -> None:
        self.assertEqual(_batch_pad_length([_pair(10, 20, 5)], 4096), 32)
        self.assertEqual(_batch_pad_length([_pair(10, 30, 1)], 4096), 64)
        self.assertEqual(_batch_pad_length([_pair(30, 1, 1)], 4096), 32)

    def test_the_longer_completion_side_wins(self) -> None:
        # 100 + 60 = 160 -> 160 (already a multiple of 32).
        self.assertEqual(_batch_pad_length([_pair(100, 10, 60)], 4096), 160)

    def test_padding_is_per_batch(self) -> None:
        small = _batch_pad_length([_pair(10, 10, 10)], 4096)
        large = _batch_pad_length([_pair(10, 10, 10), _pair(300, 100, 20)], 4096)
        self.assertLess(small, large)
        self.assertEqual(large, 416)

    def test_rounding_is_capped_at_max_seq(self) -> None:
        # longest 33 rounds to 64 but max_seq 48 caps it.
        self.assertEqual(_batch_pad_length([_pair(30, 3, 1)], 48), 48)

    def test_a_sequence_over_max_seq_is_an_error(self) -> None:
        with self.assertRaises(DpoError) as caught:
            _batch_pad_length([_pair(100, 10, 5)], 64)
        self.assertIn("--max-tokens", str(caught.exception))


@unittest.skipIf(mx is None, "mlx is not available")
class StrictAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        mx.set_default_device(mx.cpu)

    def test_missing_keys_are_reported_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapters.safetensors"
            mx.save_safetensors(
                str(path),
                {"a": mx.zeros((1,)), "b": mx.zeros((1,))},
            )
            self.assertEqual(
                dpo._missing_adapter_keys(path, ["a", "b", "c", "d"]),
                ["c", "d"],
            )
            self.assertEqual(dpo._missing_adapter_keys(path, ["a", "b"]), [])

    def test_strict_load_errors_listing_missing_keys(self) -> None:
        model = nn.Linear(2, 2)
        expected = [key for key, _ in dpo.tree_flatten(model.trainable_parameters())]
        self.assertGreater(len(expected), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapters.safetensors"
            mx.save_safetensors(str(path), {expected[0]: mx.zeros((2, 2))})
            with self.assertRaises(DpoError) as caught:
                dpo._load_adapter_strict(model, path)
        message = str(caught.exception)
        self.assertIn("missing", message)
        self.assertIn(expected[1], message)

    def test_strict_load_accepts_a_complete_adapter(self) -> None:
        model = nn.Linear(2, 2)
        weights = {
            key: mx.zeros_like(value)
            for key, value in dpo.tree_flatten(model.trainable_parameters())
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapters.safetensors"
            mx.save_safetensors(str(path), weights)
            dpo._load_adapter_strict(model, path)
        self.assertEqual(model.weight.abs().sum().item(), 0.0)


class ParameterDefaultsTests(unittest.TestCase):
    def test_new_parameters_have_stable_defaults(self) -> None:
        params = inspect.signature(train_dpo).parameters
        self.assertEqual(params["beta"].default, DPO_BETA)
        self.assertEqual(params["seed"].default, 0)
        self.assertEqual(params["batch_size"].default, 1)
        self.assertEqual(params["grad_clip"].default, 1.0)
        self.assertEqual(params["max_seq"].default, dpo.MAX_SEQ)


@unittest.skipIf(mx is None, "mlx is not available")
class RestoredPolicyReportTests(unittest.TestCase):
    """last_val_loss describes the restored best-checkpoint policy,
    not the last pre-rollback scan. CPU seams, no model loads."""

    def setUp(self) -> None:
        mx.set_default_device(mx.cpu)

    @staticmethod
    def _fake_load(model_id):
        mx.random.seed(42)

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 16)
                self.proj = nn.Linear(16, 8)

            def __call__(self, ids):
                return self.proj(self.embed(ids))

            def freeze(self, *args, **kwargs):
                self.embed.freeze()

        return TinyLM(), None

    def _run(self, train_pairs, valid_pairs, tmp) -> tuple[dict, str]:
        import contextlib
        import io

        tmp = Path(tmp)
        resume = tmp / "sft"
        resume.mkdir()
        (resume / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "lora_parameters": {"rank": 4, "scale": 8.0},
                    "num_layers": 2,
                }
            )
        )
        model, _ = self._fake_load("fake")
        mx.save_safetensors(
            str(resume / "adapters.safetensors"),
            dict(dpo.tree_flatten(model.trainable_parameters())),
        )
        saved = (dpo.load, dpo.linear_to_lora_layers, dpo._load_pairs)
        dpo.load = self._fake_load
        dpo.linear_to_lora_layers = lambda model, num_layers, config: None
        dpo._load_pairs = lambda data_dir, model_id: (train_pairs, valid_pairs)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                report = train_dpo(
                    data_dir=tmp,
                    model_id="fake",
                    iterations=5,
                    adapter_dir=tmp / "out",
                    resume_adapter=resume / "adapters.safetensors",
                    device_name="cpu-test",
                    learning_rate=1.0,
                    batch_size=2,
                )
        finally:
            dpo.load, dpo.linear_to_lora_layers, dpo._load_pairs = saved
        return report, output.getvalue()

    def test_last_val_loss_is_the_restored_best_not_the_last_scan(
        self,
    ) -> None:
        # Validation scans scripted to dip then rise, so the best
        # checkpoint is neither the initial scan nor the final one.
        # The train/probe path still uses the real loss.
        train = [([1, 2], [3, 7], [4, 7]) for _ in range(6)]
        valid = [([1, 2], [4, 7], [3, 7]) for _ in range(2)]
        scripted = iter([0.5, 0.4, 0.48, 0.45])
        real_mean = dpo._mean_dpo_loss

        def mean_loss(policy, pairs, ref_lps, beta, max_seq):
            if pairs is valid:
                return next(scripted)
            return real_mean(policy, pairs, ref_lps, beta, max_seq)

        dpo._mean_dpo_loss = mean_loss
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report, _ = self._run(train, valid, tmp)
        finally:
            dpo._mean_dpo_loss = real_mean
        self.assertEqual(report["first_val_loss"], 0.5)
        self.assertEqual(report["last_val_loss"], 0.4)
        self.assertNotEqual(report["last_val_loss"], 0.45)

    def test_empty_validation_reports_none(self) -> None:
        train = [([1, 2], [3, 7], [4, 7]) for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._run(train, [], tmp)
        self.assertIsNone(report["first_val_loss"])
        self.assertIsNone(report["last_val_loss"])
        self.assertIsNone(report["val_accuracy"])


class ConstantsTests(unittest.TestCase):
    def test_lora_config_is_importable_without_mlx(self) -> None:
        # rl.py and the config artifacts depend on these keys.
        self.assertEqual(set(LORA_CONFIG), {"rank", "scale", "dropout"})
        self.assertGreater(DPO_BETA, 0.0)


if __name__ == "__main__":
    unittest.main()
