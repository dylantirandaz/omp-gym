"""DPO on preference pairs, natively in MLX.

The mlx-lm-lora community trainer deadlocked on this stack in
three separate runs, so the loss is implemented directly: sigmoid
DPO over completion logprobs, policy versus a frozen reference.
The reference is the SFT adapter, which makes this the standard
SFT-then-DPO chain.

Integrity rules: completions carry the tokenizer's eos token so
the policy learns to end turns; the LoRA topology comes from the
resume adapter's own adapter_config.json, never from constants;
the resume adapter must contain every expected LoRA key; pair
order is shuffled per epoch with a fixed seed; gradients
accumulate over batch_size pairs per optimizer step; padding is
per batch, rounded to 32 and capped at max_seq; gradients are
global-norm clipped; the success signal is the loss on a fixed
probe set, not the first-versus-last scalar of different pairs;
accuracy is measured on validation pairs only; and when
validation exists the run ends on the best-checkpoint weights.
"""

import json
import math
import random
import time
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import Adam, clip_grad_norm
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
except ModuleNotFoundError:
    # Off-Mac machines have no mlx. The pure helpers and the
    # constants stay importable; train_dpo fails at load time.
    mx = nn = Adam = clip_grad_norm = tree_flatten = None
    load = linear_to_lora_layers = None

from .layers import (
    adapter_layer_selection,
    layer_config_fields,
    mlx_num_layers,
)

DPO_BETA = 0.1
# Fallback topology constants, kept importable for rl.py. DPO
# training derives the real topology from the resume adapter.
LORA_CONFIG = {"rank": 8, "scale": 20.0, "dropout": 0.0}
LORA_NUM_LAYERS = 16
MAX_SEQ = 4096


class DpoError(SystemExit):
    """Raised when DPO training did not run or did not learn."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"dpo training failed: {reason}")


def _require_finite(value: float, name: str) -> float:
    """Return one finite metric or stop before it can affect weights."""
    result = float(value)
    if not math.isfinite(result):
        raise DpoError(f"{name} is not finite: {result}")
    return result


def _require_finite_tensors(tree, name: str) -> None:
    """Stop when one tensor in a parameter or gradient tree is not finite."""
    for key, value in tree_flatten(tree):
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise DpoError(f"{name} tensor is not finite: {key}")


def _completion_logprob(model, prompt_ids, completion_ids, pad_to):
    """Summed logprob of the completion tokens under the model.

    Sequences are padded to pad_to: MLX compiles a graph per input
    shape, so padding is per batch, rounded to a multiple of 32.
    The mask keeps padding out of the sum.
    """
    sequence = prompt_ids + completion_ids
    ids = mx.array(sequence + [0] * (pad_to - len(sequence)))[None]
    logits = model(ids)[0].astype(mx.float32)
    targets = ids[0, 1:]
    logprobs = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    token_lp = mx.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
    positions = mx.arange(token_lp.shape[0])
    mask = (positions >= len(prompt_ids) - 1) & (positions < len(sequence) - 1)
    return (token_lp * mask).sum()


def _eos_token_id(tokenizer) -> int:
    """The tokenizer's end-of-turn token id, or a clear error.

    DPO completions must end with the eos token so the policy
    learns to terminate turns; a tokenizer without one cannot
    express that signal.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise DpoError(
            "the tokenizer has no eos token id; DPO completions must "
            "end with an end-of-turn marker, so pick a chat model "
            "whose tokenizer defines one"
        )
    return int(eos_id)


def _read_pairs(path: Path, tokenizer, eos_id: int):
    """Tokenize one validated pair file into id sequences."""

    def chat_ids(record: dict[str, object], line_number: int) -> list[int]:
        messages = record.get("messages")
        if messages is None:
            system = record.get("system")
            prompt = record.get("prompt")
            if not isinstance(system, str) or not isinstance(prompt, str):
                raise DpoError(f"{path}:{line_number} has no valid prompt messages")
            prompt_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        elif isinstance(messages, list):
            prompt_messages = messages
        else:
            raise DpoError(f"{path}:{line_number} messages is not an array")
        templated = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        if isinstance(templated, list):
            ids = (
                templated[0]
                if templated and isinstance(templated[0], list)
                else templated
            )
        elif isinstance(templated, dict):
            ids = templated.get("input_ids")
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = None
        if not isinstance(ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in ids
        ):
            raise DpoError(f"{path}:{line_number} prompt tokenization is invalid")
        return ids

    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise DpoError(f"cannot read pair file {path}: {error}") from error
    pairs = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DpoError(f"{path}:{line_number} is not valid JSON: {error}") from None
        if not isinstance(record, dict):
            raise DpoError(f"{path}:{line_number} is not a JSON object")
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            raise DpoError(f"{path}:{line_number} needs chosen and rejected text")
        prompt_ids = chat_ids(record, line_number)
        chosen_tokens = getattr(
            tokenizer(chosen, add_special_tokens=False), "input_ids", None
        )
        rejected_tokens = getattr(
            tokenizer(rejected, add_special_tokens=False), "input_ids", None
        )
        if not isinstance(chosen_tokens, list) or not isinstance(rejected_tokens, list):
            raise DpoError(f"{path}:{line_number} completion tokenization is invalid")
        chosen_ids = chosen_tokens + [eos_id]
        rejected_ids = rejected_tokens + [eos_id]
        pairs.append((prompt_ids, chosen_ids, rejected_ids))
    return pairs


def _load_pairs(data_dir: Path, model_id: str):
    """Tokenize train and valid pair files into id sequences."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eos_id = _eos_token_id(tokenizer)
    return _read_pairs(data_dir / "train.jsonl", tokenizer, eos_id), (
        _read_pairs(data_dir / "valid.jsonl", tokenizer, eos_id)
    )


def _batch_pad_length(pairs, max_seq: int) -> int:
    """Padded length for one batch.

    The longest sequence in the batch, rounded up to a multiple of
    32 and capped at max_seq. A sequence longer than max_seq cannot
    be represented; that is an export error, not a truncation.
    """
    longest = max(
        len(prompt) + max(len(chosen), len(rejected))
        for prompt, chosen, rejected in pairs
    )
    if longest > max_seq:
        raise DpoError(
            f"longest sequence is {longest} tokens, over the {max_seq} "
            "cap; re-export pairs with a tighter --max-tokens cap"
        )
    rounded = ((longest + 31) // 32) * 32
    return min(rounded, max_seq)


def _shuffled_order(count: int, seed: int, epoch: int) -> list[int]:
    """A deterministic permutation of range(count) for one epoch.

    Same seed and epoch give the same order, so a run is exactly
    reproducible and two epochs differ.
    """
    order = list(range(count))
    random.Random(f"{seed}:{epoch}").shuffle(order)  # noqa: S311 - seeded pair order
    return order


def _steps_per_epoch(count: int, batch_size: int) -> int:
    """Optimizer steps needed to cover count pairs at batch_size."""
    return -(-count // batch_size)


def _val_steps(iterations: int) -> set[int]:
    """Steps after which validation runs. The final step always
    validates, including odd iteration counts."""
    every = max(1, iterations // 2)
    steps = set(range(every, iterations + 1, every))
    steps.add(iterations)
    return steps


def _lora_topology(adapter_file: Path):
    """Derive the LoRA topology from the resume adapter's config.

    Returns (lora_config, num_layers, config_path). Hard-coded
    constants silently drift from the adapter on disk; the adapter
    directory is the source of truth. Missing or inconsistent
    config is a hard error.
    """
    config_path = adapter_file.parent / "adapter_config.json"
    if not config_path.is_file():
        raise DpoError(
            f"missing {config_path}; the LoRA topology is derived "
            "from the resume adapter's own adapter_config.json, "
            "which must sit next to adapters.safetensors"
        )
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise DpoError(f"unreadable adapter config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise DpoError(f"adapter config {config_path} is not a JSON object")
    kind = config.get("fine_tune_type")
    if kind != "lora":
        raise DpoError(
            f"adapter config {config_path} has fine_tune_type "
            f"{kind!r}, not 'lora'; dpo resumes an SFT lora adapter"
        )
    params = config.get("lora_parameters")
    if not isinstance(params, dict):
        raise DpoError(
            f"adapter config {config_path} has no lora_parameters "
            "object; cannot derive the LoRA topology"
        )
    rank = params.get("rank")
    scale = params.get("scale")
    dropout = params.get("dropout", 0.0)
    layer_selection = adapter_layer_selection(config)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise DpoError(
            f"adapter config {config_path} has invalid lora rank "
            f"{rank!r}; expected a positive integer"
        )
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise DpoError(
            f"adapter config {config_path} has invalid lora scale "
            f"{scale!r}; expected a positive number"
        )
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0.0 <= dropout < 1.0
    ):
        raise DpoError(
            f"adapter config {config_path} has invalid lora dropout "
            f"{dropout!r}; expected a number in [0, 1)"
        )
    if layer_selection is None:
        raise DpoError(
            f"adapter config {config_path} has inconsistent layer selection; "
            "use a positive num_layers value, or num_layers=0 with "
            "layer_selection='all'"
        )
    lora_config = {
        "rank": rank,
        "scale": float(scale),
        "dropout": float(dropout),
    }
    return lora_config, layer_selection, config_path


def _missing_adapter_keys(adapter_file: Path, expected_keys) -> list[str]:
    """Expected LoRA keys absent from the adapter weights file."""
    tensors = mx.load(str(adapter_file))
    return sorted(set(expected_keys) - set(tensors))


def _load_adapter_strict(model, adapter_file: Path) -> None:
    """Load the resume adapter, refusing silent partial loads.

    load_weights(strict=False) silently skips missing keys, which
    turns a truncated or mismatched adapter into partially trained
    noise. Every expected trainable LoRA key must be present.
    """
    expected = [key for key, _ in tree_flatten(model.trainable_parameters())]
    missing = _missing_adapter_keys(adapter_file, expected)
    if missing:
        shown = ", ".join(missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise DpoError(
            f"resume adapter {adapter_file} is missing "
            f"{len(missing)} expected LoRA keys: {shown}{more}"
        )
    model.load_weights(str(adapter_file), strict=False)


def _reference_logprobs(reference, pairs, max_seq):
    """Frozen reference chosen/rejected logprobs for each pair."""
    refs = []
    for index, (prompt_ids, chosen_ids, rejected_ids) in enumerate(pairs):
        pad_to = _batch_pad_length([(prompt_ids, chosen_ids, rejected_ids)], max_seq)
        chosen = _require_finite(
            _completion_logprob(reference, prompt_ids, chosen_ids, pad_to).item(),
            f"reference chosen logprob for pair {index}",
        )
        rejected = _require_finite(
            _completion_logprob(reference, prompt_ids, rejected_ids, pad_to).item(),
            f"reference rejected logprob for pair {index}",
        )
        refs.append((chosen, rejected))
    return refs


def _dpo_loss(policy, batch, ref_lps, beta, pad_to):
    """Mean sigmoid-DPO loss over a batch of pairs."""
    losses = []
    for (prompt_ids, chosen_ids, rejected_ids), (ref_c, ref_r) in zip(
        batch, ref_lps, strict=True
    ):
        pol_c = _completion_logprob(policy, prompt_ids, chosen_ids, pad_to)
        pol_r = _completion_logprob(policy, prompt_ids, rejected_ids, pad_to)
        logit = beta * ((pol_c - ref_c) - (pol_r - ref_r))
        losses.append(-nn.log_sigmoid(logit))
    return mx.stack(losses).mean()


def _mean_dpo_loss(policy, pairs, ref_lps, beta, max_seq):
    """Mean DPO loss over a pair set, or None when it is empty."""
    if not pairs:
        return None
    total = 0.0
    for pair, ref in zip(pairs, ref_lps, strict=True):
        pad_to = _batch_pad_length([pair], max_seq)
        total += _dpo_loss(policy, [pair], [ref], beta, pad_to).item()
    return total / len(pairs)


def _chosen_accuracy(policy, pairs, max_seq):
    """Fraction of pairs where the policy prefers the chosen side,
    or None when the set is empty. Only ever called on validation
    pairs, never on pairs just trained on."""
    if not pairs:
        return None
    correct = 0
    for prompt_ids, chosen_ids, rejected_ids in pairs:
        pad_to = _batch_pad_length([(prompt_ids, chosen_ids, rejected_ids)], max_seq)
        pol_c = _completion_logprob(policy, prompt_ids, chosen_ids, pad_to).item()
        pol_r = _completion_logprob(policy, prompt_ids, rejected_ids, pad_to).item()
        correct += 1 if pol_c > pol_r else 0
    return correct / len(pairs)


def _validate_dpo_parameters(
    iterations: int,
    learning_rate: float,
    batch_size: int,
    beta: float,
    grad_clip: float,
    max_seq: int,
) -> None:
    """Reject settings that cannot produce a valid DPO run."""
    if iterations < 2:
        raise DpoError("iterations must be at least 2")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise DpoError("learning rate must be finite and positive")
    if batch_size < 1:
        raise DpoError("batch size must be positive")
    if not math.isfinite(beta) or beta <= 0.0:
        raise DpoError("beta must be finite and positive")
    if not math.isfinite(grad_clip) or grad_clip < 0.0:
        raise DpoError("gradient clip must be finite and nonnegative")
    if max_seq < 1:
        raise DpoError("maximum sequence length must be positive")


def train_dpo(
    data_dir: Path,
    model_id: str,
    iterations: int,
    adapter_dir: Path,
    resume_adapter: Path,
    device_name: str,
    learning_rate: float,
    batch_size: int = 1,
    seed: int = 0,
    beta: float = DPO_BETA,
    grad_clip: float = 1.0,
    max_seq: int = MAX_SEQ,
) -> dict:
    """Train and return the loss curves as a plain dict."""
    _validate_dpo_parameters(
        iterations, learning_rate, batch_size, beta, grad_clip, max_seq
    )
    lora_config, layer_selection, topology_path = _lora_topology(resume_adapter)
    print(
        f"lora topology from {topology_path}: rank={lora_config['rank']}, "
        f"scale={lora_config['scale']}, layers={layer_selection}"
    )

    mlx_layers = mlx_num_layers(layer_selection)
    print(f"loading policy: {model_id}")
    policy, _ = load(model_id)
    policy.freeze()
    linear_to_lora_layers(policy, mlx_layers, lora_config)
    print(f"resuming SFT adapter: {resume_adapter}")
    _load_adapter_strict(policy, resume_adapter)

    print("loading frozen reference model")
    reference, _ = load(model_id)
    reference.freeze()
    linear_to_lora_layers(reference, mlx_layers, lora_config)
    _load_adapter_strict(reference, resume_adapter)

    train_pairs, valid_pairs = _load_pairs(data_dir, model_id)
    if len(train_pairs) < 2:
        raise DpoError("need at least two training pairs")

    print(f"precomputing reference logprobs for {len(train_pairs)} pairs")
    train_ref = _reference_logprobs(reference, train_pairs, max_seq)
    valid_ref = _reference_logprobs(reference, valid_pairs, max_seq)

    probe_pairs = train_pairs[:8]
    probe_ref = train_ref[:8]
    fixed_first = _require_finite(
        _mean_dpo_loss(policy, probe_pairs, probe_ref, beta, max_seq),
        "fixed probe loss before training",
    )
    print(f"fixed probe loss before training: {fixed_first:.4f}")

    def loss_fn(model, batch, ref_lps, pad_to):
        return _dpo_loss(model, batch, ref_lps, beta, pad_to)

    loss_and_grad = nn.value_and_grad(policy, loss_fn)
    optimizer = Adam(learning_rate=learning_rate)

    initial_val = _mean_dpo_loss(policy, valid_pairs, valid_ref, beta, max_seq)
    if initial_val is not None:
        initial_val = _require_finite(initial_val, "initial validation loss")
    val_losses: list[float | None] = [initial_val]
    if initial_val is None:
        print("Iter 0: validation: none")
    else:
        print(f"Iter 0: Val loss {initial_val:.4f}")

    # Best-checkpoint state: the snapshot with the lowest validation
    # loss. The initial snapshot is the resume adapter itself.
    best_val = initial_val
    best_weights = (
        list(tree_flatten(policy.trainable_parameters()))
        if initial_val is not None
        else None
    )

    steps_per_epoch = _steps_per_epoch(len(train_pairs), batch_size)
    val_at = _val_steps(iterations)
    train_losses: list[float] = []
    started = time.monotonic()
    order: list[int] = []

    for step in range(1, iterations + 1):
        if (step - 1) % steps_per_epoch == 0:
            epoch = (step - 1) // steps_per_epoch
            order = _shuffled_order(len(train_pairs), seed, epoch)
        slot = (step - 1) % steps_per_epoch
        indexes = order[slot * batch_size : (slot + 1) * batch_size]
        batch = [train_pairs[i] for i in indexes]
        ref_lps = [train_ref[i] for i in indexes]
        pad_to = _batch_pad_length(batch, max_seq)
        loss, grads = loss_and_grad(policy, batch, ref_lps, pad_to)
        value = _require_finite(loss.item(), f"train loss at step {step}")
        _require_finite_tensors(grads, f"gradients at step {step}")
        if grad_clip > 0:
            grads, norm = clip_grad_norm(grads, grad_clip)
            _require_finite(norm.item(), f"gradient norm at step {step}")
        optimizer.update(policy, grads)
        mx.eval(policy.trainable_parameters(), optimizer.state)
        _require_finite_tensors(
            policy.trainable_parameters(),
            f"policy parameters after step {step}",
        )
        train_losses.append(value)
        print(f"Iter {step}: Train loss {value:.4f} ({len(batch)} pairs, seed {seed})")
        if step in val_at:
            val = _mean_dpo_loss(policy, valid_pairs, valid_ref, beta, max_seq)
            if val is not None:
                val = _require_finite(val, f"validation loss at step {step}")
            val_losses.append(val)
            if val is not None:
                print(f"Iter {step}: Val loss {val:.4f}")
                if best_val is None or val < best_val:
                    best_val = val
                    best_weights = list(tree_flatten(policy.trainable_parameters()))

    if len(train_losses) < 2:
        raise DpoError("no loss reports produced")

    if best_weights is not None:
        policy.load_weights(best_weights, strict=False)
        print(f"restored best checkpoint: val loss {best_val:.4f}")

    fixed_last = _require_finite(
        _mean_dpo_loss(policy, probe_pairs, probe_ref, beta, max_seq),
        "fixed probe loss after training",
    )
    if fixed_last >= fixed_first:
        raise DpoError(
            f"fixed probe loss did not go down: {fixed_first:.4f} -> {fixed_last:.4f}"
        )
    val_accuracy = _chosen_accuracy(policy, valid_pairs, max_seq)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(policy.trainable_parameters()))
    _require_finite_tensors(adapter_weights, "output adapter")
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), adapter_weights)
    config = {
        "fine_tune_type": "lora",
        "method": "dpo",
        "beta": beta,
        "seed": seed,
        "batch_size": batch_size,
        "grad_clip": grad_clip,
        "model": model_id,
        "lora_parameters": lora_config,
        **layer_config_fields(layer_selection),
        "topology_source": str(topology_path),
        "resume_adapter_file": str(resume_adapter),
        "iters": iterations,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    elapsed = time.monotonic() - started
    validation = (
        f"val loss {val_losses[0]:.4f} -> {best_val:.4f} (best checkpoint)"
        if val_losses[0] is not None
        else "validation: none"
    )
    accuracy = f", val accuracy {val_accuracy:.0%}" if val_accuracy is not None else ""
    print(
        f"dpo ok: fixed probe loss {fixed_first:.4f} -> "
        f"{fixed_last:.4f}, {validation}{accuracy}, "
        f"{elapsed:.0f}s on {device_name}"
    )
    return {
        "first_train_loss": train_losses[0],
        "last_train_loss": train_losses[-1],
        "first_val_loss": val_losses[0],
        # The report describes the restored policy: the best-checkpoint
        # loss (the initial evaluation when nothing improved), not the
        # last pre-rollback scan.
        "last_val_loss": best_val,
        "beta": beta,
        "seed": seed,
        "batch_size": batch_size,
        "grad_clip": grad_clip,
        "fixed_first_loss": fixed_first,
        "fixed_last_loss": fixed_last,
        "val_accuracy": val_accuracy,
        "topology_source": str(topology_path),
        "elapsed_seconds": round(elapsed, 1),
    }
