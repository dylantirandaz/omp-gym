"""DPO on preference pairs, natively in MLX.

The community trainer deadlocked on this stack (two killed runs
and one idle process at zero CPU), so the loss is implemented
directly: sigmoid DPO over completion logprobs, policy versus a
frozen reference. The reference is the SFT adapter, which makes
this the standard SFT-then-DPO chain.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

DPO_BETA = 0.1
LORA_CONFIG = {"rank": 8, "scale": 20.0, "dropout": 0.0}
LORA_NUM_LAYERS = 16


class DpoError(SystemExit):
    """Raised when DPO training did not run or did not learn."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"dpo training failed: {reason}")


def _completion_logprob(model, prompt_ids, completion_ids, pad_to):
    """Summed logprob of the completion tokens under the model.

    Sequences are padded to one fixed length: MLX compiles a graph
    per input shape, and shape-varying sequences would pay one full
    compilation per pair. The mask keeps padding out of the sum.
    """
    sequence = prompt_ids + completion_ids
    ids = mx.array(sequence + [0] * (pad_to - len(sequence)))[None]
    logits = model(ids)[0].astype(mx.float32)
    targets = ids[0, 1:]
    logprobs = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    token_lp = mx.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
    positions = mx.arange(token_lp.shape[0])
    mask = (positions >= len(prompt_ids) - 1) & (
        positions < len(sequence) - 1
    )
    return (token_lp * mask).sum()


def _load_pairs(data_dir: Path, model_id: str):
    """Tokenize train and valid pair files into id sequences."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(path: Path):
        pairs = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            templated = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": record["system"]},
                    {"role": "user", "content": record["prompt"]},
                ],
                add_generation_prompt=True,
                tokenize=True,
            )
            if isinstance(templated, list):
                prompt_ids = (
                    templated[0]
                    if templated and isinstance(templated[0], list)
                    else templated
                )
            else:
                ids = templated["input_ids"]
                prompt_ids = (
                    ids[0] if ids and isinstance(ids[0], list) else ids
                )
            chosen_ids = tokenizer(
                record["chosen"], add_special_tokens=False
            ).input_ids
            rejected_ids = tokenizer(
                record["rejected"], add_special_tokens=False
            ).input_ids
            pairs.append((prompt_ids, chosen_ids, rejected_ids))
        return pairs

    return tokenize(data_dir / "train.jsonl"), tokenize(
        data_dir / "valid.jsonl"
    )


def _dpo_loss(policy, batch, ref_lps, beta, pad_to):
    """Mean sigmoid-DPO loss over a batch of pairs."""
    losses = []
    for (prompt_ids, chosen_ids, rejected_ids), (ref_c, ref_r) in zip(
        batch, ref_lps
    ):
        pol_c = _completion_logprob(policy, prompt_ids, chosen_ids, pad_to)
        pol_r = _completion_logprob(
            policy, prompt_ids, rejected_ids, pad_to
        )
        logit = beta * ((pol_c - ref_c) - (pol_r - ref_r))
        losses.append(-nn.log_sigmoid(logit))
    return mx.stack(losses).mean()


def train_dpo(
    data_dir: Path,
    model_id: str,
    iterations: int,
    adapter_dir: Path,
    resume_adapter: Path,
    device_name: str,
) -> dict:
    """Train and return the loss curves as a plain dict."""
    print(f"loading policy: {model_id}")
    policy, _ = load(model_id)
    policy.freeze()
    linear_to_lora_layers(policy, LORA_NUM_LAYERS, LORA_CONFIG)
    print(f"resuming SFT adapter: {resume_adapter}")
    policy.load_weights(str(resume_adapter), strict=False)

    print("loading frozen reference model")
    reference, _ = load(model_id)
    reference.freeze()
    linear_to_lora_layers(reference, LORA_NUM_LAYERS, LORA_CONFIG)
    reference.load_weights(str(resume_adapter), strict=False)

    train_pairs, valid_pairs = _load_pairs(data_dir, model_id)
    if len(train_pairs) < 2:
        raise DpoError("need at least two training pairs")

    longest = max(
        len(prompt) + max(len(chosen), len(rejected))
        for prompt, chosen, rejected in train_pairs + valid_pairs
    )
    pad_to = ((longest + 127) // 128) * 128
    if pad_to > 4096:
        raise DpoError(
            f"longest pair needs {pad_to} padded tokens; "
            "re-export pairs with a tighter --max-tokens cap"
        )
    print(f"padding all sequences to {pad_to} tokens")

    print(f"precomputing reference logprobs for {len(train_pairs)} pairs")
    train_ref = [
        (
            _completion_logprob(reference, prompt, chosen, pad_to).item(),
            _completion_logprob(
                reference, prompt, rejected, pad_to
            ).item(),
        )
        for prompt, chosen, rejected in train_pairs
    ]
    valid_ref = [
        (
            _completion_logprob(reference, prompt, chosen, pad_to).item(),
            _completion_logprob(
                reference, prompt, rejected, pad_to
            ).item(),
        )
        for prompt, chosen, rejected in valid_pairs
    ]

    def loss_fn(model, batch, ref_lps):
        return _dpo_loss(model, batch, ref_lps, DPO_BETA, pad_to)

    loss_and_grad = nn.value_and_grad(policy, loss_fn)
    optimizer = Adam(learning_rate=5e-6)

    def mean_val_loss():
        total = sum(
            _dpo_loss(policy, [pair], [ref], DPO_BETA, pad_to).item()
            for pair, ref in zip(valid_pairs, valid_ref)
        )
        return total / max(1, len(valid_pairs))

    train_losses: list[float] = []
    val_losses: list[float] = []
    accuracies: list[float] = []
    started = time.monotonic()

    val_losses.append(mean_val_loss())
    print(f"Iter 0: Val loss {val_losses[0]:.4f}")

    for iteration in range(1, iterations + 1):
        index = (iteration - 1) % len(train_pairs)
        batch = [train_pairs[index]]
        ref_lps = [train_ref[index]]
        loss, grads = loss_and_grad(policy, batch, ref_lps)
        optimizer.update(policy, grads)
        mx.eval(loss, policy.trainable_parameters(), optimizer.state)
        value = loss.item()
        if value != value:
            raise DpoError(f"NaN loss at iteration {iteration}")
        train_losses.append(value)

        pair = train_pairs[index]
        pol_c = _completion_logprob(
            policy, pair[0], pair[1], pad_to
        ).item()
        pol_r = _completion_logprob(
            policy, pair[0], pair[2], pad_to
        ).item()
        accuracies.append(1.0 if pol_c > pol_r else 0.0)

        print(
            f"Iter {iteration}: Train loss {value:.4f}, "
            f"Chosen preferred: {pol_c > pol_r}"
        )
        if iteration % max(1, iterations // 2) == 0:
            val_losses.append(mean_val_loss())
            print(f"Iter {iteration}: Val loss {val_losses[-1]:.4f}")

    if len(train_losses) < 2:
        raise DpoError("no loss reports produced")
    if train_losses[-1] >= train_losses[0]:
        raise DpoError(
            f"train loss did not go down: "
            f"{train_losses[0]} -> {train_losses[-1]}"
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(policy.trainable_parameters()))
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"),
                        adapter_weights)
    config = {
        "fine_tune_type": "lora",
        "method": "dpo",
        "beta": DPO_BETA,
        "model": model_id,
        "lora_parameters": LORA_CONFIG,
        "num_layers": LORA_NUM_LAYERS,
        "resume_adapter_file": str(resume_adapter),
        "iters": iterations,
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2)
    )
    elapsed = time.monotonic() - started
    accuracy = sum(accuracies) / len(accuracies)
    print(
        f"dpo ok: loss {train_losses[0]:.4f} -> {train_losses[-1]:.4f}, "
        f"accuracy {accuracy:.0%}, {elapsed:.0f}s on {device_name}"
    )
    return {
        "first_train_loss": train_losses[0],
        "last_train_loss": train_losses[-1],
        "first_val_loss": val_losses[0] if val_losses else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
        "train_accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
    }
