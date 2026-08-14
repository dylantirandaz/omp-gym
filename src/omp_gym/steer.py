"""Activation steering with an SAE feature direction.

Pick an SAE feature, take its decoder column as a direction in
residual space, and add alpha * direction to the hidden state at
the SAE layer during generation. The A/B eval compares steered
against unsteered completions on the same prompts and measures
how often each produces a well-formed tool call.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

from .preflight import require_metal_gpu
from .sae import SAE_LAYER

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*\{", re.DOTALL)


class SteerError(SystemExit):
    """Raised when steering cannot run."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"steer failed: {reason}")


@dataclass(frozen=True)
class SteerResult:
    """One A/B comparison outcome."""

    feature: int
    alpha: float
    prompts: int
    unsteered_tool_calls: int
    steered_tool_calls: int
    steered_texts: tuple[str, ...]
    unsteered_texts: tuple[str, ...]


def _steered_forward(model, ids, direction, alpha, capture_layer):
    """Run a forward pass with the direction added at one layer."""
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        ids.shape[1]
    ).astype(h.dtype)
    for index, layer in enumerate(inner.layers):
        h = layer(h, mask=mask, cache=None)
        if index == capture_layer:
            h = h + alpha * direction
    normed = inner.norm(h)
    if hasattr(model, "lm_head"):
        return model.lm_head(normed)
    return inner.embed_tokens.as_linear(normed)


def _generate(model, tokenizer, prompt_ids, direction, alpha, max_tokens):
    """Greedy decode with steering applied at every step."""
    ids = prompt_ids[None]
    sampler = make_sampler(temp=0.0)
    generated = []
    for _ in range(max_tokens):
        logits = _steered_forward(model, ids, direction, alpha, SAE_LAYER)
        next_token = sampler(logits[:, -1, :])
        token_id = int(next_token.item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        ids = mx.concatenate([ids, next_token[None]], axis=1)
    return tokenizer.decode(generated)


def run_ab(
    weights_path: Path,
    feature: int,
    model_id: str,
    adapter_dir: Path | None,
    alpha: float,
    prompts: list[str],
    max_tokens: int,
) -> SteerResult:
    """Compare steered and unsteered completions on the prompts."""
    require_metal_gpu()
    if not weights_path.is_file():
        raise SteerError(f"no SAE weights at {weights_path}")
    weights = mx.load(str(weights_path))
    if not (0 <= feature < weights["dec_w"].shape[1]):
        raise SteerError(
            f"feature {feature} out of range "
            f"0..{weights['dec_w'].shape[1] - 1}"
        )
    direction = weights["dec_w"][:, feature]

    model, tokenizer = load(model_id)
    if adapter_dir is not None:
        model.load_weights(
            str(adapter_dir / "adapters.safetensors"), strict=False
        )

    unsteered_texts: list[str] = []
    steered_texts: list[str] = []
    unsteered_calls = 0
    steered_calls = 0
    for prompt in prompts:
        prompt_ids = mx.array(tokenizer.encode(prompt))
        base_text = _generate(
            model, tokenizer, prompt_ids, direction, 0.0, max_tokens
        )
        steered_text = _generate(
            model, tokenizer, prompt_ids, direction, alpha, max_tokens
        )
        unsteered_texts.append(base_text)
        steered_texts.append(steered_text)
        if _TOOL_CALL_RE.search(base_text):
            unsteered_calls += 1
        if _TOOL_CALL_RE.search(steered_text):
            steered_calls += 1

    return SteerResult(
        feature=feature,
        alpha=alpha,
        prompts=len(prompts),
        unsteered_tool_calls=unsteered_calls,
        steered_tool_calls=steered_calls,
        steered_texts=tuple(steered_texts),
        unsteered_texts=tuple(unsteered_texts),
    )


def report_ab(result: SteerResult, out_dir: Path) -> dict:
    """Write the A/B artifact and return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact = out_dir / f"steer-{stamp}.json"
    payload = {
        "feature": result.feature,
        "alpha": result.alpha,
        "prompts": result.prompts,
        "unsteered_tool_calls": result.unsteered_tool_calls,
        "steered_tool_calls": result.steered_tool_calls,
        "unsteered_texts": list(result.unsteered_texts),
        "steered_texts": list(result.steered_texts),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    artifact.write_text(json.dumps(payload, indent=2))
    print(f"steer artifact: {artifact}")
    print(
        f"feature {result.feature} alpha {result.alpha}: "
        f"tool calls {result.unsteered_tool_calls} -> "
        f"{result.steered_tool_calls} of {result.prompts}"
    )
    return payload
