"""Logit-lens inspection of a local model with an optional adapter.

For a given prompt, run a full forward pass while capturing the
hidden state after every decoder layer. Each layer's hidden state
is projected through the final norm and the (tied) embedding head
to get that layer's prediction. The result shows how the model's
next-token prediction forms layer by layer.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from .preflight import require_metal_gpu


class InspectError(SystemExit):
    """Raised when inspection cannot run."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"inspect failed: {reason}")


def _layer_predictions(model, ids, top_k):
    """Capture per-layer hidden states and decode each into tokens."""
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        ids.shape[1]
    ).astype(h.dtype)
    per_layer = []
    for layer in inner.layers:
        h = layer(h, mask=mask, cache=None)
        normed = inner.norm(h)
        if hasattr(model, "lm_head"):
            logits = model.lm_head(normed)
        else:
            logits = inner.embed_tokens.as_linear(normed)
        top = mx.argsort(logits, axis=-1)[..., -top_k:]
        per_layer.append(top)
    mx.eval(*per_layer)
    return per_layer


def run_lens(
    prompt: str,
    model_id: str,
    adapter_dir: Path | None,
    top_k: int,
    out_dir: Path,
) -> dict:
    """Compute the lens and write the artifact. Returns the result."""
    gpu = require_metal_gpu()
    model, tokenizer = load(model_id)
    if adapter_dir is not None:
        if not (adapter_dir / "adapters.safetensors").is_file():
            raise InspectError(f"no adapter at {adapter_dir}")
        model.load_weights(
            str(adapter_dir / "adapters.safetensors"), strict=False
        )
    ids = mx.array(tokenizer.encode(prompt))[None]
    per_layer = _layer_predictions(model, ids, top_k)
    n_layers = len(per_layer)
    top_by_layer = []
    for layer_index, top in enumerate(per_layer):
        token_ids = top[0, -1, :].tolist()
        top_by_layer.append(
            [tokenizer.decode([int(t)]) for t in reversed(token_ids)]
        )
    result = {
        "prompt": prompt,
        "model": model_id,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "device": gpu.device_name,
        "layers": n_layers,
        "top_by_layer": top_by_layer,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"lens-{time.strftime('%Y%m%d-%H%M%S')}.json"
    artifact.write_text(json.dumps(result, indent=2))
    print(f"lens artifact: {artifact}")
    print(f"device: {gpu.device_name}, layers: {n_layers}")
    for layer_index, tokens in enumerate(top_by_layer):
        shown = ", ".join(repr(t) for t in tokens)
        print(f"layer {layer_index:2d}: {shown}")
    return result
