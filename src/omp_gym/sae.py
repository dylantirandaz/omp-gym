"""Tiny sparse autoencoder over residual-stream activations.

Research preview. Activations are captured at one decoder layer
while the local model reads exported training samples. A small SAE
is trained with an L1 sparsity penalty. The report lists, for the
most active features, the samples that activate them most. Samples
are named by sha256 and length only — raw training text never
appears in a report.
"""

import hashlib
import json
import time
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import Adam
    from mlx_lm import load
except ModuleNotFoundError:
    # Off-Mac machines have no mlx. The pure helpers and the
    # constants stay importable; train_sae fails at preflight.
    mx = nn = Adam = load = None

from .preflight import require_metal_gpu

SAE_LAYER = 12
SAE_FEATURES = 4096
SAE_STEPS = 400
SAE_BATCH = 4096
SAE_L1 = 3e-3
SAE_LR = 1e-3
SAMPLE_TOKEN_CAP = 512
MAX_SAMPLES = 400


class SaeError(SystemExit):
    """Raised when SAE training cannot run or did not learn."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"sae failed: {reason}")


def _excerpt_fingerprint(text: str) -> dict[str, object]:
    """sha256 and length stand in for a raw excerpt in artifacts.

    Reports never carry raw training text. An auditor with the
    dataset recomputes the digest to find the exact sample.
    """
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "length": len(text),
    }


def _captured_forward(model, ids, capture_layer):
    """Run a forward pass and return hidden states at one layer."""
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(ids.shape[1]).astype(
        h.dtype
    )
    captured = None
    for index, layer in enumerate(inner.layers):
        h = layer(h, mask=mask, cache=None)
        if index == capture_layer:
            captured = h
    if captured is None:
        raise SaeError(f"layer {capture_layer} out of range")
    return captured[0]


def _collect_activations(
    model,
    tokenizer,
    data_dir: Path,
    *,
    layer: int = SAE_LAYER,
    samples: int = MAX_SAMPLES,
):
    """Forward dataset samples and gather activations at one layer.

    Returns the stacked activations plus one fingerprint per kept
    sample; the raw sample text never leaves this function.
    """
    activations = []
    excerpts = []
    lines = (data_dir / "train.jsonl").read_text().splitlines()
    for line in lines[:samples]:
        record = json.loads(line)
        text = record["messages"][-1]["content"]
        ids = tokenizer.encode(text)[:SAMPLE_TOKEN_CAP]
        if len(ids) < 8:
            continue
        hidden = _captured_forward(model, mx.array(ids)[None], layer)
        activations.append(hidden[1:])
        excerpts.append(_excerpt_fingerprint(text))
    return mx.concatenate(activations), excerpts


def _token_boundaries(data_dir: Path, tokenizer, *, samples: int = MAX_SAMPLES):
    """Map flat token indices back to their dataset sample."""
    boundaries = []
    cursor = 0
    visible = 0
    lines = (data_dir / "train.jsonl").read_text().splitlines()[:samples]
    for line in lines:
        record = json.loads(line)
        text = record["messages"][-1]["content"]
        ids = tokenizer.encode(text)[:SAMPLE_TOKEN_CAP]
        if len(ids) < 8:
            continue
        end = cursor + len(ids) - 1
        boundaries.append((cursor, end, visible))
        cursor = end
        visible += 1
    return boundaries


def _sample_hits(z, feature, boundaries, excerpts, top_n=3):
    """Find the samples where one feature activates most.

    Each hit names the sample by sha256 and length, never by its
    raw text.
    """
    column = z[:, feature]
    order = mx.argsort(-column)[: top_n * 4].tolist()
    hits = []
    seen = set()
    for token_index in order:
        value = float(column[token_index])
        if value <= 0:
            break
        for start, end, excerpt_index in boundaries:
            if start <= token_index < end:
                if excerpt_index in seen:
                    break
                seen.add(excerpt_index)
                fingerprint = excerpts[excerpt_index]
                hits.append(
                    {
                        "activation": round(value, 3),
                        "excerpt_sha256": fingerprint["sha256"],
                        "excerpt_length": fingerprint["length"],
                    }
                )
                break
        if len(hits) >= top_n:
            break
    return hits


def _feature_report(z, boundaries, excerpts, activity, feature_order):
    """Build the per-feature top-sample report."""
    return [
        {
            "feature": int(feature),
            "activity_rate": round(float(activity[feature]), 4),
            "top_samples": _sample_hits(z, int(feature), boundaries, excerpts),
        }
        for feature in feature_order
    ]


def train_sae(
    data_dir: Path,
    model_id: str,
    adapter_dir: Path | None,
    out_dir: Path,
    *,
    layer: int = SAE_LAYER,
    features: int = SAE_FEATURES,
    samples: int = MAX_SAMPLES,
    steps: int = SAE_STEPS,
    l1: float = SAE_L1,
    seed: int = 0,
) -> dict:
    """Train the SAE and write a feature report. Returns metrics."""
    gpu = require_metal_gpu()
    if adapter_dir is None:
        model, tokenizer = load(model_id)
    else:
        if not (adapter_dir / "adapters.safetensors").is_file():
            raise SaeError(f"no adapter weights at {adapter_dir}")
        model, tokenizer = load(model_id, adapter_path=str(adapter_dir))
    mx.random.seed(seed)
    activations, excerpts = _collect_activations(
        model, tokenizer, data_dir, layer=layer, samples=samples
    )
    count, dim = activations.shape
    print(f"activations: {count} tokens x {dim} dims at layer {layer}")

    encoder = nn.Linear(dim, features)
    decoder = nn.Linear(features, dim)
    params = dict(
        enc_w=encoder.weight,
        enc_b=encoder.bias,
        dec_w=decoder.weight,
        dec_b=decoder.bias,
    )
    optimizer = Adam(learning_rate=SAE_LR)

    def loss_fn(params, batch):
        z = mx.maximum(batch @ params["enc_w"].T + params["enc_b"], 0)
        recon = z @ params["dec_w"].T + params["dec_b"]
        mse = ((recon - batch) ** 2).mean()
        sparsity = mx.abs(z).mean()
        return mse + l1 * sparsity

    value_and_grad = mx.value_and_grad(loss_fn)
    losses = []
    for step in range(1, steps + 1):
        index = mx.random.randint(0, count, (SAE_BATCH,))
        batch = activations[index]
        loss, grads = value_and_grad(params, batch)
        optimizer.update(params, grads)
        mx.eval(loss, params, optimizer.state)
        value = float(loss)
        if value != value:
            raise SaeError(f"NaN loss at step {step}")
        losses.append(value)
        if step % 100 == 0:
            print(f"step {step}: loss {value:.5f}")
    if losses[-1] >= losses[0]:
        raise SaeError(f"loss did not go down: {losses[0]} -> {losses[-1]}")

    z = mx.maximum(activations @ params["enc_w"].T + params["enc_b"], 0)
    mx.eval(z)
    activity = (z > 0).astype(mx.float32).mean(axis=0)
    feature_order = mx.argsort(-activity)[:64].tolist()
    boundaries = _token_boundaries(data_dir, tokenizer, samples=samples)
    features_report = _feature_report(z, boundaries, excerpts, activity, feature_order)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"sae-weights-{stamp}.safetensors"
    mx.save_safetensors(
        str(weights_path),
        {
            "enc_w": params["enc_w"],
            "enc_b": params["enc_b"],
            "dec_w": params["dec_w"],
            "dec_b": params["dec_b"],
        },
    )
    artifact = out_dir / f"sae-{stamp}.json"
    artifact.write_text(
        json.dumps(
            {
                "model": model_id,
                "adapter": str(adapter_dir) if adapter_dir else None,
                "layer": layer,
                "tokens": count,
                "features": features,
                "samples": samples,
                "steps": steps,
                "l1": l1,
                "seed": seed,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "device": gpu.device_name,
                "report": features_report,
                "weights": str(weights_path),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
    )
    print(f"sae artifact: {artifact}")
    print(f"loss {losses[0]:.5f} -> {losses[-1]:.5f} on {gpu.device_name}")
    return {
        "tokens": count,
        "features": features,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "artifact": str(artifact),
        "weights": str(weights_path),
    }
