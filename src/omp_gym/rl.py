"""REINFORCE with a group-mean baseline over live episodes.

For each iteration: sample a group of episodes for one task with
the served policy, score each with the task tests, and update the
policy toward episodes that beat the group mean. The advantage is
reward minus the group mean, divided by the group standard
deviation, applied to the summed logprob of EVERY captured
assistant turn of the rollout (multi-turn credit, sum objective -
no per-token mean normalization).

Rewards are improvement-over-baseline when the runner provides
`reward_improvement` (falling back to `reward_partial` then
`reward`): the runner scores each episode against the task's
pre-run test baseline, so the group compares how far each rollout
moved the task, not raw pass rates.

Rollouts run serially within an iteration, so the captures that
belong to rollout i are exactly captures[before:after], where
before/after are the capture-list lengths bracketing that
rollout's run_episode call. Failed or untrainable rollouts stay in
the group at reward 0.0, so the baseline mean and variance always
use the requested group size K; only rollouts with captured turns
contribute gradient.

The policy server restarts at every iteration, so each group is
sampled from the current policy weights. A run that produced no
update never writes adapter weights or a success ledger entry and
exits nonzero; it never prints `rl ok`.
"""

import fcntl
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import Adam
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
except ModuleNotFoundError:
    # Off-Mac machines have no mlx. The pure helpers stay
    # importable; run_rl fails at preflight.
    mx = nn = Adam = tree_flatten = tree_map = None
    load = linear_to_lora_layers = None

from . import serve as serve_mod
from . import shim as shim_mod
from .layers import (
    LayerSelection,
    adapter_layer_selection,
    layer_config_fields,
    mlx_num_layers,
)
from .ledger import append_entry
from .preflight import require_metal_gpu
from .runner import EpisodeFailure, run_episode
from .task import TaskLoadError, load_task, workspace_digest


class RlError(SystemExit):
    """Raised when the RL round cannot run or cannot update."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"rl failed: {reason}")


# The floor for the mean completion length, as a fraction of the
# first iteration's mean. A policy that collapses to near-empty
# completions still collects partial rewards, so reward curves do
# not show the failure; length does.
COMPLETION_COLLAPSE_RATIO = 0.3

# Absolute floor for the iteration-0 mean completion length. A
# first group already below this many tokens is collapsed by
# construction; the ratio check would only mask it.
COLLAPSE_ABSOLUTE_FLOOR = 8.0


@dataclass(frozen=True)
class IterationMetrics:
    """One iteration of grouped episodes and at most one update."""

    iteration: int
    task: str
    rewards: tuple[float, ...]
    mean_reward: float
    mean_completion_length: float | None
    n_failed: int
    n_captured_turns: int
    n_skipped_long: int = 0
    request_ids: tuple[str, ...] = ()
    loss: float | None = None


@dataclass(frozen=True)
class AdapterTopology:
    """LoRA shape derived from the input adapter_config.json."""

    rank: int
    scale: float
    dropout: float
    layer_selection: LayerSelection


def _validate_parameters(
    group_size: int,
    iterations: int,
    sample_temperature: float,
    kl_beta: float,
    max_seq_len: int,
    grad_clip: float,
    eval_episodes: int,
    port: int,
) -> None:
    """Reject settings that cannot produce a valid RL run."""
    if group_size < 2:
        raise RlError("group size must be at least 2")
    if iterations < 1:
        raise RlError("iterations must be at least 1")
    if not math.isfinite(sample_temperature) or sample_temperature < 0.0:
        raise RlError("sample temperature must be finite and nonnegative")
    if not math.isfinite(kl_beta) or kl_beta < 0.0:
        raise RlError("KL beta must be finite and nonnegative")
    if max_seq_len < 1:
        raise RlError("maximum sequence length must be positive")
    if not math.isfinite(grad_clip) or grad_clip < 0.0:
        raise RlError("gradient clip must be finite and nonnegative")
    if eval_episodes < 0:
        raise RlError("evaluation episode count must be nonnegative")
    if not 1 <= port <= 65535:
        raise RlError("port must be from 1 through 65535")


def _group_mean(rewards: list[float]) -> float:
    """Failure-inclusive group mean over the requested group size K."""
    return math.fsum(rewards) / len(rewards) if rewards else 0.0


def _normalized_advantages(rewards: list[float]) -> list[float]:
    """Group-mean baseline advantages, scaled by the group std.

    Failures sit in the group at reward 0.0, so K is always the
    denominator. A zero standard deviation would divide by zero;
    the divisor falls back to 1.0, which makes every advantage zero
    when all rewards are equal.
    """
    mean_reward = math.fsum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    group_std = variance**0.5
    if group_std == 0.0:
        group_std = 1.0
    return [(reward - mean_reward) / group_std for reward in rewards]


def _completion_length_collapsed(first_mean: float | None, current_mean: float) -> bool:
    """Decide the early stop for completion-length collapse.

    Iteration 0 has no ratio reference; an absolute floor flags it
    immediately when the policy already emits near-empty
    completions. A zero or negative reference gives no meaningful
    ratio and disables the relative check.
    """
    if first_mean is None:
        return 0.0 < current_mean < COLLAPSE_ABSOLUTE_FLOOR
    if first_mean <= 0.0:
        return False
    return current_mean < COMPLETION_COLLAPSE_RATIO * first_mean


def _task_schedule(task_dirs: Sequence[Path], iterations: int, seed: int) -> list[int]:
    """Round-robin task indexes, shuffled deterministically by seed.

    Every task runs once before any task repeats; the same seed
    always produces the same order.
    """
    order = list(range(len(task_dirs)))
    shuffler = random.Random(seed)  # noqa: S311 - seeded task schedule
    shuffler.shuffle(order)
    return [order[i % len(order)] for i in range(iterations)]


def _policy_gradient_terms(
    turn_logprobs: list[list[float]], advantages: list[float]
) -> list[float]:
    """One policy-gradient term per rollout.

    Every captured assistant turn of the rollout contributes its
    logprob to a single sequence SUM; there is no per-token mean
    normalization, so longer completions are not averaged away.
    """
    return [
        advantage * math.fsum(rollout_logprobs)
        for rollout_logprobs, advantage in zip(turn_logprobs, advantages, strict=True)
    ]


def _clip_scale(total_norm: float, max_norm: float) -> float:
    """Global-norm clip factor; 1.0 leaves the tree unchanged."""
    if max_norm <= 0.0 or total_norm <= max_norm or total_norm == 0.0:
        return 1.0
    return max_norm / total_norm


def _apply_grad_clip(
    grads: dict[str, list[float]], max_norm: float
) -> tuple[dict[str, list[float]], float]:
    """Clip a flat gradient dict to a global norm.

    Pure-python reference for the clip used on the mlx gradient
    tree; returns the clipped gradients and the pre-clip norm.
    """
    total = math.sqrt(math.fsum(g * g for grad in grads.values() for g in grad))
    scale = _clip_scale(total, max_norm)
    return (
        {name: [g * scale for g in grad] for name, grad in grads.items()},
        total,
    )


def _require_finite_metric(value: float, name: str) -> float:
    """Return one finite metric or stop before an optimizer update."""
    result = float(value)
    if not math.isfinite(result):
        raise RlError(f"{name} is not finite: {result}")
    return result


def _require_finite_tensors(flat_tensors, name: str) -> None:
    """Stop when one named MLX tensor is not finite."""
    for key, value in flat_tensors:
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise RlError(f"{name} tensor is not finite: {key}")


def _normalize_capture(entry: object) -> dict | None:
    """Validate one capture record in the current dict shape."""
    if not isinstance(entry, dict):
        return None
    if shim_mod.read_capture(entry) is None:
        return None
    return entry


def _window_captures(captured: list, before: int, after: int) -> list[dict]:
    """The captures of one rollout: exactly captured[before:after].

    Rollouts run serially, so ordinal slicing is the only sound
    attribution; prompt-substring matching cannot disambiguate
    repeated prompts and is removed.
    """
    out = []
    for entry in captured[before:after]:
        normalized = _normalize_capture(entry)
        if normalized is not None:
            out.append(normalized)
    return out


def _select_reward(episode) -> float:
    """Reward of one episode record for the group baseline.

    Improvement-over-baseline is the honest RL signal: it compares
    the episode against the task's pre-run test baseline. Older
    records without reward_improvement fall back to reward_partial,
    then reward.
    """
    for name in ("reward_improvement", "reward_partial", "reward"):
        value = getattr(episode, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _exceeds_seq_cap(
    prompt_ids: list[int], completion_ids: list[int], max_seq_len: int
) -> bool:
    """Skip turns whose prompt+completion would exceed the cap."""
    return len(prompt_ids) + len(completion_ids) > max_seq_len


def _read_adapter_topology(adapter_dir: Path) -> AdapterTopology:
    """Derive a complete LoRA topology from adapter_config.json."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise RlError(f"no adapter_config.json at {adapter_dir}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise RlError(f"{config_path} is not valid JSON: {error}") from None
    if not isinstance(config, dict):
        raise RlError(f"{config_path} must contain a JSON object")
    params = config.get("lora_parameters")
    if not isinstance(params, dict):
        raise RlError(f"{config_path} has no lora_parameters object")
    missing = [name for name in ("rank", "scale", "dropout") if name not in params]
    if "num_layers" not in config:
        missing.append("num_layers")
    if missing:
        raise RlError(
            f"{config_path} has an incomplete LoRA topology: " + ", ".join(missing)
        )
    rank = params["rank"]
    scale = params["scale"]
    dropout = params["dropout"]
    layer_selection = adapter_layer_selection(config)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise RlError(f"{config_path} has invalid rank: {rank!r}")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
        raise RlError(f"{config_path} has invalid scale: {scale!r}")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= dropout < 1
    ):
        raise RlError(f"{config_path} has invalid dropout: {dropout!r}")
    if layer_selection is None:
        raise RlError(f"{config_path} has inconsistent layer selection")
    return AdapterTopology(
        rank=rank,
        scale=float(scale),
        dropout=float(dropout),
        layer_selection=layer_selection,
    )


def _lora_config_dict(topology: AdapterTopology) -> dict:
    """linear_to_lora_layers config derived from the input adapter."""
    return {
        "rank": topology.rank,
        "scale": topology.scale,
        "dropout": topology.dropout,
    }


def _missing_lora_keys(weight_names: list[str], expected_names: list[str]) -> list[str]:
    """Expected trainable tensors absent from a saved weight file."""
    return sorted(set(expected_names) - set(weight_names))


def _verify_output_adapter(weights_path: Path, expected_names: list[str]) -> None:
    """Check that the save contains every trainable model tensor."""
    weights = mx.load(str(weights_path))
    missing = _missing_lora_keys(list(weights.keys()), expected_names)
    if missing:
        raise RlError(
            f"{weights_path.name} is missing {len(missing)} LoRA "
            f"tensors: {', '.join(missing[:5])}" + (", ..." if len(missing) > 5 else "")
        )


def _provider_config_path() -> Path:
    """The omp provider file the run snapshots, rewrites, restores."""
    return Path.home() / ".omp" / "agent" / "models.yml"


@contextmanager
def _provider_lock(models_yml: Path):
    """Exclusive advisory lock on the sibling .lock file."""
    lock_path = models_yml.with_name(models_yml.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _snapshot_provider_config(models_yml: Path) -> bytes | None:
    """Bytes of the provider file before the run, None when absent."""
    with _provider_lock(models_yml):
        return models_yml.read_bytes() if models_yml.is_file() else None


def _restore_provider_config(models_yml: Path, original: bytes | None) -> None:
    """Restore the pre-run bytes; remove a file the run created."""
    with _provider_lock(models_yml):
        if original is None:
            models_yml.unlink(missing_ok=True)
        else:
            models_yml.parent.mkdir(parents=True, exist_ok=True)
            models_yml.write_bytes(original)


def _register_provider(
    models_yml: Path, port: int, model_id: str, base_model: str
) -> None:
    """Register the policy server as an omp provider, or fail loudly.

    ensure_provider only rewrites a file omp-gym owns. A foreign
    file (return False) means episodes would silently run against
    the user's own providers instead of the policy, and an RL round
    under the wrong provider is never valid.
    """
    with _provider_lock(models_yml):
        in_place = serve_mod.ensure_provider(models_yml, port, model_id, base_model)
    if not in_place:
        raise RlError(
            f"{models_yml} is not managed by omp-gym and cannot point "
            "episodes at the policy server; merge the printed entry "
            "by hand or let omp-gym own the file"
        )


def _git_sha() -> str | None:
    """Best-effort HEAD sha; None when git is unavailable or fails."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed static argv
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed static argv
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    """Versions of the training stack that import on this machine."""
    versions: dict[str, str] = {}
    for display, module_name in (
        ("mlx", "mlx"),
        ("mlx-lm", "mlx_lm"),
        ("transformers", "transformers"),
    ):
        try:
            module = __import__(module_name)
        except ModuleNotFoundError:
            continue
        versions[display] = str(getattr(module, "__version__", "unknown"))
    return versions


def _task_digests(tasks: list) -> dict[str, str]:
    """Content digests of each task workspace for the report."""
    digests: dict[str, str] = {}
    for task in tasks:
        try:
            digests[task.name] = workspace_digest(task.workspace)
        except OSError:
            digests[task.name] = ""
    return digests


def _seed_rngs(seed: int) -> None:
    """Seed every RNG the round touches for reproducible sampling."""
    random.seed(seed)
    if mx is not None:
        mx.random.seed(seed)
    try:
        import numpy
    except ModuleNotFoundError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(seed)


def _stop_process(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    """Terminate, escalate to kill on timeout, and always reap."""
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _start_policy_server(
    base_model: str,
    adapter_dir: Path,
    port: int,
    capture: list[dict] | None = None,
    sample_temperature: float | None = None,
) -> tuple[object, object, int]:
    """Start the model server plus shim.

    Returns the server process, the shim server, and the port the
    shim actually binds. The shim binds directly instead of probing
    first, so no other process can win the port in between.

    sample_temperature is scoped to the serve subprocess environment
    (OMP_GYM_SAMPLE_TEMP) and handed to the shim explicitly; the
    caller's os.environ is never mutated.
    """
    httpd = None
    chosen = None
    for candidate in (port, port + 2, port + 4):
        try:
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", candidate),
                shim_mod.make_handler(
                    candidate + 1,
                    str(adapter_dir),
                    capture=capture,
                    sample_temp=sample_temperature,
                ),
            )
        except OSError:
            continue
        chosen = candidate
        break
    if httpd is None or chosen is None:
        raise RlError(f"no free port near {port}")
    backend_port = chosen + 1
    server_env = None
    if sample_temperature is not None:
        server_env = dict(os.environ)
        server_env["OMP_GYM_SAMPLE_TEMP"] = str(sample_temperature)
    server_proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            base_model,
            "--adapter-path",
            str(adapter_dir),
            "--port",
            str(backend_port),
        ],
        env=server_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", backend_port), timeout=1):
                return server_proc, httpd, chosen
        except OSError:
            time.sleep(1)
    _stop_process(server_proc)
    httpd.shutdown()
    httpd.server_close()
    raise RlError("policy server did not become ready")


def _encode_capture(entry: dict, tokenizer) -> tuple[list[int], list[int]] | None:
    """Encode one captured request and completion for the log-prob.

    The completion text is the raw upstream output the policy
    sampled, not a reconstruction from the session file. Residual
    approximation: re-encoding raw text approximates the sampled
    token ids; exact ids need server-side logprob capture, which
    the mlx-lm server does not expose.
    """
    messages = entry.get("messages") or []
    text = entry.get("text") or ""
    if not messages or not text:
        return None
    templated = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    if isinstance(templated, list):
        prompt_ids = (
            templated[0] if templated and isinstance(templated[0], list) else templated
        )
    else:
        ids = templated["input_ids"]
        prompt_ids = ids[0] if ids and isinstance(ids[0], list) else ids
    completion_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not completion_ids:
        return None
    return prompt_ids, completion_ids


def _load_tokenizer(base_model: str):
    """Tokenizer seam; the only piece of transformers RL uses."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model)


def _token_logprobs(model, prompt_ids, completion_ids, pad_to):
    """Masked per-token logprobs of the completion under the model.

    Padded to one fixed length: MLX compiles a graph per input
    shape, and shape-varying sequences would pay one full
    compilation per turn. The mask keeps padding out of the sum.
    """
    sequence = prompt_ids + completion_ids
    ids = mx.array(sequence + [0] * (pad_to - len(sequence)))[None]
    logits = model(ids)[0].astype(mx.float32)
    targets = ids[0, 1:]
    logprobs = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
    token_lp = mx.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
    positions = mx.arange(token_lp.shape[0])
    mask = (positions >= len(prompt_ids) - 1) & (positions < len(sequence) - 1)
    return token_lp * mask


def _build_model(
    base_model: str,
    adapter_dir: Path,
    topology: AdapterTopology,
    *,
    trainable: bool,
):
    """Base model with the input adapter applied on its topology."""
    model, _ = load(base_model)
    model.freeze()
    linear_to_lora_layers(
        model,
        mlx_num_layers(topology.layer_selection),
        _lora_config_dict(topology),
    )
    adapter_weights = mx.load(str(adapter_dir / "adapters.safetensors"))
    policy_keys = {name for name, _ in tree_flatten(model.parameters())}
    unmatched = [name for name in adapter_weights if name not in policy_keys]
    if unmatched:
        raise RlError(
            f"{len(unmatched)} adapter keys did not match the policy; "
            f"first: {unmatched[0]}"
        )
    model.load_weights(list(adapter_weights.items()), strict=False)
    if not trainable:
        model.freeze()
    return model


def run_rl(
    task_dir: Path | None = None,
    *,
    base_model: str,
    adapter_dir: Path,
    out_adapter: Path,
    group_size: int,
    iterations: int,
    port: int,
    runs_dir: Path,
    ledger_path: Path,
    task_dirs: Sequence[Path] | None = None,
    sample_temperature: float = 0.7,
    kl_beta: float = 0.0,
    seed: int = 0,
    max_seq_len: int = 4096,
    grad_clip: float = 1.0,
    eval_episodes: int = 0,
) -> dict:
    """Run a REINFORCE round and record it in the ledger.

    task_dirs may supply several tasks (scheduled round-robin,
    shuffled by seed); the singular task_dir alias covers the
    one-task case.
    """
    gpu = require_metal_gpu()
    dirs = (
        list(task_dirs)
        if task_dirs is not None
        else ([task_dir] if task_dir is not None else [])
    )
    if not dirs:
        raise RlError("no task directory given")
    tasks = []
    for path in dirs:
        task = load_task(path)
        if isinstance(task, TaskLoadError):
            raise RlError(f"bad task {task.path}: {task.reason}")
        tasks.append(task)
    if not (adapter_dir / "adapters.safetensors").is_file():
        raise RlError(f"no adapter at {adapter_dir}")
    if (out_adapter / "adapters.safetensors").is_file():
        raise RlError(
            f"{out_adapter} already contains adapters.safetensors; "
            "delete it or pass a fresh --out-adapter directory"
        )
    _validate_parameters(
        group_size,
        iterations,
        sample_temperature,
        kl_beta,
        max_seq_len,
        grad_clip,
        eval_episodes,
        port,
    )
    topology = _read_adapter_topology(adapter_dir)
    _seed_rngs(seed)

    # Rollout diversity: the served policy defaults to temperature 0,
    # which makes every episode in a group identical. The value is
    # scoped to the serve subprocess environment, the shim, and the
    # episode extra_env - never os.environ.
    model_id = f"omp-gym/{base_model}"
    provider_id = f"rl-{out_adapter.name}"
    models_yml = _provider_config_path()
    provider_snapshot = _snapshot_provider_config(models_yml)

    policy = None
    reference = None

    def load_policy():
        nonlocal policy
        if policy is None:
            policy = _build_model(base_model, adapter_dir, topology, trainable=True)
        return policy

    def load_reference():
        nonlocal reference
        if reference is None:
            reference = _build_model(base_model, adapter_dir, topology, trainable=False)
            reference.freeze()
        return reference

    tokenizer_holder: list = []

    def get_tokenizer():
        if not tokenizer_holder:
            tokenizer_holder.append(_load_tokenizer(base_model))
        return tokenizer_holder[0]

    optimizer = None
    rounds: list[IterationMetrics] = []
    first_mean_length: float | None = None
    stopped_early = False
    stop_reason: str | None = None
    started = time.monotonic()
    schedule = _task_schedule(dirs, iterations, seed)

    try:
        for iteration in range(1, iterations + 1):
            task = tasks[schedule[iteration - 1]]
            print(
                f"iteration {iteration}: serving policy and sampling "
                f"{group_size} episodes on {task.name}"
            )
            # The run starts with a fresh out_adapter (guarded above),
            # so a file there was written by this run: serve the
            # newest weights and stay on-policy across iterations.
            source_adapter = (
                out_adapter
                if (out_adapter / "adapters.safetensors").is_file()
                else adapter_dir
            )
            captured: list[dict] = []
            server_proc, httpd, chosen_port = _start_policy_server(
                base_model,
                source_adapter,
                port,
                capture=captured,
                sample_temperature=sample_temperature,
            )
            episodes: list = []
            windows: list[tuple[int, int]] = []
            try:
                _register_provider(models_yml, chosen_port, provider_id, base_model)
                for _ in range(group_size):
                    before = len(captured)
                    episodes.append(
                        run_episode(
                            task,
                            runs_dir,
                            model_id,
                            extra_env={"OMP_GYM_SAMPLE_TEMP": str(sample_temperature)},
                        )
                    )
                    windows.append((before, len(captured)))
            finally:
                httpd.shutdown()
                httpd.server_close()
                _stop_process(server_proc)

            group_rewards: list[float] = []
            turns_by_rollout: list[list[tuple[list[int], list[int]]]] = []
            request_ids: list[str] = []
            n_failed = 0
            n_skipped_long = 0
            for rollout, (episode, (before, after)) in enumerate(
                zip(episodes, windows, strict=True), start=1
            ):
                if isinstance(episode, EpisodeFailure):
                    group_rewards.append(0.0)
                    turns_by_rollout.append([])
                    n_failed += 1
                    print(
                        f"iteration {iteration} rollout {rollout} failed: "
                        f"{episode.reason}; reward 0.0"
                    )
                    continue
                encoded: list[tuple[list[int], list[int]]] = []
                for entry in _window_captures(captured, before, after):
                    pair = _encode_capture(entry, get_tokenizer())
                    if pair is None:
                        continue
                    if _exceeds_seq_cap(pair[0], pair[1], max_seq_len):
                        n_skipped_long += 1
                        continue
                    request_id = entry.get("request_id")
                    if request_id is not None:
                        request_ids.append(str(request_id))
                    encoded.append(pair)
                if not encoded:
                    group_rewards.append(0.0)
                    turns_by_rollout.append([])
                    n_failed += 1
                    print(
                        f"iteration {iteration} rollout {rollout} "
                        "untrainable: no usable captured turns; "
                        "reward 0.0"
                    )
                    continue
                group_rewards.append(_select_reward(episode))
                turns_by_rollout.append(encoded)

            mean_reward = _group_mean(group_rewards)
            trainable = [turns for turns in turns_by_rollout if turns]
            mean_length = (
                sum(len(c) for turns in trainable for _, c in turns)
                / sum(len(turns) for turns in trainable)
                if trainable
                else None
            )
            n_captured_turns = sum(len(turns) for turns in turns_by_rollout)

            if mean_length is not None and _completion_length_collapsed(
                first_mean_length, mean_length
            ):
                rounds.append(
                    IterationMetrics(
                        iteration=iteration,
                        task=task.name,
                        rewards=tuple(group_rewards),
                        mean_reward=mean_reward,
                        mean_completion_length=mean_length,
                        n_failed=n_failed,
                        n_captured_turns=n_captured_turns,
                        n_skipped_long=n_skipped_long,
                        request_ids=tuple(request_ids),
                        loss=None,
                    )
                )
                stopped_early = True
                stop_reason = "completion length collapse"
                print(
                    f"iteration {iteration}: mean completion length "
                    f"{mean_length:.1f} collapsed; stopping early"
                )
                break
            if first_mean_length is None and mean_length is not None:
                first_mean_length = mean_length

            variance = sum((r - mean_reward) ** 2 for r in group_rewards) / len(
                group_rewards
            )
            print(
                f"iteration {iteration}: rewards "
                f"{[round(r, 2) for r in group_rewards]} "
                f"mean {mean_reward:.2f} (n_failed {n_failed})"
            )

            if variance == 0.0 or not trainable:
                rounds.append(
                    IterationMetrics(
                        iteration=iteration,
                        task=task.name,
                        rewards=tuple(group_rewards),
                        mean_reward=mean_reward,
                        mean_completion_length=mean_length,
                        n_failed=n_failed,
                        n_captured_turns=n_captured_turns,
                        n_skipped_long=n_skipped_long,
                        request_ids=tuple(request_ids),
                        loss=None,
                    )
                )
                if not trainable:
                    print("no trainable turns; no update this iteration")
                else:
                    print("no group variance; no update this iteration")
                continue

            advantages = _normalized_advantages(group_rewards)
            batch = [
                (turns_by_rollout[i], advantages[i])
                for i in range(group_size)
                if turns_by_rollout[i]
            ]
            longest = max(
                len(prompt) + len(completion)
                for turns, _ in batch
                for prompt, completion in turns
            )
            pad_to = ((longest + 127) // 128) * 128

            del episodes, windows
            if mx is not None:
                mx.clear_cache()
            reference_model = load_reference() if kl_beta > 0.0 else None

            def pg_loss(
                model,
                batch_items,
                pad_to=pad_to,
                reference_model=reference_model,
            ):
                terms = []
                for rollout_turns, adv in batch_items:
                    rollout_lp = None
                    rollout_kl = None
                    for prompt_ids, completion_ids in rollout_turns:
                        pol_lp = _token_logprobs(
                            model, prompt_ids, completion_ids, pad_to
                        )
                        total = pol_lp.sum()
                        rollout_lp = total if rollout_lp is None else rollout_lp + total
                        if reference_model is not None:
                            ref_lp = _token_logprobs(
                                reference_model,
                                prompt_ids,
                                completion_ids,
                                pad_to,
                            )
                            kl_residual = ref_lp - pol_lp
                            kl = (mx.exp(kl_residual) - kl_residual - 1).sum()
                            rollout_kl = kl if rollout_kl is None else rollout_kl + kl
                    term = adv * rollout_lp
                    if rollout_kl is not None:
                        term = term - kl_beta * rollout_kl
                    terms.append(term)
                return -mx.stack(terms).mean()

            current = load_policy()
            loss_and_grad = nn.value_and_grad(current, pg_loss)
            loss, grads = loss_and_grad(current, batch)
            value = _require_finite_metric(
                loss.item(), f"policy loss at iteration {iteration}"
            )
            flat_grads = tree_flatten(grads)
            _require_finite_tensors(flat_grads, f"gradients at iteration {iteration}")
            if grad_clip > 0.0:
                total_norm = _require_finite_metric(
                    math.sqrt(
                        math.fsum(float((grad * grad).sum()) for _, grad in flat_grads)
                    ),
                    f"gradient norm at iteration {iteration}",
                )
                scale = _clip_scale(total_norm, grad_clip)
                if scale < 1.0:
                    grads = tree_map(lambda grad, scale=scale: grad * scale, grads)
                    print(
                        f"iteration {iteration}: grad norm {total_norm:.3f} "
                        f"clipped to {grad_clip:.3f}"
                    )
            if optimizer is None:
                if Adam is None:
                    raise RlError("MLX optimizer is not available after preflight")
                optimizer = Adam(learning_rate=5e-6)
            optimizer.update(current, grads)
            mx.eval(current.trainable_parameters(), optimizer.state)
            _require_finite_tensors(
                tree_flatten(current.trainable_parameters()),
                f"policy parameters after iteration {iteration}",
            )
            print(f"iteration {iteration}: pg loss {value:.5f}")

            out_adapter.mkdir(parents=True, exist_ok=True)
            weights_path = out_adapter / "adapters.safetensors"
            trainable_weights = dict(tree_flatten(current.trainable_parameters()))
            if not trainable_weights:
                raise RlError("the updated policy has no trainable tensors")
            _require_finite_tensors(list(trainable_weights.items()), "output adapter")
            mx.save_safetensors(str(weights_path), trainable_weights)
            _verify_output_adapter(weights_path, list(trainable_weights))
            (out_adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "fine_tune_type": "lora",
                        "method": "reinforce",
                        "model": base_model,
                        "lora_parameters": _lora_config_dict(topology),
                        **layer_config_fields(topology.layer_selection),
                        "resume_adapter_file": str(adapter_dir),
                    },
                    indent=2,
                )
            )
            rounds.append(
                IterationMetrics(
                    iteration=iteration,
                    task=task.name,
                    rewards=tuple(group_rewards),
                    mean_reward=mean_reward,
                    mean_completion_length=mean_length,
                    n_failed=n_failed,
                    n_captured_turns=n_captured_turns,
                    n_skipped_long=n_skipped_long,
                    request_ids=tuple(request_ids),
                    loss=value,
                )
            )

        updates = sum(1 for r in rounds if r.loss is not None)
        eval_summary = None
        if updates > 0 and eval_episodes > 0:
            print(f"evaluating output adapter on {eval_episodes} episodes")
            eval_summary = _run_evaluation(
                base_model=base_model,
                out_adapter=out_adapter,
                port=port,
                tasks=tasks,
                eval_episodes=eval_episodes,
                sample_temperature=sample_temperature,
                models_yml=models_yml,
                provider_id=provider_id,
                runs_dir=runs_dir,
                model_id=model_id,
            )
    finally:
        _restore_provider_config(models_yml, provider_snapshot)

    updates = sum(1 for r in rounds if r.loss is not None)
    elapsed = time.monotonic() - started
    summary = {
        "tasks": [task.name for task in tasks],
        "base_model": base_model,
        "group_size": group_size,
        "iterations": iterations,
        "updates": updates,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "n_failed": sum(r.n_failed for r in rounds),
        "n_skipped_long": sum(r.n_skipped_long for r in rounds),
        "rounds": [
            {
                "iteration": r.iteration,
                "task": r.task,
                "rewards": list(r.rewards),
                "mean_reward": r.mean_reward,
                "mean_completion_length": r.mean_completion_length,
                "n_failed": r.n_failed,
                "n_captured_turns": r.n_captured_turns,
                "n_skipped_long": r.n_skipped_long,
                "request_ids": list(r.request_ids),
                "loss": r.loss,
            }
            for r in rounds
        ],
        "mean_reward_first": rounds[0].mean_reward,
        "mean_reward_last": rounds[-1].mean_reward,
        "elapsed_seconds": round(elapsed, 1),
        "out_adapter": str(out_adapter),
        "reproducibility": {
            "seed": seed,
            "git_sha": _git_sha(),
            "python_version": platform.python_version(),
            "packages": _package_versions(),
            "task_digests": _task_digests(tasks),
            "model": base_model,
        },
    }
    if eval_summary is not None:
        summary["eval"] = eval_summary
    out_adapter.mkdir(parents=True, exist_ok=True)
    (out_adapter / "rl_report.json").write_text(json.dumps(summary, indent=2))
    if updates == 0:
        no_turns = all(r.n_captured_turns == 0 for r in rounds)
        reason = (
            "no captured assistant turns in any iteration"
            if no_turns
            else "zero advantage variance in every iteration"
        )
        raise RlError(f"no policy update over {iterations} iteration(s): {reason}")
    append_entry(
        ledger_path,
        kind="rl",
        config={
            "tasks": [task.name for task in tasks],
            "model": base_model,
            "adapter": str(out_adapter),
            "group_size": group_size,
            "iterations": iterations,
            "method": "reinforce",
            "seed": seed,
            "kl_beta": kl_beta,
            "grad_clip": grad_clip,
            "max_seq_len": max_seq_len,
            "sample_temperature": sample_temperature,
        },
        metrics={
            "mean_reward_first": summary["mean_reward_first"],
            "mean_reward_last": summary["mean_reward_last"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "updates": updates,
            "n_failed": summary["n_failed"],
        },
        artifacts={"rl_report": str(out_adapter / "rl_report.json")},
    )
    print(
        f"rl ok: mean reward {summary['mean_reward_first']:.2f} -> "
        f"{summary['mean_reward_last']:.2f} on {gpu.device_name}"
    )
    return summary


def _run_evaluation(
    base_model: str,
    out_adapter: Path,
    port: int,
    tasks: list,
    eval_episodes: int,
    sample_temperature: float,
    models_yml: Path,
    provider_id: str,
    runs_dir: Path,
    model_id: str,
) -> dict:
    """Serve the output adapter and run evaluation episodes.

    A pass is one completed episode whose verified reward is 1.0.
    The evaluation run only records results. It never updates
    weights.
    """
    server_proc, httpd, chosen_port = _start_policy_server(
        base_model,
        out_adapter,
        port,
        sample_temperature=sample_temperature,
    )
    passed = 0
    completed = 0
    try:
        _register_provider(models_yml, chosen_port, provider_id, base_model)
        for index in range(eval_episodes):
            task = tasks[index % len(tasks)]
            result = run_episode(
                task,
                runs_dir,
                model_id,
                extra_env={"OMP_GYM_SAMPLE_TEMP": str(sample_temperature)},
            )
            if isinstance(result, EpisodeFailure):
                continue
            completed += 1
            if result.reward >= 1.0:
                passed += 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        _stop_process(server_proc)
    return {
        "episodes": eval_episodes,
        "completed": completed,
        "passed": passed,
        "pass_rate": round(passed / eval_episodes, 4),
    }
