"""Group-relative policy gradient over live episodes.

For each iteration: sample a group of episodes for one task with
the served policy, score each with the task tests, and update the
policy toward episodes that beat the group mean. This is GRPO
without the KL term: reward minus group baseline, applied to the
logprob of each episode's first assistant turn.

The server that produces episodes restarts at every iteration, so
each group is sampled from the current policy weights.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from . import serve as serve_mod
from . import shim as shim_mod
from .dpo import LORA_CONFIG, LORA_NUM_LAYERS, _completion_logprob
from .export import SYSTEM_PROMPT, _render_messages
from .ledger import append_entry
from .preflight import require_metal_gpu
from .runner import EpisodeFailure, run_episode
from .task import load_task, TaskLoadError
from .trajectory import AssistantStep, parse_session


class RlError(SystemExit):
    """Raised when the RL round cannot run."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"rl failed: {reason}")


@dataclass(frozen=True)
class IterationMetrics:
    """One iteration of grouped episodes and one update."""

    iteration: int
    rewards: tuple[float, ...]
    mean_reward: float
    loss: float | None


def _start_policy_server(
    base_model: str, adapter_dir: Path, port: int
) -> tuple[object, object, int]:
    """Start the model server plus shim.

    Returns the server process, the shim server, and the port the
    shim actually binds (a fallback port may be chosen when the
    requested one is taken).
    """
    import subprocess
    import sys

    import socket as _socket

    chosen = None
    for candidate in (port, port + 2, port + 4):
        probe = _socket.socket()
        try:
            probe.bind(("127.0.0.1", candidate))
            chosen = candidate
        except OSError:
            pass
        finally:
            probe.close()
        if chosen is not None:
            break
    if chosen is None:
        raise RlError(f"no free port near {port}")
    backend_port = chosen + 1
    server_proc = subprocess.Popen(
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", chosen), shim_mod.make_handler(backend_port)
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    deadline = time.monotonic() + 120
    import socket

    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                ("127.0.0.1", backend_port), timeout=1
            ):
                return server_proc, httpd, chosen
        except OSError:
            time.sleep(1)
    server_proc.terminate()
    raise RlError("policy server did not become ready")


def _first_turn_completion(session_file: Path, prompt: str, tokenizer):
    """Render (prompt_ids, completion_ids) for an episode's first turn."""
    trajectory = parse_session(session_file)
    for step in trajectory.steps:
        if isinstance(step, AssistantStep) and (step.text or step.tool_calls):
            messages = _render_messages(trajectory, prompt)
            turn = next(
                (m for m in messages if m["role"] == "assistant"), None
            )
            if turn is None:
                return None
            templated = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
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
            completion_ids = tokenizer(
                turn["content"], add_special_tokens=False
            ).input_ids
            return prompt_ids, completion_ids
    return None


def run_rl(
    task_dir: Path,
    base_model: str,
    adapter_dir: Path,
    out_adapter: Path,
    group_size: int,
    iterations: int,
    port: int,
    runs_dir: Path,
    ledger_path: Path,
) -> dict:
    """Run a GRPO round and record it in the ledger."""
    gpu = require_metal_gpu()
    task = load_task(task_dir)
    if isinstance(task, TaskLoadError):
        raise RlError(f"bad task {task.path}: {task.reason}")
    if not (adapter_dir / "adapters.safetensors").is_file():
        raise RlError(f"no adapter at {adapter_dir}")

    import os

    # Rollout diversity: the served policy defaults to temperature 0,
    # which makes every episode in a group identical.
    os.environ["OMP_GYM_SAMPLE_TEMP"] = "0.8"



    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    policy = None

    def load_policy():
        nonlocal policy
        if policy is not None:
            return policy
        loaded, _ = load(base_model)
        loaded.freeze()
        linear_to_lora_layers(loaded, LORA_NUM_LAYERS, LORA_CONFIG)
        source = (
            out_adapter
            if (out_adapter / "adapters.safetensors").is_file()
            else adapter_dir
        )
        loaded.load_weights(
            str(source / "adapters.safetensors"), strict=False
        )
        policy = loaded
        return loaded

    optimizer = Adam(learning_rate=5e-6)
    rounds: list[IterationMetrics] = []
    started = time.monotonic()

    for iteration in range(1, iterations + 1):
        print(f"iteration {iteration}: serving policy and sampling {group_size} episodes")
        server_proc, httpd, chosen_port = _start_policy_server(
            base_model, out_adapter if (out_adapter / "adapters.safetensors").is_file() else adapter_dir, port
        )
        serve_mod.ensure_provider(
            Path.home() / ".omp" / "agent" / "models.yml",
            chosen_port,
            f"rl-{out_adapter.name}",
            base_model,
        )
        try:
            with ThreadPoolExecutor(max_workers=group_size) as pool:
                futures = [
                    pool.submit(
                        run_episode,
                        task,
                        runs_dir,
                        f"omp-gym/{base_model}",
                    )
                    for _ in range(group_size)
                ]
                episodes = [future.result() for future in futures]
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_proc.terminate()
            server_proc.wait(timeout=15)

        rewards = []
        completions = []
        for episode in episodes:
            if isinstance(episode, EpisodeFailure):
                continue
            reward = (
                episode.reward_partial
                if episode.reward_partial is not None
                else episode.reward
            )
            prompt_file = Path(episode.episode_dir) / "prompt.txt"
            prompt = (
                prompt_file.read_text().strip()
                if prompt_file.is_file()
                else task.prompt
            )
            pair = _first_turn_completion(
                Path(episode.session_file), prompt, tokenizer
            )
            if pair is None:
                continue
            rewards.append(reward)
            completions.append(pair)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        baseline = mean_reward
        advantages = [r - baseline for r in rewards]
        print(
            f"iteration {iteration}: rewards "
            f"{[round(r, 2) for r in rewards]} mean {mean_reward:.2f}"
        )

        if not completions or all(a == 0 for a in advantages):
            rounds.append(
                IterationMetrics(
                    iteration=iteration,
                    rewards=tuple(rewards),
                    mean_reward=mean_reward,
                    loss=None,
                )
            )
            print("no group variance; no update this iteration")
            continue

        longest = max(
            len(p) + len(c) for p, c in completions
        )
        pad_to = ((longest + 127) // 128) * 128

        del episodes
        mx.clear_cache()

        def pg_loss(model, batch, advs):
            terms = []
            for (prompt_ids, completion_ids), adv in zip(batch, advs):
                lp = _completion_logprob(
                    model, prompt_ids, completion_ids, pad_to
                )
                terms.append(adv * lp / max(1, len(completion_ids)))
            return -mx.stack(terms).mean()

        current = load_policy()
        loss_and_grad = nn.value_and_grad(current, pg_loss)
        loss, grads = loss_and_grad(current, completions, advantages)
        optimizer.update(current, grads)
        mx.eval(loss, current.trainable_parameters(), optimizer.state)
        value = float(loss)
        if value != value:
            raise RlError(f"NaN loss at iteration {iteration}")
        print(f"iteration {iteration}: pg loss {value:.5f}")

        out_adapter.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(out_adapter / "adapters.safetensors"),
            dict(tree_flatten(current.trainable_parameters())),
        )
        (out_adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "method": "grpo",
                    "model": base_model,
                    "lora_parameters": LORA_CONFIG,
                    "num_layers": LORA_NUM_LAYERS,
                    "resume_adapter_file": str(adapter_dir),
                },
                indent=2,
            )
        )
        rounds.append(
            IterationMetrics(
                iteration=iteration,
                rewards=tuple(rewards),
                mean_reward=mean_reward,
                loss=value,
            )
        )

    elapsed = time.monotonic() - started
    summary = {
        "task": task.name,
        "base_model": base_model,
        "group_size": group_size,
        "iterations": iterations,
        "rounds": [
            {
                "iteration": r.iteration,
                "rewards": list(r.rewards),
                "mean_reward": r.mean_reward,
                "loss": r.loss,
            }
            for r in rounds
        ],
        "mean_reward_first": rounds[0].mean_reward,
        "mean_reward_last": rounds[-1].mean_reward,
        "elapsed_seconds": round(elapsed, 1),
        "out_adapter": str(out_adapter),
    }
    out_adapter.mkdir(parents=True, exist_ok=True)
    (out_adapter / "rl_report.json").write_text(
        json.dumps(summary, indent=2)
    )
    append_entry(
        ledger_path,
        kind="rl",
        config={
            "task": task.name,
            "model": base_model,
            "adapter": str(out_adapter),
            "group_size": group_size,
            "iterations": iterations,
        },
        metrics={
            "mean_reward_first": summary["mean_reward_first"],
            "mean_reward_last": summary["mean_reward_last"],
            "elapsed_seconds": summary["elapsed_seconds"],
        },
        artifacts={"rl_report": str(out_adapter / "rl_report.json")},
    )
    print(
        f"rl ok: mean reward {summary['mean_reward_first']:.2f} -> "
        f"{summary['mean_reward_last']:.2f} on {gpu.device_name}"
    )
    return summary
