---
name: omp-gym
description: Operate the Prime Verifiers v1 OMP coding environment. Use it to mint sealed repository tasks from previous OMP sessions, run sealed coding evaluations and model benchmarks, export exact Prime traces, train MLX LoRA adapters on Apple Metal, and compare an adapter with its fixed baseline.
---

# omp-gym operator

Use `omp-coding` version `1.2.0`. Use Prime Verifiers v1 for all rollouts. Do
not use the removed local runner, model shim, ledger, dashboard, or custom RL
commands.

## Mint tasks from sessions

OMP sessions live under `~/.omp/agent/sessions`. Scan first, then mint:

```sh
uv run --package omp-coding omp-coding-mint scan
uv run --package omp-coding omp-coding-mint mint --output tasks/minted
```

The minter keeps an episode only when the repository still exists on this
host, a commit matches the pre-edit file contents, and the session recorded a
test command that failed and then passed after the edits. It builds a Docker
image per task and proves the reference patch flips the sealed tests. Read
`mint-report.json` for every rejection reason. Edit a task `Dockerfile` and
run `omp-coding-mint gate TASK_DIR` to rebuild and regate it. Never publish
minted tasks: they are `sensitive_data = "private"`.

## Evaluate and benchmark minted tasks

Repository tasks run OMP with its native tools. Pass the task directory, a
large total token budget, and the policy's real context window:

```sh
eval omp-coding \
  --model MODEL_ID \
  --no-push \
  --no-rich \
  --env.taskset.tasks-dir tasks/minted \
  --env.taskset.split holdout \
  --env.agent.max-total-tokens 400000 \
  --env.agent.harness.context-window 200000 \
  --client.base-url OPENAI_BASE_URL \
  --client.api-key-var API_KEY_VARIABLE
```

Rank models with `omp-coding-bench run --models a,b --tasks-dir tasks/minted
--output bench/NAME` or aggregate existing result directories with
`omp-coding-bench aggregate RUN_DIR... --tasks-dir tasks/minted --output
bench/NAME`. Compare runs only when `bench.json` reports one
`benchmark_digest`. The reference rows come from the sessions' own models and
are not sealed rewards.

## Prime login and hosted evaluation

Check credentials before any hosted work: `prime whoami` (or
`PRIME_API_KEY` in the ignored `.env`). If it fails, run `uv tool install
prime` and `prime login`; never paste a key into a command line. Hosted
sandboxes and Prime Inference bill the account, so a `Payment required`
error means the wallet at https://app.primeintellect.ai/dashboard/billing
needs credits, not that the run is broken.

`omp-coding-eval --hosted` replaces `eval` when Docker is unavailable
(Windows included): rollouts and grading run in Prime VM sandboxes, `.env`
is loaded, and the model defaults to Prime Inference with the same key.
Packaged tasks run directly; minted tasks need `omp-coding-mint publish
TASK_DIR...` first so Prime holds their images. Keep the same flags,
task split, rollout count, and model across compared runs. On Windows,
Defender may quarantine the tunnel client under `~/.prime/bin`; report that
instead of disabling protection, and offer WSL 2 or a custom tunnel as the
alternatives.

## Install

Install the environment in the Prime tool environment. Do not let an active
workspace environment change the install target:

```sh
env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT \
  prime --plain env install omp-coding --path environments
```

Install the local Metal option when training is necessary:

```sh
uv sync --package omp-coding --extra metal
uv run --package omp-coding --extra metal omp-coding-train preflight
```

The preflight must report `metal`, `gpu:0`, the device name, and a checked
matrix result. Never use CPU as a replacement.

## Evaluate

Use `eval`, which is the Prime Verifiers v1 evaluation command in the installed
Prime tool environment:

```sh
eval omp-coding \
  --model MODEL_ID \
  --no-push \
  --no-rich \
  --env.taskset.split train \
  --client.base-url OPENAI_BASE_URL \
  --client.api-key-var API_KEY_VARIABLE \
  --max-concurrent 1
```

Use the 10 fixed train tasks and deterministic generated tasks for trace
collection. Use 4 validation tasks for model selection. Use 4 holdout tasks
only for the final result. Keep task count, rollout count, sampling values,
token limits, and model endpoint fixed for each comparison.

Each rollout must run the real OMP 17.2.15 process. OMP must call the five host
tools through the Verifiers route. A fresh container must grade the output.
Import success and task setup are not evaluation proof.

## Export training data

Use successful train and validation result directories:

```sh
uv run --package omp-coding --extra metal omp-coding-train export \
  outputs/TRAIN_RUN outputs/VALIDATION_RUN \
  --output dataset/VERSION \
  --minimum-train-trajectories 200 \
  --minimum-validation-trajectories 4
```

The exporter accepts only completed traces with a `tests` reward of `1.0`. It
writes one sample for each successful trajectory. The sample contains all
assistant action turns and the final assistant turn. It keeps the exact branch,
message order, tool schemas, tool call identifiers, tool result identifiers,
and tool arguments.
It requires at least 200 train trajectories and 4 validation trajectories.

Do not train on holdout traces. Do not train on failed or incomplete traces.

## Train

Use a new adapter path for each run:

```sh
uv run --package omp-coding --extra metal omp-coding-train run \
  --data dataset/VERSION \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --adapter adapters/VERSION \
  --iters 300 \
  --checkpoint-interval 50 \
  --max-seq-length 8192 \
  --num-layers 8
```

The command runs the Metal preflight again. It uses the selected model tokenizer
and keeps only complete samples that fit the sequence limit. It reports the kept
and removed sample counts. It calculates loss on all assistant turns and masks
the system prompt, user turns, and tool results. Each reported loss
must be finite. The command installs `adapters.safetensors` and
`adapter_config.json` only after all checks pass.

## Compare

Use the built-in evaluator. It fuses one checkpoint and controls both local
Metal servers:

```sh
uv run --package omp-coding --extra metal omp-coding-evaluate \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --data dataset/VERSION \
  --adapter adapters/VERSION \
  --weights adapters/VERSION/0000300_adapters.safetensors \
  --workspace . \
  --split validation \
  --max-tokens 1024 \
  --num-rollouts 1 \
  --output adapters/VERSION/comparison.json
```

The command checks one native OMP tool response before each evaluation. It
uses the same task split, prompt, parser, temperature, token limit, rollout
count, and model endpoint for the base model and fused adapter. A saved
adapter and a lower training loss do not prove an improvement.

Report these values:

- Exact evaluation commands.
- OMP version and arm64 image digest.
- Metal backend, `gpu:0`, device name, MLX version, and dtype.
- Train, validation, and holdout rewards.
- Passed and total sealed case counts.
- Token use and elapsed time from Prime traces.
- Adapter path and byte count.

Claim an improvement only when the adapter has a higher sealed validation or
holdout reward than the fixed base model.

## Remote Metal jobs

Read the `tailscale-compute-fleet` skill before remote work. Select an Apple
Metal node. Run the same preflight and training command there. Do not copy
`.env`. Fetch the adapter only after the job has a successful terminal state.
Then run the fixed comparison through Prime Verifiers v1.

## Rules

- Keep keys in the ignored `.env` file. Never print or commit them.
- Keep `outputs/`, `dataset/`, and `adapters/` out of Git.
- Do not expose sealed task cases to the model.
- Do not change the device, model, precision, task split, or workload to make a
  failed run pass.
- Use the reward from the fresh verifier container. Do not trust model claims.
