# omp-coding

`omp-coding` version `1.2.0` is a Prime Verifiers v1 environment. Each rollout
runs one real OMP episode in a new task container. Prime records the policy
turns. A second container grades the result with sealed tests.

## Public classes

The package exports three Verifiers v1 classes:

- `OmpTaskset` loads one fixed task split from the packaged tasks or from an
  external directory (`--env.taskset.tasks-dir`).
- `OmpHarness` runs pinned OMP 17.2.15 through the Prime policy endpoint.
- `OmpEnv` grades the collected files and records the reward.

The wheel contains all packaged task data, OMP RPC support, candidate workers,
the session reader and task minter (`omp-coding-mint`), the benchmark
aggregator (`omp-coding-bench`), and the training data exporter. It does not
use a Verifiers v0 adapter.

## Runtime

Packaged (schema 2) tasks use this image:

```text
node@sha256:934240a162082fd8b8a2f90cd5114446443f1eba1c5378f6687167ca405e6584
```

That runtime has one CPU, 768 MiB of memory, 64 processes, and no network. OMP
gets five host tools:

- `sandbox_read`
- `sandbox_write`
- `sandbox_edit`
- `sandbox_exec`
- `run_tests`

The file tools reject undeclared paths. `sandbox_exec` runs as the unprivileged
user and has a 30 second command limit. `run_tests` sends only aggregate case
counts to the model. The sealed case file stays outside the solver container.

Minted repository (schema 3) tasks use their own image, built by
`omp-coding-mint` from the task `Dockerfile` with the repository start state
committed at `/workspace` and tagged `omp-gym-start`. OMP runs with its native
`read`, `write`, `edit`, `bash`, `grep`, and `glob` tools as the unprivileged
agent user, with the resource limits from `task.toml` and no network. At the
end of the rollout the harness records `git diff` against the start tag as
the workspace patch. Grading applies that patch in a fresh container from the
same image, restores the sealed test files, and runs the sealed command. The
reward is the fraction of passed cases; removed tests, timeouts, patch
failures, and unrecognized test output score zero.

The harness downloads the pinned OMP release for the container architecture
(arm64 or x64) during setup and checks its SHA-256 value and version before
use. `--env.agent.harness.context-window` sets the context size OMP assumes
for the policy; raise it for repository tasks.

## Install

From the repository root, install it in the Prime tool environment:

```sh
env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT \
  prime --plain env install omp-coding --path environments
```

## Evaluate

```sh
set -a
. .env
set +a

eval omp-coding \
  --model openai/gpt-4.1-mini \
  --num-tasks 1 \
  --no-push \
  --no-rich \
  --env.taskset.split validation \
  --client.base-url https://openrouter.ai/api/v1 \
  --client.api-key-var OPENROUTER_API_KEY \
  --max-concurrent 1
```

The result is a Prime v1 `traces.jsonl` file. The `tests` reward is the number
of passed sealed cases divided by the total case count. An infrastructure
failure and a token limit failure get reward `0.0`.

## Mint and benchmark

Turn your OMP sessions into repository tasks, then evaluate and rank models on
them:

```sh
omp-coding-mint scan
omp-coding-mint mint --output tasks/minted
omp-coding-mint gate tasks/minted/TASK

eval omp-coding \
  --model openai/gpt-4.1-mini \
  --no-push \
  --no-rich \
  --env.taskset.tasks-dir tasks/minted \
  --env.taskset.split holdout \
  --env.agent.max-total-tokens 400000 \
  --env.agent.harness.context-window 200000 \
  --client.base-url https://openrouter.ai/api/v1 \
  --client.api-key-var OPENROUTER_API_KEY

omp-coding-bench aggregate outputs/RUN_A outputs/RUN_B \
  --tasks-dir tasks/minted --output bench/v1
```

`omp-coding-mint` keeps an episode only when the repository still exists on
this host, a commit matches the pre-edit files, and the agent ran a test
command that failed and then passed. It builds the task image and proves the
reference patch flips the sealed tests before it keeps the task.
`omp-coding-bench run` drives `eval` for a list of models and aggregates the
traces into `bench.md` and `bench.json`.

## Train

Install the optional Metal packages:

```sh
uv sync --package omp-coding --extra metal
```

Export successful Prime traces:

```sh
uv run --package omp-coding --extra metal omp-coding-train export \
  outputs/TRAIN_RUN outputs/VALIDATION_RUN \
  --output dataset/v1
```

The exporter accepts only completed traces with a `tests` reward of `1.0`. It
writes one sample for each successful trajectory. Each sample has all assistant
action turns and the final assistant turn.

Run MLX LoRA training:

```sh
uv run --package omp-coding --extra metal omp-coding-train run \
  --data dataset/v1 \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --adapter adapters/omp-coding-v1 \
  --iters 300 \
  --checkpoint-interval 50 \
  --max-seq-length 8192 \
  --num-layers 8
```

The command prints the Metal backend, `gpu:0`, device name, architecture,
memory, MLX version, and checked result before training. It uses the selected
model tokenizer. It keeps only complete samples that fit the sequence limit,
and it reports the kept and removed sample counts. It calculates loss on all
assistant turns. It masks the system prompt, user turns, and tool results.
Each reported loss must be
finite. The command installs the adapter files only after all checks pass.

Compare the base model with a saved checkpoint:

```sh
uv run --package omp-coding --extra metal omp-coding-evaluate \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --data dataset/v1 \
  --adapter adapters/omp-coding-v1 \
  --weights adapters/omp-coding-v1/0000300_adapters.safetensors \
  --workspace . \
  --split validation \
  --max-tokens 1024 \
  --num-rollouts 1 \
  --output adapters/omp-coding-v1/comparison.json
```

The comparison uses the same tasks, parser, prompt, sampling values, and token
limits for the base model and the fused adapter. It reports the sealed reward
and the OMP tool-protocol rates.

## Release

Build the wheel:

```sh
uv build --wheel environments/omp_coding --out-dir dist
```

Inspect the wheel before publication. It must contain `omp_coding/tasks`, the
five OMP RPC files, the three runtime workers, and the training and evaluation
modules.
