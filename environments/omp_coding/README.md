# omp-coding

`omp-coding` version `1.0.0` is a Prime Verifiers v1 environment. Each rollout
runs one real OMP episode in a new task container. Prime records the policy
turns. A second container grades the declared output files with sealed cases.

## Public classes

The package exports three Verifiers v1 classes:

- `OmpTaskset` loads one fixed task split.
- `OmpHarness` runs pinned OMP 17.2.15 through the Prime policy endpoint.
- `OmpEnv` grades the collected files and records the reward.

The wheel contains all task data, OMP RPC support, candidate workers, and the
training data exporter. It does not use a Verifiers v0 adapter.

## Runtime

Each solver uses this image:

```text
node@sha256:934240a162082fd8b8a2f90cd5114446443f1eba1c5378f6687167ca405e6584
```

The task runtime has one CPU, 768 MiB of memory, 64 processes, and no network.
The harness downloads the pinned arm64 OMP file during setup. It checks the
SHA-256 value and version before use.

OMP has five host tools:

- `sandbox_read`
- `sandbox_write`
- `sandbox_edit`
- `sandbox_exec`
- `run_tests`

The file tools reject undeclared paths. `sandbox_exec` runs as the unprivileged
user and has a 30 second command limit. `run_tests` sends only aggregate case
counts to the model. The sealed case file stays outside the solver container.

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

Run MLX LoRA training:

```sh
uv run --package omp-coding --extra metal omp-coding-train run \
  --data dataset/v1 \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter adapters/omp-coding-v1 \
  --iters 20
```

The command prints the Metal backend, `gpu:0`, device name, architecture,
memory, MLX version, and checked result before training. It uses the selected
model tokenizer. It keeps only complete samples that fit the sequence limit,
and it reports the kept and removed sample counts. Each reported loss must be
finite. The command installs the adapter files only after all checks pass.

## Release

Build the wheel:

```sh
uv build --wheel environments/omp_coding --out-dir dist
```

Inspect the wheel before publication. It must contain `omp_coding/tasks`, the
five OMP RPC files, the three runtime workers, and `training.py`.
