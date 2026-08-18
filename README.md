# Silico for coding agents

`omp-gym` is Silico for coding agents. It is a training environment for
coding models. It uses Prime Verifiers v1 and the real
[Oh My Pi](https://github.com/can1357/oh-my-pi) agent. It records complete
tool-use trajectories, checks the changed files in sealed containers, exports
successful trajectories, and trains MLX LoRA adapters on Apple silicon.

The active environment is `omp-coding` version `1.1.0`.

## Data flow

```text
Prime eval
  -> OmpHarness
  -> pinned OMP 17.2.15 in an arm64 Linux container
  -> five OMP tools through an authenticated host route
  -> bounded task workspace
  -> fresh verifier container
  -> sealed reward and Prime trace
  -> successful full-trajectory export
  -> assistant-masked MLX LoRA training
  -> fixed base and adapter comparison
```

The five tools are `sandbox_read`, `sandbox_write`, `sandbox_edit`,
`sandbox_exec`, and `run_tests`. The model cannot read the sealed cases. The
model can change only the declared task files. Each `run_tests` call checks a
new file snapshot in a different container.

## Requirements

- macOS or Linux with Docker.
- An arm64 host or an arm64 Docker service.
- Python 3.11, 3.12, or 3.13.
- Prime CLI 0.6.23.
- One OpenAI-compatible model endpoint.
- Apple silicon and MLX for local adapter training and comparison.

## Install

Install the environment in the Prime tool environment:

```sh
env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT \
  prime --plain env install omp-coding --path environments
```

Install the Metal training option in this workspace:

```sh
uv sync --package omp-coding --extra metal
uv run --package omp-coding --extra metal omp-coding-train preflight
```

The preflight reports the Metal backend, `gpu:0`, the Apple device name, the
architecture, memory, MLX version, data type, and checked result. It runs one
real matrix operation and checks its output. It does not replace a training
run.

## Collect trajectories

The package has 18 fixed public tasks. It can also make deterministic generated
tasks from safe templates. Run GPT-4.1 mini on the train split:

```sh
set -a
. .env
set +a

env -u VIRTUAL_ENV uv run eval omp-coding \
  --model openai/gpt-4.1-mini \
  --num-rollouts 5 \
  --max-concurrent 4 \
  --no-push \
  --no-rich \
  --env.taskset.split train \
  --env.taskset.generated-tasks 43 \
  --env.taskset.generation-seed 1701 \
  --env.retries.max-retries 1 \
  --env.retries.include ProviderError \
  --sampling.max-tokens 32768 \
  --client.base-url https://openrouter.ai/api/v1 \
  --client.api-key-var OPENROUTER_API_KEY
```

Each result directory has a `traces.jsonl` file. Each trace has the exact model
requests, model responses, tool calls, tool results, token use, reward, and
task data.

## Export training data

Export successful Prime traces:

```sh
uv run --package omp-coding --extra metal omp-coding-train export \
  outputs/TRAIN_RUN outputs/VALIDATION_RUN \
  --output dataset/v1 \
  --minimum-train-trajectories 200 \
  --minimum-validation-trajectories 4
```

The exporter accepts only completed traces with a `tests` reward of `1.0`. It
writes one sample for each successful trajectory. Each sample has all assistant
action turns and the final assistant turn. It keeps the exact branch, message
order, tool schemas, tool call identifiers, tool result identifiers, and tool
arguments. It removes variable OMP model and date lines from the system prompt.

## Train on Apple Metal

Train one Qwen3 4B LoRA adapter:

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

The trainer uses the selected model tokenizer. It keeps only complete samples
that fit the sequence limit. It reports the kept and removed sample counts. It
calculates loss on all assistant turns. It masks the system prompt, user turns,
and tool results. Each reported loss must be finite. The command installs the
adapter files only after all checks pass.

## Measure the adapter

Compare the fixed base model with the fused adapter:

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

The command starts one local Metal server for the base model. It checks one
native OMP tool response. It then fuses the adapter, starts a second server,
and uses the same task split, task count, prompt, parser, temperature, token
limit, and rollout count. It reports the base reward, adapter reward, valid
tool-call rate, invalid tool-call rate, end-token rate, repeated tool-call
rate, and comparison data hash.

Use `validation` while you select an adapter. Use `holdout` one time for the
final result.

## Task contract

Each task is in `environments/omp_coding/omp_coding/tasks`. Each task declares:

- A fixed task and revision identifier.
- A train, validation, or holdout split.
- A pinned arm64 Linux image.
- CPU, memory, process, disk, home, and time limits.
- Public context files and editable files.
- One sealed structured case file.
- Source, source revision, license, and data class.

The task loader rejects unknown fields, links, special files, wrong image data,
wrong runtime data, duplicate identifiers, invalid limits, and case files in
the public workspace.

## Verification

Run the local contracts:

```sh
uv run ruff check environments/omp_coding/omp_coding \
  environments/omp_coding/tests
uv run --package omp-coding python -m unittest discover \
  -s environments/omp_coding/tests -p 'test_v1.py'
```

Run the real Docker path:

```sh
uv run --package omp-coding python \
  environments/omp_coding/tests/integration_runtime.py
```

This command checks the pinned OMP file, arm64 image, task copy, candidate
worker, sealed verifier, and expected reward.

Build the release wheel:

```sh
uv build --wheel environments/omp_coding --out-dir dist
```
