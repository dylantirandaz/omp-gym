# omp-gym

`omp-gym` is a coding environment for Prime Verifiers v1. It runs the real
[Oh My Pi](https://github.com/can1357/oh-my-pi) agent. It records each model
turn through the Verifiers interception service. It grades changed files in a
new sealed container.

The active environment is `omp-coding` version `1.0.0`. The old local runner,
model shim, ledger, dashboard, and custom RL code are not part of this version.

## Data flow

```text
Prime eval or train
  -> OmpHarness
  -> pinned OMP 17.2.15 in an arm64 Linux container
  -> five OMP tools through an authenticated host route
  -> bounded task workspace
  -> fresh verifier container
  -> structured reward and Prime trace
```

The five tools are `sandbox_read`, `sandbox_write`, `sandbox_edit`,
`sandbox_exec`, and `run_tests`. The model cannot read the sealed cases. The
model can change only the declared task files. Each `run_tests` call grades a
new snapshot in a different container.

## Requirements

- macOS or Linux with Docker.
- An arm64 host or an arm64 Docker service.
- Python 3.11, 3.12, or 3.13.
- Prime CLI 0.6.23.
- One OpenAI compatible model endpoint.
- Apple silicon and MLX for local adapter training.

## Install

Install the environment in the Prime tool environment. Remove active virtual
environment variables so that Prime does not install it only in this workspace:

```sh
env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT \
  prime --plain env install omp-coding --path environments
```

Install the Metal training option in this workspace:

```sh
uv sync --package omp-coding --extra metal
uv run --package omp-coding --extra metal omp-coding-train preflight
```

The preflight must report `metal`, `gpu:0`, and the Apple device name. It runs
one checked matrix operation. It does not replace a training run.

## Evaluate

The package has 18 public tasks. The task splits are fixed: 10 train tasks, 4
validation tasks, and 4 holdout tasks.

Run one validation task through OpenRouter:

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

Run the full train split by removing `--num-tasks 1`. Use `validation` for
model selection. Use `holdout` only for the final comparison.

Each result directory contains `traces.jsonl`. Each trace includes the exact
model requests, model responses, tool calls, tool results, token use, reward,
and task data.

## Train on Apple Metal

First, collect successful Prime v1 traces from train and validation runs. Then
export one sample for each sampled assistant turn:

```sh
uv run --package omp-coding --extra metal omp-coding-train export \
  outputs/TRAIN_RUN outputs/VALIDATION_RUN \
  --output dataset/v1
```

The exporter keeps tool schemas, tool call identifiers, tool result
identifiers, branch order, and message order. It accepts only traces with a
`tests` reward of `1.0`. It requires train and validation samples.

Train one LoRA adapter on the Metal GPU:

```sh
uv run --package omp-coding --extra metal omp-coding-train run \
  --data dataset/v1 \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter adapters/omp-coding-v1 \
  --iters 20 \
  --max-seq-length 4096 \
  --num-layers 8
```

The command calls the shared Metal preflight first. It uses the selected model
tokenizer. It keeps only complete samples that fit the sequence limit, and it
reports the kept and removed sample counts. It uses MLX LoRA with prompt
masking. It fails if a reported loss is not finite or if MLX does not write a
complete adapter. It installs the adapter only after all checks pass.

Fuse the adapter before the comparison. MLX-LM 0.30 does not apply its
command-line adapter when the request names a model:

```sh
uv run --package omp-coding --extra metal python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-4bit')"
uv run --package omp-coding --extra metal python -m mlx_lm fuse \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path adapters/omp-coding-v1 \
  --save-path adapters/omp-coding-v1-fused
uv run --package omp-coding --extra metal python -m mlx_lm server \
  --model adapters/omp-coding-v1-fused \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 4096
```

Run `eval omp-coding --model default_model` against
`http://127.0.0.1:8080/v1`. Serve the unfused base model for the baseline.
Use `default_model` for both runs. Use the same task split, task count,
sampling values, and limits. Do not claim an improvement unless the adapter
has a higher sealed reward on the fixed validation or holdout set.

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
uv run ruff check environments/omp_coding/omp_coding environments/omp_coding/tests
uv run --package omp-coding python -m unittest discover \
  -s environments/omp_coding/tests -p 'test_v1.py'
```

Run the real Docker path:

```sh
uv run --package omp-coding python environments/omp_coding/tests/integration_runtime.py
```

This command checks the pinned OMP binary, arm64 image, task copy, candidate
worker, sealed verifier, and expected baseline score.
