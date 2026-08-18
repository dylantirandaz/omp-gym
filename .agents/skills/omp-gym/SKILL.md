---
name: omp-gym
description: Operate the Prime Verifiers v1 OMP coding environment. Use it to run sealed coding evaluations, export exact Prime traces, train MLX LoRA adapters on Apple Metal, and compare an adapter with its fixed baseline.
---

# omp-gym operator

Use `omp-coding` version `1.0.0`. Use Prime Verifiers v1 for all rollouts. Do
not use the removed local runner, model shim, ledger, dashboard, or custom RL
commands.

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

Use 10 train tasks for trace collection. Use 4 validation tasks for model
selection. Use 4 holdout tasks only for the final result. Keep task count,
rollout count, sampling values, token limits, and model endpoint fixed for each
comparison.

Each rollout must run the real OMP 17.2.15 process. OMP must call the five host
tools through the Verifiers route. A fresh container must grade the output.
Import success and task setup are not evaluation proof.

## Export training data

Use successful train and validation result directories:

```sh
uv run --package omp-coding --extra metal omp-coding-train export \
  outputs/TRAIN_RUN outputs/VALIDATION_RUN \
  --output dataset/VERSION
```

The exporter accepts only a `tests` reward of `1.0`. It writes one sample for
each sampled assistant turn. It keeps the exact branch, message order, tool
schemas, tool call identifiers, tool result identifiers, and tool arguments.
It requires train and validation samples.

Do not train on holdout traces. Do not train on failed or incomplete traces.

## Train

Use a new adapter path for each run:

```sh
uv run --package omp-coding --extra metal omp-coding-train run \
  --data dataset/VERSION \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter adapters/VERSION \
  --iters 20 \
  --max-seq-length 4096 \
  --num-layers 8
```

The command runs the Metal preflight again. It uses the selected model tokenizer
and keeps only complete samples that fit the sequence limit. It reports the kept
and removed sample counts. It trains with prompt masking. Each reported loss
must be finite. The command installs `adapters.safetensors` and
`adapter_config.json` only after all checks pass.

## Serve and compare

MLX-LM 0.30 does not apply its command-line adapter when a request names a
model. Fuse the adapter before evaluation. Do not put a model shim in front of
the server:

```sh
uv run --package omp-coding --extra metal python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-4bit')"
uv run --package omp-coding --extra metal python -m mlx_lm fuse \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path adapters/VERSION \
  --save-path adapters/VERSION-fused
uv run --package omp-coding --extra metal python -m mlx_lm server \
  --model adapters/VERSION-fused \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 4096
```

Run `eval omp-coding --model default_model` against
`http://127.0.0.1:8080/v1`. Serve the unfused base model for the baseline.
Use `default_model` for both runs. Use the same MLX version, server settings,
task split, and sampling settings. A saved adapter and a lower training loss
do not prove an improvement.

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
