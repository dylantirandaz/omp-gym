# Silico for coding agents

`omp-gym` is Silico for coding agents. It is a training environment for
coding models. It uses Prime Verifiers v1 and the real
[Oh My Pi](https://github.com/can1357/oh-my-pi) agent. It turns your own OMP
sessions into sealed repository tasks, runs those tasks as evaluations and
benchmarks, records complete tool-use trajectories, checks the changed files
in sealed containers, exports successful trajectories, and trains MLX LoRA
adapters on Apple silicon.

The active environment is `omp-coding` version `1.2.0`.

## Data flow

```text
~/.omp/agent/sessions/*.jsonl
  -> omp-coding-mint (episode, git-anchored start state, fail -> pass test run)
  -> task directory with a Docker image, sealed tests, and a reference patch
  -> Prime eval with --env.taskset.tasks-dir
  -> OmpHarness
  -> pinned OMP 17.2.15 in the task image with its native tools
  -> workspace patch, graded by the sealed test command in a fresh container
  -> sealed reward and Prime trace
  -> omp-coding-bench leaderboard, or successful full-trajectory export
  -> assistant-masked MLX LoRA training
  -> fixed base and adapter comparison
```

Two task kinds share the environment:

- Repository tasks (schema 3, `test-command-v1`) are minted from sessions.
  OMP runs with its native `read`, `write`, `edit`, `bash`, `grep`, and
  `glob` tools inside the task image. The sealed test command grades the
  final workspace after the trusted test files are restored.
- Packaged tasks (schema 2, `call-cases-v1`) are the 18 fixed public tasks.
  OMP gets five host tools (`sandbox_read`, `sandbox_write`, `sandbox_edit`,
  `sandbox_exec`, `run_tests`) and can change only the declared files. Each
  `run_tests` call checks a new file snapshot in a different container.

## Requirements

- Minting: any host that holds the sessions, with `git` and Docker. The
  minter is standard-library Python and runs on Windows, macOS, and Linux.
- Evaluation on Prime's hosted sandboxes: any host after `prime login`,
  including Windows. No Docker needed.
- Evaluation with local Docker rollouts and training: macOS or Linux with
  Docker (on Windows use WSL 2 with Docker Desktop's WSL integration).
- Python 3.11, 3.12, or 3.13.
- Prime CLI 0.6.23 and a funded Prime Intellect account for hosted sandboxes
  and Prime Inference.
- One OpenAI-compatible model endpoint, or Prime Inference through the same
  Prime login.
- Apple silicon and MLX for local adapter training and comparison.

## Log in to Prime

One login covers sandboxes, tunnels, image builds, and Prime Inference:

```sh
uv tool install prime
prime login          # browser challenge; or: prime config set-api-key
prime whoami         # shows the account and the key's permissions
```

`prime login` writes `~/.prime/config.json`. Everything here reads that file,
and the ignored `.env` at the repository root overrides it:

```sh
PRIME_API_KEY=        # optional; overrides ~/.prime/config.json
PRIME_TEAM_ID=        # optional; run under a team
OPENROUTER_API_KEY=   # only for --client.base-url https://openrouter.ai/api/v1
```

`omp-coding-eval` and `omp-coding-mint publish` load `.env` automatically;
the `eval` command needs `set -a; . .env; set +a` first. Sandboxes and
inference are billed to the account; fund it at
https://app.primeintellect.ai/dashboard/billing before the first hosted run.

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

## Mint tasks from your sessions

OMP keeps every session under `~/.omp/agent/sessions`. The minter reads them,
splits each session into episodes (one per user request), and keeps an
episode only when it can prove it: the repository must still exist on this
host, a commit must match the files as they were before the agent's first
edit, and the agent must have run a recognized test command that failed and
then passed after its edits. There are no synthetic tests.

See what would be minted, and why episodes are rejected:

```sh
uv run --package omp-coding omp-coding-mint scan
```

Mint the tasks:

```sh
uv run --package omp-coding omp-coding-mint mint --output tasks/minted
```

For each accepted episode the minter writes `tasks/minted/<repo>-<session>-e<n>/`
with `task.toml` (schema 3), a `Dockerfile`, the repository start state under
`workspace/`, the trusted end-state test files under `verifier/files/`, the
reference patch, and `provenance.json` (session id, models, tokens, cost, base
commit, and both gate runs). It then builds the image, runs the sealed test
command on the start state (must fail) and on the reference patch (must pass),
and records the passed case count as `expected_cases`. Tasks that fail the
gate are deleted unless you pass `--keep-failed`; edit the `Dockerfile` and
run `omp-coding-mint gate TASK_DIR` to rebuild and regate.

Tasks from the same repository share a split (`--split auto`, the default),
or force one with `--split holdout`. Minted tasks are `sensitive_data =
"private"`: `tasks/` is ignored by Git, the prompt is scrubbed of the home
directory, and episodes whose prompt, patch, or tests look like they contain a
secret are rejected. Prior episodes of the same session are overlaid on the
base commit; edits that OMP persisted without full file contents make an
episode unmintable, and the scan output says so.

## Evaluate minted tasks

Point the taskset at the minted directory. Repository tasks need a larger
token budget and context window than the packaged tasks:

```sh
env -u VIRTUAL_ENV uv run eval omp-coding \
  --model openai/gpt-4.1-mini \
  --no-push \
  --no-rich \
  --env.taskset.tasks-dir tasks/minted \
  --env.taskset.split holdout \
  --env.agent.max-total-tokens 400000 \
  --env.agent.harness.context-window 200000 \
  --client.base-url https://openrouter.ai/api/v1 \
  --client.api-key-var OPENROUTER_API_KEY \
  --max-concurrent 2
```

Each rollout starts a container from the task image, hands the repository to
OMP with its native tools, and records the workspace patch. A fresh container
applies the patch, restores the trusted test files, and runs the sealed
command. The `tests` reward is the fraction of passed cases; a run that
removes tests, times out, or produces no test summary scores zero.

## Evaluate on Prime without Docker

`omp-coding-eval` wraps the same `eval` command. With `--hosted` every
rollout and every grading container is a Prime VM sandbox, so no Docker is
needed on the host, and Windows works: the launcher loads `.env`, exports the
`prime login` key, defaults the model endpoint to Prime Inference, and adds
the shims the Verifiers process needs on Windows.

Packaged tasks use a public image and run as they are:

```sh
uv run --package omp-coding omp-coding-eval --hosted omp-coding \
  --model openai/gpt-4.1-mini \
  --no-push \
  --no-rich \
  --env.taskset.split validation \
  --max-concurrent 4
```

Minted tasks use private images. Publish them once; Prime builds each task
image from the task directory and the task is rewritten to reference it:

```sh
uv run --package omp-coding omp-coding-mint publish tasks/minted/*
uv run --package omp-coding omp-coding-eval --hosted omp-coding \
  --model openai/gpt-4.1-mini \
  --no-push \
  --no-rich \
  --env.taskset.tasks-dir tasks/minted \
  --env.taskset.split holdout \
  --env.agent.max-total-tokens 400000 \
  --env.agent.harness.context-window 200000
```

The first hosted run of an image waits for Prime to build its VM image
(about ten minutes); later runs start in seconds. Pass `--client.base-url`
and `--client.api-key-var` to use another model endpoint instead of Prime
Inference. Without `--hosted` the launcher is a plain `eval` with `.env`
loaded.

On Windows the Verifiers tunnel client is the official `frpc` release, placed
under `~/.prime/bin` after a checksum check. Windows Defender may quarantine
it (`Trojan:Win32/Kepavll!rfn` is its usual verdict for frp); either add an
exclusion for that directory or run your own tunnel and pass
`--env.interception.type server --env.interception.tunnel.type custom
--env.interception.tunnel.url URL --env.interception.tunnel.port PORT`.

## Benchmark models

Run several models on the same task set and build a leaderboard:

```sh
uv run --package omp-coding omp-coding-bench run \
  --models openai/gpt-4.1-mini,anthropic/claude-sonnet-4 \
  --tasks-dir tasks/minted \
  --split holdout \
  --num-rollouts 3 \
  --client-base-url https://openrouter.ai/api/v1 \
  --client-api-key-var OPENROUTER_API_KEY \
  --output bench/v1
```

Or aggregate result directories you already have:

```sh
uv run --package omp-coding omp-coding-bench aggregate \
  outputs/RUN_A outputs/RUN_B --tasks-dir tasks/minted --output bench/v1
```

`bench/v1/bench.md` and `bench.json` hold the leaderboard (mean reward, pass
rate, pass@k, error rate, tokens, tool calls, seconds), the per-task matrix,
the runs, and a reference row per model that produced the original sessions.
Runs are comparable only when their task-set digest matches; the report warns
otherwise.

## Train on minted tasks

The minted directory is the RL environment: `prime-rl` and `eval` load the
same `omp-coding` package with `--env.taskset.tasks-dir`, so the benchmark
reward and the training reward come from the same sealed grader. The trace
exporter below also accepts native-tool traces; a dataset cannot mix the two
tool contracts.

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

Packaged tasks live in `environments/omp_coding/omp_coding/tasks`; minted
tasks live wherever `--env.taskset.tasks-dir` points. Every task declares:

- A fixed task and revision identifier.
- A train, validation, or holdout split.
- A pinned Linux image (the shared arm64 image for schema 2; a per-task
  `omp-gym/<name>:<revision>` image with its digest and architecture for
  schema 3).
- CPU, memory, process, disk, home, and time limits.
- Source, source revision, license, and data class (`public` or `private`).
- Schema 2: public context files, editable files, and one sealed structured
  case file.
- Schema 3: one sealed test command, the sealed test files, and the reference
  patch, plus the `Dockerfile` that bakes the repository start state into the
  image and tags it `omp-gym-start`.

The task loader rejects unknown fields, links, special files, wrong image data,
wrong runtime data, duplicate identifiers, invalid limits, and sealed files in
the public workspace.

## Verification

Run the local contracts:

```sh
uv run ruff check environments/omp_coding/omp_coding \
  environments/omp_coding/tests
uv run --package omp-coding python -m unittest discover \
  -s environments/omp_coding/tests -p 'test_*.py'
```

The session, minting, test-command, and benchmark tests run on any host. Set
`OMP_GYM_DOCKER_TESTS=1` to also build a real image and run the gate.

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
