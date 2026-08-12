# omp-gym

A training environment for coding models, built on the
[omp](https://github.com/can1357/oh-my-pi) tool surface.

The idea: omp is the environment, real agent sessions are the
dataset, and a local trainer closes the loop. The same pattern that
Goodfire Silico applies to model internals, applied to agent
behavior.

## The loop

```
tasks/            one task = prompt + workspace + test command
   |
   v
omp-gym run       runs a real omp session on the task workspace,
   |              then runs the test; the test result is the reward
   v
runs/             episode = session.jsonl + episode.json + logs

omp-gym export    two sources -> chat-format train/valid JSONL:
   |              1. scored episodes under runs/, filtered by reward
   |              2. every omp session under the sessions root,
   |                 past and current, with no filter
   v
omp-gym train     LoRA on the Apple silicon GPU through mlx-lm,
   |              gated by a fail-fast Metal preflight
   v
adapters/         adapter weights + train_report.json
```

## Quickstart

Requirements: `omp` on the path with a configured model, `uv`,
Apple silicon.

```sh
uv sync
uv run omp-gym preflight                      # verify the Metal GPU
uv run omp-gym run --task tasks/fizzbuzz-fix  # one scored episode
uv run omp-gym export                         # episodes + all sessions
uv run omp-gym train --data dataset --iters 100 \
  --adapter adapters/v2 --max-seq-length 2048
```

`export` harvests every session below `~/.omp/agent/sessions` by
default. Point `--sessions` at a different root when needed. Run
`export` again at any time; it sweeps everything on disk, so new
sessions enter the next dataset.

Compare the base model with the tuned model:

```sh
uv run python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path adapters/v2 \
  --system-prompt "$(uv run python -c 'from omp_gym.export import SYSTEM_PROMPT; print(SYSTEM_PROMPT)')" \
  --prompt "The test file test_parser.py fails. Fix parser.py." \
  --max-tokens 80
```

## Tasks

A task is one directory:

```
tasks/<name>/
  task.toml        prompt, test_command, optional tools and max_time
  workspace/       initial repository state; tests must fail here
```

The runner copies the workspace, so tasks stay reproducible. The
test command is the reward function: exit 0 gives reward 1.0,
everything else gives 0.0.

## Verified on this machine

- Device: Apple M3, Metal backend through MLX, 12124 MiB.
- Two scored episodes with the default omp model, both reward 1.0.
- Harvest: 109 sessions seen, 107 trainable, 0 torn lines;
  together with the episodes: 99 train + 10 valid documents, 38 MB.
- LoRA on Qwen2.5-0.5B-Instruct-4bit, 100 iterations, sequence cap
  2048: train loss 2.997 -> 0.1, peak memory 6.6 GB, 6.4 minutes.
- Behavior check: the base model refuses to act; the tuned model
  answers a new task with one well-formed `<tool_call>` block and
  stops at the turn boundary.

## What the harvest means

- Every session below the sessions root becomes training data:
  past, current, and future ones on the next export. Sessions with
  no assistant turn are skipped because they cannot train anything.
- Harvested sessions have no tests, so no reward exists and no
  quality filter applies. Failed work trains the model too.
- Everything stays on this machine. The exporter and trainer make
  no network calls; only `omp-gym run` talks to your model provider,
  the same as any omp use.

## Limits of this version

- SFT only. Low-reward episodes are dropped, not used as negatives.
- Thinking blocks are not exported.
- Tool results are cut at 4000 characters in the export.
- Documents longer than the sequence cap train only on their head.

## Next steps

- Task library: import tasks from real repositories and issues.
- Rejection sampling: many episodes per task, keep the winners.
- Preference pairs: reward 1.0 versus reward 0.0 episodes -> DPO.
- Close the loop: serve the tuned model with `mlx_lm server`,
  point omp at it as a custom provider, and measure the reward of
  the tuned policy inside the same environment.
