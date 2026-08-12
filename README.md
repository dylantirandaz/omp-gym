# omp-gym

A training environment for coding models, built on the
[omp](https://github.com/can1357/oh-my-pi) tool surface.

The idea: omp is the environment, real agent sessions are the
dataset, and a local trainer closes the loop. The same pattern that
Goodfire Silico applies to model internals, applied to agent
behavior.

## The loop

```
tasks/           one task = prompt + workspace + test command
   |
   v
omp-gym run      runs a real omp session on the task workspace,
   |             then runs the test; the test result is the reward
   v
runs/            episode = session.jsonl + episode.json + logs
   |
   v
omp-gym export   scored trajectories -> chat-format train/valid JSONL
   |
   v
omp-gym train    LoRA on the Apple silicon GPU through mlx-lm,
   |             gated by a fail-fast Metal preflight
   v
adapters/        adapter weights + train_report.json
```

## Quickstart

Requirements: `omp` on the path with a configured model, `uv`,
Apple silicon.

```sh
uv sync
uv run omp-gym preflight                      # verify the Metal GPU
uv run omp-gym run --task tasks/fizzbuzz-fix  # one real episode
uv run omp-gym run --task tasks/csv-total
uv run omp-gym export --runs runs --out dataset
uv run omp-gym train --data dataset --iters 60 --adapter adapters/v1
```

Compare the base model with the tuned model:

```sh
uv run python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path adapters/v1 \
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

The runner copies the workspace, so tasks stay reproducible. Add
tasks to grow the dataset. The test command is the reward function:
exit 0 gives reward 1.0, everything else gives 0.0.

## Verified on this machine

- Device: Apple M3, Metal backend through MLX, 12124 MiB.
- Two episodes with the default omp model, both reward 1.0.
- Export: 2 episodes -> 1 train + 1 valid document.
- LoRA on Qwen2.5-0.5B-Instruct-4bit, 60 iterations:
  train loss 1.967 -> 0.000, val loss 1.646 -> 1.435 on the
  held-out episode, about 850 tokens/s, peak memory 4.0 GiB.
- Behavior check: the base model refuses to act; the tuned model
  answers a new task with a well-formed `<tool_call>` block.

## Limits of this version

- SFT only. Episodes with reward below the threshold are dropped,
  not used as negatives.
- Thinking blocks are not exported.
- Tool results are cut at 4000 characters in the export.
- The included run is pipeline-scale proof, not a full training run.
  Two episodes cannot teach a general policy.

## Next steps

- Task library: import tasks from real repositories and issues.
- Rejection sampling: many episodes per task, keep the winners.
- Preference pairs: reward 1.0 versus reward 0.0 episodes -> DPO.
- Close the loop: serve the tuned model with `mlx_lm server`,
  point omp at it as a custom provider, and measure the reward of
  the tuned policy inside the same environment.
