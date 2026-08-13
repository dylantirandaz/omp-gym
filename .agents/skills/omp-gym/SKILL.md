---
name: omp-gym
description: Operate the omp-gym training loop - run scored episodes, harvest sessions, train LoRA adapters on the Metal GPU, benchmark models, and serve tuned adapters as omp providers. Use when asked to improve, evaluate, or train coding models with this repository.
---

# omp-gym operator

You operate a training platform for coding models. Every action
runs through `uv run omp-gym <verb>`.

## Verbs

1. `preflight` — verify the Metal GPU. Run before training.
2. `run --task tasks/<name> [--model <m>]` — one scored episode.
   Fresh workspace copy, real omp session, tests decide the reward
   (1.0 or 0.0). Artifacts land in `runs/<task>-<stamp>/`.
3. `export` — build `dataset/train.jsonl` and `dataset/valid.jsonl`
   from scored episodes and from every omp session on this
   machine. Per-turn samples, exact token budgets, split by
   trajectory. `--pairs` writes DPO pairs instead.
4. `train --data dataset --model <mlx-model> --iters N --adapter
   adapters/<name> [--method sft|dpo] [--resume-adapter FILE]` —
   LoRA on the Metal GPU. Fails on NaN loss or a flat loss curve.
   Writes `adapters/<name>/train_report.json`.
5. `bench --models "a,b,c" --tasks tasks --trials N` — model x
   task grid. Writes `bench-report.md` and `bench-report.jsonl`
   with pass rate, latency, tokens, and cost. Provider errors are
   reported separately from task failures.
6. `serve --adapter adapters/<name> --port 8800` — publish an
   adapter as an omp provider. Blocks; run it in the background.
7. `report` — compare adapters and models from the ledger.
8. `ui --port 8900` — serve the dashboard.

## Procedure for improving the model

1. Read `experiments/ledger.jsonl` before proposing anything. Do
   not repeat recorded experiments.
2. Add tasks under `tasks/<name>/` when the suite cannot separate
   models (task.toml plus a workspace where the test fails).
3. Collect episodes with `bench --trials N`. Winners feed the
   dataset on the next `export`.
4. Train a new adapter version. For DPO, resume from the SFT
   adapter.
5. Serve the new adapter and bench it against the previous version
   and one API model.
6. Report reward, loss, cost, and latency deltas with exact
   numbers from the ledger. Do not claim improvement without a
   bench delta.

## Rules

- Keys live in the gitignored `.env`. Never print or commit them.
- Training must pass the GPU preflight. Do not fall back to CPU.
- Judge episodes only by test reward.
- `runs/`, `dataset/`, `adapters/`, `experiments/` stay out of git.
- Write the session summary early and update it as you work. The
  clock can cut the session at any time.
