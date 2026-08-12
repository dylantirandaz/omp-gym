---
name: omp-gym
description: Operate the omp-gym training loop - run scored episodes, harvest sessions, train LoRA adapters on the Metal GPU, benchmark models, and serve tuned adapters as omp providers. Use when asked to improve, evaluate, or train coding models with this repository.
---

# omp-gym operator

You operate a training platform for coding models. The loop has
five verbs. All of them run through `uv run omp-gym <verb>`.

## The loop

1. `preflight` — verify the Metal GPU. Run before training.
2. `run --task tasks/<name> [--model <m>]` — one scored episode:
   fresh workspace, real omp session, tests decide reward (1.0/0.0).
   Episodes land in `runs/<task>-<stamp>/` with `episode.json`.
3. `export` — build `dataset/train.jsonl` + `dataset/valid.jsonl`
   from scored episodes and every omp session on this machine.
   Per-turn samples, exact token budgets, trajectory-level split.
4. `train --data dataset --model <mlx-model> --iters N
   --adapter adapters/<name> --max-seq-length 2048` — LoRA on the
   Metal GPU. Fails on NaN loss or a flat loss curve. Writes
   `adapters/<name>/train_report.json`.
5. `bench --models "a,b,c" --tasks tasks --trials N` — model x task
   grid. Writes `bench-report.md` and `bench-report.jsonl` with
   pass rate, latency, tokens, cost. Provider errors are separated
   from task failures.

`serve --adapter adapters/<name> --port 8800 --model-id <id>`
publishes a tuned adapter as omp model `omp-gym/<id>`, so the tuned
model can run episodes and appear in bench like any other model.
Serve blocks; run it as a background process.

## How to improve the model

1. Add tasks under `tasks/<name>/` (task.toml + workspace where the
   test fails). Harder tasks separate models better.
2. Collect episodes across models with `bench --trials 3`; winners
   feed the dataset automatically.
3. `export`, then `train` a new adapter version (v4, v5, ...).
4. `serve` the new adapter and `bench` it against the previous
   version and an API anchor model.
5. Compare `bench-report.md` and `train_report.json` between
   versions. Report reward, loss, cost, and latency changes with
   exact numbers. Never claim improvement without a bench delta.

## Rules

- Keys live in the gitignored `.env`; never print or commit them.
- Training must pass GPU preflight; never fall back to CPU.
- Judge episodes only by test reward, never by transcript vibes.
- `runs/`, `dataset/`, `adapters/` stay out of git.
