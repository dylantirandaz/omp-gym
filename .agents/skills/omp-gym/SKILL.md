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
9. `inspect --prompt "..." --adapter adapters/<name>` — logit lens
   over every layer. Artifact under `experiments/`.
10. `sae --adapter adapters/<name>` — tiny SAE on residual
    activations. Research preview.
11. `rl --task tasks/<name> --adapter adapters/<name> --group K
    --iters N` — group-relative policy gradient on live episodes.
12. `mint` — write runnable tasks from failed sessions into
    `tasks/minted/`.
13. `import --from claude|codex` — import other agents' sessions.
14. `clusters` — failure-mode counts with example artifacts.
15. `fix --task tasks/<name> --anchor <model>` — measured
    before/after retrain on one task.
16. `doctor` / `init` — environment checks; first scored episode.
17. `publish --push` — render the ledger to GitHub Pages.

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

## Remote Metal jobs

Use the registered `tailscale-compute` MCP server for long Metal
jobs when a remote Apple GPU is available.

1. Call `compute_status`. Check the target, platform, architecture,
   memory, storage, and active jobs.
2. Use `.tailscale-compute-ignore` to include only the required
   dataset and source adapter. Never include `.env`.
3. Call `compute_run` with `uv sync && uv run omp-gym preflight`.
   The remote workload must report the selected Metal device.
4. Call `compute_job_start` for training. Set
   `PYTHONUNBUFFERED=1` so that job logs show live progress. Split
   work that can exceed the 12-hour job limit into complete runs.
5. Read `compute_job_status` and `compute_job_logs`. Keep the byte
   offset from each log result. Do not stop another trainer until
   the remote log shows the real model, source adapter, and training
   loop.
6. Call `compute_fetch` only after a successful terminal state.
   Fetch the adapter and `train_report.json`.
7. Serve and benchmark the fetched adapter against the prior
   adapter. A completed training job is not proof of improvement.

## Rules

- Keys live in the gitignored `.env`. Never print or commit them.
- Training must pass the GPU preflight. Do not fall back to CPU.
- Judge episodes only by test reward.
- `runs/`, `dataset/`, `adapters/`, `experiments/` stay out of git.
- Write the session summary early and update it as you work. The
  clock can cut the session at any time.
