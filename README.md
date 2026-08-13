# omp-gym

A training environment for coding models, built on the
[omp](https://github.com/can1357/oh-my-pi) tool surface.

The idea: omp is the environment, real agent sessions are the
dataset, and a local trainer closes the loop. The same pattern that
Goodfire Silico applies to model internals, applied to agent
behavior.

## The loop

## The platform

Seven verbs plus a dashboard, all feeding one ledger:

- `run` / `bench` — scored episodes and model x task grids.
- `export` — SFT samples, or DPO pairs with `--pairs`.
- `train` — SFT through mlx-lm; DPO through mlx-lm-lora with
  `--method dpo --resume-adapter` (the SFT-then-DPO chain).
- `serve` — adapters become omp providers behind a tool-call shim.
- `report` — adapters and models compared from the ledger: val
  loss, pass rate, cost per pass, tokens per solve.
- `improve --goal "..." --budget N` — hands a research goal to an
  omp operator agent with the ledger as memory and a hard clock.
  Verified live: the operator refreshed the dataset, root-caused
  the local models' 0% pass rate to 1% dataset coverage (with
  transcript evidence), and wrote a costed next-experiment
  proposal to its summary.
- `ui --port 8900` — web dashboard: leaderboard, adapters,
  timeline, episode browser with full transcripts.

Everything writes one JSON line per action to
`experiments/ledger.jsonl` (gitignored). The report, the
dashboard, and the operator all read the same ledger.

## The loop

```
tasks/            one task = prompt + workspace + test command
   |
   v
omp-gym run       runs a real omp session on the task workspace,
   |              then runs the test; the test result is the reward
   v
runs/             episode = session.jsonl + episode.json + logs

omp-gym export    two sources -> per-turn train/valid samples:
   |              1. scored episodes under runs/, filtered by reward
   |              2. every omp session under the sessions root,
   |                 past and current, with no filter
   v
omp-gym train     LoRA on the Apple silicon GPU through mlx-lm,
   |              gated by a fail-fast Metal preflight
   v
adapters/         adapter weights + train_report.json
   |
   v
omp-gym serve     publishes the adapter as omp model
   |              `omp-gym/<base-model>` through mlx-lm + a shim
   v
omp-gym bench     the tuned policy runs episodes and lands on the
                  same leaderboard as the API models
```

## Serve: the tuned model becomes the agent

`serve` closes the loop. It starts an mlx-lm server on the Metal
GPU with the adapter applied, registers an omp provider entry in
`~/.omp/agent/models.yml` (only when omp-gym owns that file), and
fronts the server with a small shim:

```sh
uv run omp-gym serve --adapter adapters/v3 --port 8800
# then, in another terminal:
uv run omp-gym run --task tasks/fizzbuzz-fix \
  --model "omp-gym/mlx-community/Qwen2.5-3B-Instruct-4bit"
```

The shim exists because the mlx-lm server (0.32) swallows tool
calls: with a `tools` parameter it returns neither content nor
`tool_calls`. The shim strips `tools` from the request, describes
the tools in the system message, and parses the model's text into
real OpenAI tool calls. Three envelopes are accepted: the trained
`<tool_call>` block, a fenced ```json block, and a bare JSON
object.

Verified on this machine: the v3 adapter, served locally, drove a
real omp tool call (grep with omp's intent argument) inside a
scored episode, and benched 0% at $0.0000 across three tasks with
3 tool calls. It does not solve tasks yet — a 3B model trained on
104 trajectories starts the loop, it does not win it.

## Per-turn samples

Long agent sessions do not fit a training window, and head
truncation teaches nothing but opening moves. The exporter makes
one sample per assistant turn instead:

- system prompt, the most recent context that fits the token
  budget, then the assistant turn as the final message;
- the budget is measured with the trainee's own tokenizer, message
  costs cached, so no sample can lose its completion to truncation;
- training passes `--mask-prompt`, so loss lands only on the final
  assistant message — never on tool output or user text;
- the train/valid split holds out whole trajectories, so no session
  leaks into both files;
- the trainer hard-fails when any loss report is NaN.

## Quickstart

Requirements: `omp` on the path with a configured model, `uv`,
Apple silicon.

```sh
uv sync
uv run omp-gym preflight                      # verify the Metal GPU
uv run omp-gym run --task tasks/fizzbuzz-fix  # one scored episode
uv run omp-gym export                         # episodes + all sessions
uv run omp-gym train --data dataset \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --iters 200 --adapter adapters/v3 --max-seq-length 2048
```

`export` harvests every session below `~/.omp/agent/sessions` by
default; `--sessions` overrides the root. `--tokenizer` and
`--max-tokens` must match the trainee family and the training
sequence cap. Run `export` again at any time; it sweeps everything
on disk, so new sessions enter the next dataset.

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

## Benchmark

`bench` runs every model on every task under identical conditions:
a fresh workspace copy, one real omp session, then the tests. The
session file supplies tokens and cost, so the leaderboard shows
what a pass rate costs.

```sh
uv run omp-gym bench \
  --models "claude-haiku-4-5,claude-sonnet-4-6,cerebras/gpt-oss-120b" \
  --tasks tasks --trials 1 --report bench-report.md
```

Provider keys live in a gitignored `.env` file at the project
root (`KEY=VALUE` lines). omp-gym loads it for every omp session
it starts, and its values override the shell environment. Put
`OPENROUTER_API_KEY=...` there and OpenRouter models join the
grid: `--models "openrouter/qwen/qwen3-coder,claude-haiku-4-5"`.
A malformed line stops the run with its file and line number.

Provider errors (bad model id, no access, dead key) do not count
as failed tasks. They appear in an errors column, as `E` cells in
the task matrix, and as a list with the recorded error message.

Real result from this machine, 9 episodes in 154 seconds:

| model | pass rate | mean seconds | mean tokens | total cost usd |
| --- | --- | --- | --- | --- |
| cerebras/gpt-oss-120b | 100% (3 runs) | 7.0 | 83653 | 0.0189 |
| claude-haiku-4-5 | 100% (3 runs) | 17.6 | 61523 | 0.0744 |
| claude-sonnet-4-6 | 100% (3 runs) | 26.5 | 62567 | 0.2097 |

Three simple tasks cannot separate these models on capability; they
separate them on speed and price. Add harder tasks to spread the
pass-rate column. Benchmark episodes land in `runs/` like every
other episode, so winners feed the next training export.


## Verified on this machine

- Device: Apple M3, Metal backend through MLX, 12124 MiB.
- Two scored episodes with the default omp model, both reward 1.0.
- Benchmark: 3 models x 3 tasks, 9/9 episodes scored, provider
  errors separated from task failures (verified against a real
  404 model id).
- Harvest: 109 sessions seen, 104 trajectories exported,
  21952 turn samples (19753 train / 2199 valid), 346 oversize
  turns skipped, 0 torn lines. Export takes 26 seconds.
- Independent check: worst sample is 1978 template tokens against
  the 2048 cap.
- LoRA on Qwen2.5-3B-Instruct-4bit, 200 iterations, sequence cap
  2048: train loss 2.058 -> 1.404, val loss 2.151 -> 2.001 on
  held-out sessions, zero NaN reports, peak memory 11.5 GB,
  42 minutes.
- Behavior check on a fresh bug report: the base model invents a
  fake edit API and writes fabricated content without reading
  anything; the tuned model emits one well-formed `<tool_call>`
  that reads a line range around the failing line, then stops.

## What the harvest means

- Every session below the sessions root becomes training data:
  past, current, and future ones on the next export. Sessions with
  no assistant turn are skipped because they cannot train anything.
- Harvested sessions have no tests, so no reward exists and no
  quality filter applies. Failed work trains the model too.
- Everything stays on this machine. The exporter and trainer make
  no network calls; only `omp-gym run` talks to your model provider,
  the same as any omp use.

## DPO, honestly measured

- `export --pairs` builds chosen/rejected pairs from scored
  episodes (wins versus real losses; provider errors excluded;
  pairs over the token cap skipped).
- Training runs a native MLX sigmoid-DPO loop (`--method dpo
  --resume-adapter`), because the community trainer deadlocked on
  this stack in three separate attempts. v4: loss 0.695 -> 0.000
  and 100% preference accuracy in 176 s on the M3.
- The bench verdict is the honest part: v4 is indistinguishable
  from v2 at episode level (0% pass, ~300 tool calls both). The
  mechanism works; 26 first-turn pairs are too thin to move
  behavior. The ledger holds the full evidence chain.

## Limits of this version

- Thinking blocks are not exported.
- Tool results are cut at 4000 characters in the export.
- Turns whose bare sample exceeds the token budget are skipped.

## Next steps

- Task library: import tasks from real repositories and issues.
- Rejection sampling: many episodes per task, keep the winners.
- Preference pairs: reward 1.0 versus reward 0.0 episodes -> DPO.
- Close the loop: serve the tuned model with `mlx_lm server`,
  point omp at it as a custom provider, and measure the reward of
  the tuned policy inside the same environment.
