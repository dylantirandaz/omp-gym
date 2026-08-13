# omp-gym

omp-gym is a training and evaluation toolkit for coding agents. It
runs the [omp](https://github.com/can1357/oh-my-pi) agent on
test-scored tasks, records the sessions, converts them to training
data, and fine-tunes local models on Apple silicon. Trained
adapters are served back as omp providers, so trained models run
and score in the same environment as API models.

## Requirements

- macOS on Apple silicon (Metal GPU)
- omp installed, with at least one configured model provider
- uv

## Install

```sh
uv sync
```

## Quickstart

```sh
uv run omp-gym preflight                      # verify the Metal GPU
uv run omp-gym run --task tasks/fizzbuzz-fix  # one scored episode
uv run omp-gym bench --models "claude-haiku-4-5,openrouter/moonshotai/kimi-k2" --tasks tasks
uv run omp-gym export                         # build the dataset
uv run omp-gym train --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --iters 200 --adapter adapters/v3 --max-seq-length 2048
uv run omp-gym serve --adapter adapters/v3 --port 8800
uv run omp-gym report                         # compare adapters and models
uv run omp-gym ui                             # dashboard on :8900
```

## Concepts

**Task.** A directory with `task.toml` and `workspace/`. The
workspace is a repository state where the test command fails. The
prompt asks the agent to make the tests pass.

**Episode.** One omp run on a task in a copied workspace. The test
command scores the episode: exit 0 gives reward 1.0, any other
exit code gives 0.0.

**Ledger.** `experiments/ledger.jsonl`. Every command appends one
JSON line with its config, metrics, and artifact paths. The report
command, the dashboard, and the operator all read this file.

**Sample.** One assistant turn with its context window, exported
for SFT. The exporter measures tokens with the trainee tokenizer
and never truncates the completion.

**Pair.** A chosen and a rejected assistant turn for the same task
prompt, exported for DPO. Chosen turns come from episodes with
reward 1.0. Rejected turns come from episodes that made real
attempts and failed. Provider errors are excluded.

**Adapter.** LoRA weights trained on the Metal GPU. `serve`
publishes an adapter as an omp provider, so `bench` scores it like
an API model.

## Commands

`preflight` — verify the Metal GPU. Run before training. Exits
nonzero when the GPU cannot run a checked operation.

`run --task DIR [--model M]` — run one episode. Artifacts go to
`runs/`.

`bench --models a,b,c [--trials N] [--tasks DIR]` — run the model
x task grid. Writes a markdown report and a JSONL row set.
Provider errors are reported separately from task failures.

`export [--pairs] [--out DIR]` — write the dataset. Default:
per-turn SFT samples from scored episodes and from all omp
sessions under `~/.omp/agent/sessions`. `--pairs` writes DPO
preference pairs.

`train --model M --iters N --adapter DIR [--method sft|dpo]
[--resume-adapter FILE]` — train LoRA weights on the Metal GPU.
`sft` uses mlx-lm. `dpo` uses a native MLX sigmoid-DPO loop and
requires `--resume-adapter`. Training fails when the loss does not
decrease, when a loss is NaN, or when the adapter file is not
written.

`serve --adapter DIR [--port N]` — serve an adapter behind an
OpenAI-compatible endpoint and register it as an omp provider.
Blocks until interrupted.

`improve --goal "..." [--budget N] [--max-time S]` — run an
operator agent that plans and executes experiments through the
commands above. The operator writes a summary to its work
directory.

`report` — render adapter and model comparisons from the ledger.

`ui [--port N]` — serve the dashboard. Read-only.

## Serve, in the omp UI

A served adapter appears in the omp model picker like any other
provider:

![omp model picker showing the omp-gym provider with the local
adapter, marked free](assets/models-picker.png)

The shim in front of the model server parses the model's text into
OpenAI tool calls, because the mlx-lm server drops tool calls when
a request carries a `tools` parameter. Three output envelopes are
accepted: `<tool_call>` blocks, fenced ```json blocks, and bare
JSON objects.

## Provider keys

Put provider keys in `.env` at the project root (`KEY=VALUE`
lines). The file is gitignored. Every episode loads it; its values
override the shell environment. A malformed line stops the run
with the file and line number.

## Data locations

`runs/`, `dataset/`, `dataset-dpo/`, `adapters/`, `experiments/`
and `.env` are gitignored. Session data does not leave the
machine. The exporter and the trainer make no network calls.
Episodes contact the configured model provider, as any omp run
does.

## Measured results

Hardware: Apple M3, Metal through MLX, 12124 MiB.

- Bench: 3 tasks x 7 models. Every API model passed every task.
  kimi-k2 used the fewest tokens per solve (36,378).
  cerebras/gpt-oss-120b was fastest (7.0 s mean) and cheapest
  ($0.0189 for 3 tasks). The suite is too easy to rank API models
  on capability.
- SFT: Qwen2.5-3B-Instruct-4bit, 200 iterations, 21,952 samples.
  Train loss 2.058 -> 1.404. Validation loss 2.151 -> 2.001.
  42 minutes.
- DPO: 26 pairs, 36 iterations. Loss 0.695 -> 0.000 in 176 s.
  Bench after DPO matched bench before DPO: 26 pairs did not
  change episode behavior.
- Operator: one session refreshed the dataset, identified the
  cause of the local models' 0% pass rate (the v3 run covered
  about 1% of the dataset), and proposed the next experiment with
  decision metrics.

## Limits

- Harvested sessions have no quality filter. Failed work trains
  the model too.
- Thinking blocks are not exported.
- Tool results are cut at 4000 characters in the export.
- Turns whose bare sample exceeds the token budget are skipped.

## Prime Intellect Environments Hub

`environments/omp-coding/` wraps the episode runner as a
verifiers-compatible environment. Rollouts run real omp episodes
against the policy endpoint that the trainer provides. To publish,
install the Prime CLI, log in, and run `prime env push` from that
directory.
