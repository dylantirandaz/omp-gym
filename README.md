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

## Choose your model

The platform trains and serves any MLX-format open model. Write
one line into a `gym.toml` file at the repository root:

```toml
model = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
```

The value is a Hugging Face repository id or a local directory
with an MLX model you provide (for example a model you converted
with `mlx_lm.convert`). Every verb - `train`, `serve`, `gate`,
`export`, `rl`, `fix`, `inspect`, `sae`, `steer` - uses it as the
default model. Each `--model`, `--base-model`, or `--tokenizer`
flag still overrides it per command. Without `gym.toml` the
default is `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`.

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

`train --model M --iters N --adapter DIR [--num-layers N]
[--learning-rate R] [--method sft|dpo] [--resume-adapter FILE]`
— train LoRA weights
on the Metal GPU. SFT trains the last 16 layers by default. Set
`--num-layers -1` to train all layers. `sft` uses mlx-lm. `dpo`
uses a native MLX sigmoid-DPO loop and requires `--resume-adapter`.
Training fails when the loss does not decrease, when a loss is
NaN, or when the adapter file is not written.

`serve --adapter DIR [--port N]` — serve an adapter behind an
OpenAI-compatible endpoint and register it as an omp provider.
Blocks until interrupted.

`improve --goal "..." [--budget N] [--max-time S]` — run an
operator agent that plans and executes experiments through the
commands above. The operator writes a summary to its work
directory.

`report` — render adapter and model comparisons from the ledger.

`ui [--port N]` — serve the dashboard. Read-only. Shows the
models leaderboard, adapters, episodes, failure clusters, SAE
features, and training curves for every adapter that has a
recorded loss series. Five panels are interactive:

- Run monitor: every `train_report.json` with its loss curve,
  plus a live view of `runs/live-train.log`. Mirror a remote
  trainer log to that file to follow loss, tokens per second,
  and peak memory while the job runs.
- Bench matrix: the run x task grid from the ledger and from
  fetched `*report*.jsonl` row files. Amber columns are the
  sealed holdout tasks. Green cells pass, red cells fail.
- Lens diff: type a prompt and an adapter directory; the panel
  shows the base lens and the adapter lens side by side and
  marks each layer where the top prediction diverges.
- SAE explorer: per-token feature activations as a heat row,
  the strongest features as chips, and a steer slider that
  compares an unsteered and a steered completion live.
- Replay: select an episode, scrub through it step by step,
  watch full-file writes build up, and compare two episodes of
  the same task side by side.

`inspect --prompt "..." [--adapter DIR]` — logit-lens a local
model: the top predicted tokens after every decoder layer. Writes
a JSON artifact under `experiments/`.

`sae [--data DIR] [--adapter DIR]` — train a tiny sparse
autoencoder on residual-stream activations from the dataset.
Research preview. Writes a feature report under `experiments/`.

`rl --task DIR --adapter DIR --group K --iters N` —
group-relative policy gradient over live episodes. Each iteration
serves the current adapter, samples K episodes in parallel, scores
them with the task tests, and updates toward episodes that beat
the group mean. Graded rewards (from tests that print "N of M
cases failed") are used when available.

`mint [--limit N]` — scan sessions for failure signals (user
corrections, late test failures) and write runnable tasks from
them into `tasks/minted/`. Workspaces are reconstructed from
write-tool contents only; path fields that are device URLs are
skipped.

`import --from claude|codex` — convert another agent's session
store to the omp session schema under `imported/`. Export with
`--sessions imported`.

`clusters` — group every session and failed episode by failure
mode (tool errors, edit mismatches, provider errors, user
corrections, gave up). Writes `experiments/clusters.json`; the
dashboard shows it.

`fix --task DIR --anchor MODEL` — measured repair loop: bench the
local policy on the task, collect winning episodes from the anchor
model, retrain the adapter, bench again, record the delta in the
ledger.

`doctor` — check omp, uv, Metal GPU, keys, sessions, disk. Prints
the fix for each failure.

`init` — doctor plus one scored episode with the default model.

`publish [--push]` — render the ledger report to
`docs/index.html`. With `--push`: commit, push, and enable GitHub
Pages for the repo.

## Serve, in the omp UI

A served adapter appears in the omp model picker like any other
provider, marked free.

The shim in front of the model server parses the model's text into
OpenAI tool calls, because the mlx-lm server drops tool calls when
a request carries a `tools` parameter. Three output envelopes are
accepted: `<tool_call>` blocks, fenced ```json blocks, and bare
JSON objects.

## Provider keys

Put provider keys in `.env` at the project root (`KEY=VALUE`
lines). The file is gitignored. Every episode loads it; its values
are part of the episode environment. A malformed line stops the
run with the file and line number.

## Trust boundary

Episodes are not sandboxed. `omp-gym run` starts a real omp
session with auto-approval on this machine, in a copied workspace
that is not a security boundary. The episode can run shell
commands, read the filesystem, and see the provider keys from
`.env`. Task test commands come from `task.toml` and run with your
user account; the loader accepts only `python3`, `python`, `node`,
`pytest`, and `sh` as the first word. Run only tasks and session
imports that you trust, and use a dedicated key with a spend limit.

## Data locations

`runs/`, `dataset*/`, `imported/`, `adapters/`, `experiments/`,
`holdout-results/`, `gym.toml`, and `.env` are gitignored. The
exporter redacts credential-shaped text and keeps datasets on this
machine; the trainer makes no network calls beyond the model
download. Episodes contact the configured model provider, as any
omp run does.

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
- Inspect: logit lens over the tuned 3B adapter shows its 36
  layers forming the prediction "the" from context tokens; the
  artifact is on the dashboard.
- SAE preview: 4,096 features over 68,035 residual tokens of the
  tuned 0.5B; loss 0.307 -> 0.135 in 51 s. Most active features
  fire densely; the preview labels itself as such.
- Retrain: v5 resumes v3 with 2x coverage (200 more iters, clean
  loss 2.340 -> 0.884). The 3B policy still prefers chat over
  tool calls at this coverage. The shim now accepts a fourth
  envelope — a fenced `python3 test_x.py` / `pytest` command
  becomes a bash tool call — because that is the exact
  near-miss format the model produced in a real episode.
- Dataset signal: 21,039 train samples are unique. 21,015 samples
  contain tool calls and only 24 are prose-only. v3 plus v5 saw
  about 400 samples, which is less than 2% of this train set.
- RL: GRPO rounds on fizzbuzz-fix with the served 0.5B policy.
  Graded rewards (0.7 = 7 of 10 cases) arrived correctly. The
  group showed no variance, so no update happened and the ledger
  says so. The real finding: at temperature 1.0 the overfit
  adapter emits empty turns, so the 0.7 partial rewards come from
  the unmodified buggy workspace passing 7 of 10 cases on its
  own. GRPO cannot learn from a policy that says nothing. The
  mechanism and the honest no-signal path are verified; the
  blocker is policy entropy, which the base model has and the
  adapter lost to over-fitting.
- RL on v5: three parallel rollouts overloaded the single-request
  MLX server. Two rollouts were lost, but the old code accepted one
  reward as a full group. Rollouts are now serial and every rollout
  must return a trainable assistant turn. The same Apple M3 run
  completed with three rewards: `[0.7, 0.7, 0.7]`. Direct reward
  comparison found no variance and skipped the update. This also
  prevents float round-off from starting a false update and loading
  a second 3B model into memory.
- Import: 968 Codex sessions converted to the omp schema; export
  with them grew the train set from 19,753 to 47,683 samples.
- Mint: tasks mined from real failed sessions run end-to-end
  (minted-2: reward 1.0, 10/10 tests). Workspaces are rebuilt
  from read and write tool calls, keyed to the session's working
  directory, with the final content winning. Test commands are
  validated against the reconstructed files; a `-k` selector
  whose names do not appear is stripped.
- Clusters: 2,040 tool errors, 497 edit mismatches, 313 provider
  errors, 29 user corrections over the harvest.

## Design decisions and limits

- Minted workspaces are reconstructed from both read and write
  tool calls, keyed to the session's working directory. The
  latest content of each file wins. Test commands are validated
  against the reconstructed files; a `-k` selector whose names
  do not appear is stripped. Tasks whose dependencies extend past
  what the session touched are labeled `partial` and may need
  their original repo to pass.
- Harvested sessions go through a quality filter: sessions that
  ended as failures do not enter the SFT dataset. Pass
  `--no-quality-filter` to include them. DPO pairs still use
  losing episodes by design.
- Thinking blocks are exported as `<think>` blocks when the
  session recorded them.
- Tool results longer than 4000 characters keep the head and the
  tail with an elision marker; the middle is dropped, because the
  error output lives at the end.
- Assistant turns that exceed the token budget are kept with
  their middle elided rather than skipped.
- The dashboard is read-only and local. The training chart draws
  real loss curves only for adapters trained after series
  recording was added (v6 onward).

## Prime Intellect Environments Hub

`environments/omp-coding/` wraps the episode runner as a
verifiers-compatible environment. Rollouts run real omp episodes
against the policy endpoint that the trainer provides. To publish,
install the Prime CLI, log in, and run `prime env push` from that
directory.
