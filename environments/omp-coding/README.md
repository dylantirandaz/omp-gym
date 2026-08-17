# omp-coding

`omp-coding` is a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers)
evaluation environment for coding agents. Each rollout runs one real `omp`
episode in a new copy of a task workspace. The task test command gives the
reward.

The environment uses the Verifiers 0.3.0 v0 API. It returns a `vf.State` from
`rollout()`. A registered rubric scores the state with the binary
`EpisodeRecord.reward` value. A passing test command with valid test evidence
gives `1.0`. All other completed episodes give `0.0`.

This package is not a policy-gradient training backend. `omp` runs as a
separate process. Verifiers sampling arguments do not control the requests
from that process. Use `omp-gym export` to create training data from scored
episodes.

## Runtime requirements

- Python 3.11, 3.12, or 3.13.
- `omp` on `PATH`.
- An OpenAI-compatible policy endpoint.
- The API key variable selected by the Verifiers provider.

The Python wheel does not install the external `omp` program. Install it on
each evaluation host before you load the environment:

```sh
curl -fsSL https://omp.sh/install | sh
omp --version
```

For each rollout, the environment creates a temporary `models.yml`. This file
points `omp` at `client.config.api_base_url`. It uses
`client.config.api_key_var` and the configured static headers. The runner
copies this file into the private episode home before it starts `omp`. A
loopback endpoint gets loopback access only. A remote endpoint gets outbound
HTTPS.

The API key value stays in the child process environment and out of the model
file and rollout state. The source file has mode `0600`. The runner removes
the private copy when `omp` exits. The temporary source file is then removed.
The environment does not change `~/.omp/agent/models.yml`.

## Tasks

The wheel includes these 18 public tasks:

- `config-overlay`
- `csv-join`
- `csv-total`
- `event-sourcing`
- `fizzbuzz-fix`
- `graph-topo`
- `js-csv-parse`
- `js-deep-get`
- `js-event-emitter`
- `js-query-string`
- `js-router-tree`
- `py-rle`
- `rate-limiter`
- `retry-policy`
- `rpn-calc`
- `slugify`
- `temp-convert`
- `word-freq`

Each task has a `task.toml` file and a `workspace/` directory. The workspace
starts with failing tests.

## Rollout state

Verifiers provides the standard state fields. This environment also sets:

- `episode_reward`: the exact binary `EpisodeRecord.reward` value.
- `episode_result`: a tagged result object.
  - Success has `status = "success"` and the full `EpisodeRecord` under
    `record`.
  - Failure has `status = "failure"`, `task`, `failure_class`, and `reason`.

`completion` contains the recorded `omp` messages without the first user
prompt. A session parse failure gives a short assistant completion and adds
`completion_error = "session_parse"` to the success result.

## Local evaluation

Use a model ID that the policy endpoint returns from `/v1/models`. This
example uses the Verifiers `local` provider and one rollout:

```sh
cd environments/omp-coding
export MODEL_ID='<exact-model-id>'
export VLLM_API_KEY='local-placeholder'

uv run vf-eval omp-coding \
  -m "$MODEL_ID" -p local \
  -n 1 -r 1 -c 1 --max-retries 0 \
  -d -s -C 'episode_reward,episode_result'
```

The endpoint must accept the value in `VLLM_API_KEY`. A keyless local server
can use a placeholder value.

## Python use

```python
import verifiers as vf

environment = vf.load_environment("omp-coding")
```

## Public release

The package pins Verifiers to its reviewed 0.3.0 commit. It also pins
`omp-gym` to a reviewed repository commit. Build and inspect the wheel before
publication. Then publish with Prime CLI 0.6.23:

```sh
uv tool install 'prime==0.6.23'
prime login
prime env push --path environments/omp-coding --visibility PUBLIC
```

A team release can add `--team <team-name>`. The account must have a public
user name. Do not publish until a real one-rollout evaluation passes on a host
that has `omp` installed.
