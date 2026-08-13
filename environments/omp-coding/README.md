# omp-coding

Agentic coding tasks driven by the omp harness.

Each rollout is a real omp episode: the policy model calls omp's
tools (read, bash, edit, write, grep, glob) inside a fresh copy of
the task workspace. The task's test command produces the reward:
exit 0 gives reward 1.0, any other exit code gives 0.0.

The environment receives the policy endpoint from the verifiers
client and hands it to omp through the keyless LM Studio discovery
path, so no omp configuration is required.

## Tasks

Three tasks ship in the package:

- `fizzbuzz-fix` — find and fix a condition-order bug.
- `csv-total` — implement a CSV column sum.
- `slugify` — implement a URL slug function.

Each task is a `task.toml` prompt plus a `workspace/` directory
where the tests fail. Add tasks by copying the layout.

## Use

```python
import verifiers as vf

env = vf.load_environment("omp-coding")
```

Evaluate with `vf-eval omp-coding -m <model> -p <provider>`.

Requires `omp` installed and on the PATH of the machine that runs
the rollouts.
