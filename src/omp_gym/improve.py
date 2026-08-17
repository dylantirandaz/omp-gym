"""The operator loop: hand a research goal to an omp agent.

`improve` spawns omp in this repository with the omp-gym skill,
the ledger, and a hard wall-clock budget. The agent proposes and
runs experiments through the normal verbs; every verb records into
the ledger, so the loop leaves an auditable trail. The harness
enforces both budgets: it polls the ledger while the operator
runs and kills the process group when the verb count or the wall
clock exceeds its limit.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .ledger import append_entry, read_ledger
from .runner import _episode_environment, _kill_process_group

PROMPT_TEMPLATE = """You are the omp-gym operator. Read the skill at
.agents/skills/omp-gym/SKILL.md first and follow it.

Research goal: {goal}

State:
- The experiment ledger is {ledger}. Read it before proposing
  anything; do not repeat experiments it already records.
- Provider keys are in the gitignored .env file.

Budget: you may run at most {budget} omp-gym verb commands
(run, bench, export, train, serve). Count them against the ledger
as you go. The wall clock is limited; prefer cheap experiments
first.

Rules:
- Within your first three actions, write {work_dir}/summary.md
  with your plan, and update it after every verb with what you
  ran, exact numbers (rewards, losses, costs), and what should
  happen next. The clock may cut the session at any moment, so
  the summary must always reflect the current state.
- After each verb, read its output and check the ledger entry.
- Keep working notes in {work_dir}/notes.md as you go.
- Judge only by test rewards and measured metrics.
"""

# How often the harness recounts the ledger while the operator
# runs. Tests shrink this to keep the kill fast.
_POLL_SECONDS = 15.0

# Grace on top of max_time before the harness kills a session
# whose own --max-time did not stop it.
_TIMEOUT_GRACE_SECONDS = 120.0


def _operator_command(prompt: str, max_time: int) -> list[str]:
    """Build the operator omp invocation."""
    return [
        "omp",
        "-p",
        prompt,
        "--auto-approve",
        "--mode",
        "json",
        "--max-time",
        str(max_time),
    ]


def _entries_since(ledger_path: Path, start_count: int) -> int:
    """Count ledger entries appended after the recorded start."""
    entries, _ = read_ledger(ledger_path)
    return max(len(entries) - start_count, 0)


@dataclass(frozen=True)
class ImproveResult:
    """What one operator session produced."""

    work_dir: str
    exit_code: int
    duration_seconds: float
    ledger_entries_before: int
    ledger_entries_after: int
    summary_written: bool


def run_improve(
    goal: str,
    budget: int,
    max_time: int,
    ledger_path: Path,
) -> ImproveResult:
    """Run one bounded operator session and record it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work_dir = Path("experiments") / f"improve-{stamp}"
    work_dir.mkdir(parents=True, exist_ok=False)

    entries_before, _ = read_ledger(ledger_path)
    prompt = PROMPT_TEMPLATE.format(
        goal=goal,
        budget=budget,
        ledger=ledger_path,
        work_dir=work_dir,
    )
    (work_dir / "prompt.md").write_text(prompt)

    command = _operator_command(prompt, max_time)
    started = time.monotonic()
    deadline = started + max_time + _TIMEOUT_GRACE_SECONDS
    timed_out = False
    budget_exceeded = False
    # The child gets the same whitelisted environment as an
    # episode: basic host variables plus the .env provider
    # keys, not the full parent environment. The remaining
    # trust is the verb itself: the operator can edit this
    # repository, so run it only when you accept that.
    #
    # A new session gives the operator its own process group, so
    # one killpg reaches omp and everything omp spawned.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_episode_environment(os.environ, None),
        start_new_session=True,
    )
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            # start_new_session makes the child its group leader,
            # so its pid names the group.
            _kill_process_group(process.pid)
            stdout, stderr = process.communicate()
            exit_code = -1
            break
        try:
            stdout, stderr = process.communicate(
                timeout=min(_POLL_SECONDS, remaining)
            )
            exit_code = process.returncode
            break
        except subprocess.TimeoutExpired:
            pass
        if _entries_since(ledger_path, len(entries_before)) > budget:
            budget_exceeded = True
            _kill_process_group(process.pid)
            stdout, stderr = process.communicate()
            exit_code = process.returncode
            break
    duration = time.monotonic() - started
    (work_dir / "events.jsonl").write_text(stdout)
    if stderr:
        (work_dir / "stderr.log").write_text(stderr)

    entries_after, _ = read_ledger(ledger_path)
    summary = (work_dir / "summary.md").is_file()
    result = ImproveResult(
        work_dir=str(work_dir),
        exit_code=exit_code,
        duration_seconds=round(duration, 1),
        ledger_entries_before=len(entries_before),
        ledger_entries_after=len(entries_after),
        summary_written=summary,
    )
    append_entry(
        ledger_path,
        kind="improve",
        config={"goal": goal, "budget": budget, "max_time": max_time},
        metrics={
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "verbs_recorded": len(entries_after) - len(entries_before),
            "summary_written": summary,
            "timed_out": timed_out,
            "budget_exceeded": budget_exceeded,
        },
        artifacts={"work_dir": str(work_dir)},
    )
    return result
