"""Experiment ledger: one JSON line per platform action.

Every verb (run, bench, train, serve, export, improve) appends one
entry with its config, its metrics, and its artifact paths. The
ledger is the platform's memory: the report command diffs versions
from it, and the operator reads it to decide the next experiment.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LEDGER = Path("experiments/ledger.jsonl")


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded action with everything needed to compare it."""

    kind: str
    timestamp: str
    config: dict[str, object]
    metrics: dict[str, object]
    artifacts: dict[str, str]
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def append_entry(
    ledger_path: Path,
    kind: str,
    config: dict[str, object],
    metrics: dict[str, object],
    artifacts: dict[str, str],
) -> LedgerEntry:
    """Append one entry and return it."""
    entry = LedgerEntry(
        kind=kind,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config=config,
        metrics=metrics,
        artifacts=artifacts,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as handle:
        handle.write(json.dumps(asdict(entry)) + "\n")
    return entry


def read_ledger(ledger_path: Path) -> tuple[list[LedgerEntry], int]:
    """Read all entries. A torn last line is counted, not fatal."""
    if not ledger_path.is_file():
        return [], 0
    entries: list[LedgerEntry] = []
    torn = 0
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        entries.append(
            LedgerEntry(
                kind=str(raw["kind"]),
                timestamp=str(raw["timestamp"]),
                config=dict(raw.get("config", {})),
                metrics=dict(raw.get("metrics", {})),
                artifacts=dict(raw.get("artifacts", {})),
                entry_id=str(raw.get("entry_id", "")),
            )
        )
    return entries, torn
