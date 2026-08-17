"""Experiment ledger: one JSON line per platform action.

Every verb (run, bench, train, serve, export, improve) appends one
entry with its config, its metrics, and its artifact paths. The
report command and the operator read this file to compare runs
and to decide the next experiment.

Integrity: append_entry serializes writers with an fcntl exclusive
lock on <ledger>.lock and fsyncs before returning, and every line
carries a sha256 over its canonical JSON (sorted keys, the digest
field itself excluded). read_ledger enforces the schema exactly:
schema_version 1, a verifying digest, nonempty kind/timestamp/
entry_id, dict-shaped config/metrics/artifacts with string
artifact keys and values, and no extra top-level fields. Anything
else is malformed — counted, never silently used.
"""

import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LEDGER = Path("experiments/ledger.jsonl")

SCHEMA_VERSION = 1

_ALLOWED_FIELDS = frozenset(
    (
        "kind",
        "timestamp",
        "config",
        "metrics",
        "artifacts",
        "entry_id",
        "schema_version",
        "sha256",
    )
)


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded action with everything needed to compare it."""

    kind: str
    timestamp: str
    config: dict[str, object]
    metrics: dict[str, object]
    artifacts: dict[str, str]
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    schema_version: int = SCHEMA_VERSION


def _canonical_json(payload: dict[str, object]) -> str:
    """The canonical encoding of one record: sorted keys, tight separators."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _record_sha256(payload: dict[str, object]) -> str:
    """Hex digest over the canonical encoding of one record."""
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _lock_path(ledger_path: Path) -> Path:
    """The advisory lock file paired with one ledger."""
    return Path(str(ledger_path) + ".lock")


def append_entry(
    ledger_path: Path,
    kind: str,
    config: dict[str, object],
    metrics: dict[str, object],
    artifacts: dict[str, str],
) -> LedgerEntry:
    """Append one entry under the ledger lock and fsync it, then return it.

    Every writer opens its own lock descriptor, so the exclusive
    flock serializes concurrent writers across threads and
    processes. The appended line carries a sha256 over its
    canonical body; readers reject mismatches as malformed.
    """
    entry = LedgerEntry(
        kind=kind,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config=config,
        metrics=metrics,
        artifacts=artifacts,
    )
    payload: dict[str, object] = asdict(entry)
    payload["sha256"] = _record_sha256(payload)
    line = _canonical_json(payload)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_path(ledger_path).open("a") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            with ledger_path.open("a") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return entry


def _entry_from_raw(raw: dict[str, object]) -> LedgerEntry | None:
    """Validate one decoded record and build its entry, or return None.

    Every rule rejects exactly one failure shape: wrong field set,
    missing or wrong schema_version, missing/unverifying digest,
    empty kind/timestamp/entry_id, non-dict config/metrics/
    artifacts, or non-string artifact keys/values.
    """
    if not _ALLOWED_FIELDS.issuperset(raw.keys()):
        return None
    body = {key: value for key, value in raw.items() if key != "sha256"}
    digest = raw.get("sha256")
    if not isinstance(digest, str) or len(body) != len(_ALLOWED_FIELDS) - 1:
        return None
    if digest != _record_sha256(body):
        return None
    version = raw.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        return None
    kind = raw.get("kind")
    timestamp = raw.get("timestamp")
    entry_id = raw.get("entry_id")
    if not (
        isinstance(kind, str)
        and kind
        and isinstance(timestamp, str)
        and timestamp
        and isinstance(entry_id, str)
        and entry_id
    ):
        return None
    config = raw.get("config")
    metrics = raw.get("metrics")
    artifacts = raw.get("artifacts")
    if not (
        isinstance(config, dict)
        and isinstance(metrics, dict)
        and isinstance(artifacts, dict)
    ):
        return None
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in artifacts.items()
    ):
        return None
    return LedgerEntry(
        kind=kind,
        timestamp=timestamp,
        config=dict(config),
        metrics=dict(metrics),
        artifacts=dict(artifacts),
        entry_id=entry_id,
        schema_version=version,
    )


def read_ledger(ledger_path: Path) -> tuple[list[LedgerEntry], int]:
    """Read all entries; return (entries, malformed_count).

    A line is malformed when it is not UTF-8, not JSON, not a JSON
    object, or fails the strict schema validation in
    _entry_from_raw. Malformed lines are counted and dropped;
    valid lines are never affected by them.
    """
    if not ledger_path.is_file():
        return [], 0
    entries: list[LedgerEntry] = []
    malformed = 0
    for raw_line in ledger_path.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            malformed += 1
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(raw, dict):
            malformed += 1
            continue
        entry = _entry_from_raw(raw)
        if entry is None:
            malformed += 1
            continue
        entries.append(entry)
    return entries, malformed
