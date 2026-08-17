"""Serve a trained adapter as an omp model provider.

`serve` starts an OpenAI-compatible mlx-lm server on the Metal GPU
with the adapter applied, and registers an omp provider entry so
that the tuned model is addressable as `omp-gym/<model-id>` in
`omp-gym run`, `omp-gym bench`, and plain omp sessions.

The provider file is only written when omp-gym owns it: a missing
file is created with a marker comment, and a file with the marker
is rewritten. A models.yml written by hand is never touched; the
snippet is printed instead. Writes are atomic: the rendered YAML is
written to a temp file in the same directory under a sibling .lock
fcntl lock and installed with os.replace, and an unchanged file is
left alone.
"""

import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from .preflight import require_metal_gpu

PROVIDER_MARKER = "# managed by omp-gym"


def provider_yaml(port: int, model_id: str, base_model: str) -> str:
    """Render the omp provider entry for the local server.

    The wire id must be the base model path: the mlx server
    resolves the request's model field as a model path, and the
    adapter loaded at startup applies to that model. The model_id
    only names the entry; it is not sent on the wire. safe_dump
    quoting keeps model ids and paths with spaces, colons, or
    other YAML-special characters intact.
    """
    document = {
        "providers": {
            "omp-gym": {
                "baseUrl": f"http://127.0.0.1:{port}/v1",
                "auth": "none",
                "api": "openai-completions",
                "models": [
                    {
                        "id": base_model,
                        "name": f"omp-gym {model_id}",
                        "contextWindow": 32768,
                        "maxTokens": 4096,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }
    return PROVIDER_MARKER + "\n" + yaml.safe_dump(document, sort_keys=False)


def _write_owned(models_yml: Path, rendered: str) -> None:
    """Install the provider file atomically, skipping unchanged content."""
    models_yml.parent.mkdir(parents=True, exist_ok=True)
    lock_path = models_yml.with_name(models_yml.name + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if models_yml.is_file() and models_yml.read_text() == rendered:
            return
        tmp_path = models_yml.with_name(models_yml.name + ".tmp")
        tmp_path.write_text(rendered)
        os.replace(tmp_path, models_yml)


def ensure_provider(
    models_yml: Path,
    port: int,
    model_id: str,
    base_model: str,
) -> bool:
    """Write the provider entry when omp-gym owns the file.

    Return True when the file is in place, False when a foreign
    models.yml exists and the user must merge the entry by hand.
    """
    rendered = provider_yaml(port, model_id, base_model)
    if models_yml.is_file():
        first_line = models_yml.read_text().splitlines()[:1]
        if first_line != [PROVIDER_MARKER]:
            print(
                f"{models_yml} exists and omp-gym does not own it.\n"
                "Merge this provider entry by hand:\n\n"
                f"{rendered}"
            )
            return False
    _write_owned(models_yml, rendered)
    print(f"provider registered: --model 'omp-gym/{base_model}' -> 127.0.0.1:{port}")
    return True


def verify_adapter(adapter_dir: Path) -> list[str]:
    """Check an adapter directory before serving; list every failure.

    adapters.safetensors must exist, be nonempty, and parse as a
    safetensors file (8-byte little-endian header length followed by
    a JSON header object); adapter_config.json must parse as JSON.
    An empty list means the adapter is servable.
    """
    problems: list[str] = []
    weights = adapter_dir / "adapters.safetensors"
    if not weights.is_file():
        problems.append(f"missing {weights}")
    elif weights.stat().st_size == 0:
        problems.append(f"{weights} is empty")
    else:
        try:
            with open(weights, "rb") as stream:
                header_length = int.from_bytes(stream.read(8), "little")
                if header_length > weights.stat().st_size - 8:
                    raise ValueError("header length exceeds file size")
                header = json.loads(stream.read(header_length))
            if not isinstance(header, dict):
                problems.append(f"{weights} header is not a JSON object")
        except (
            OSError,
            ValueError,
            UnicodeDecodeError,
        ) as error:
            problems.append(f"{weights} does not parse as safetensors: {error}")
    config = adapter_dir / "adapter_config.json"
    try:
        json.loads(config.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        problems.append(f"{config} does not parse: {error}")
    return problems


def wait_ready(port: int, timeout_seconds: float = 30.0) -> bool:
    """Poll /v1/models until the backend answers 200 or time runs out."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def run_server(
    base_model: str,
    adapter_dir: Path,
    port: int,
    model_id: str,
    models_yml: Path,
) -> int:
    """Run the adapter server and the tool-call shim until interrupted.

    The mlx server listens on port+1; the shim on `port` is what omp
    talks to. See shim.py for why the shim exists.
    """
    from .shim import serve_shim

    require_metal_gpu()
    problems = verify_adapter(adapter_dir)
    if problems:
        print(
            f"adapter at {adapter_dir} is not servable:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    if not ensure_provider(models_yml, port, model_id, base_model):
        return 1
    backend_port = port + 1
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        base_model,
        "--adapter-path",
        str(adapter_dir),
        "--port",
        str(backend_port),
    ]
    print("+", " ".join(command))
    backend = subprocess.Popen(command)  # noqa: S603 - argv list, no shell
    try:
        if not wait_ready(backend_port):
            print(
                f"backend on 127.0.0.1:{backend_port} did not become ready within 30s",
                file=sys.stderr,
            )
            return 1
        print(
            f"serving omp-gym {model_id} as --model 'omp-gym/{base_model}'; "
            "stop with Ctrl+C"
        )
        serve_shim(port, backend_port, str(adapter_dir))
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait()
    return 0
