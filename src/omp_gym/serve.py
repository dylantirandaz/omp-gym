"""Serve a trained adapter as an omp model provider.

`serve` starts an OpenAI-compatible mlx-lm server on the Metal GPU
with the adapter applied, and registers an omp provider entry so
that the tuned model is addressable as `omp-gym/<model-id>` in
`omp-gym run`, `omp-gym bench`, and plain omp sessions.

The provider file is only written when omp-gym owns it: a missing
file is created with a marker comment, and a file with the marker
is rewritten. A models.yml written by hand is never touched; the
snippet is printed instead.
"""

import subprocess
import sys
from pathlib import Path

from .preflight import require_metal_gpu

PROVIDER_MARKER = "# managed by omp-gym"


def provider_yaml(port: int, model_id: str, base_model: str) -> str:
    """Render the omp provider entry for the local server.

    The wire id must be the base model path: the mlx server
    resolves the request's model field as a model path, and the
    adapter loaded at startup applies to that model. The model_id
    only names the entry; it is not sent on the wire.
    """
    return (
        f"{PROVIDER_MARKER}\n"
        "providers:\n"
        "  omp-gym:\n"
        f"    baseUrl: http://127.0.0.1:{port}/v1\n"
        "    auth: none\n"
        "    api: openai-completions\n"
        "    models:\n"
        f"      - id: {base_model}\n"
        f"        name: omp-gym {model_id}\n"
        "        contextWindow: 32768\n"
        "        maxTokens: 4096\n"
        "        cost:\n"
        "          input: 0\n"
        "          output: 0\n"
        "          cacheRead: 0\n"
        "          cacheWrite: 0\n"
    )


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
    models_yml.parent.mkdir(parents=True, exist_ok=True)
    models_yml.write_text(rendered)
    print(
        f"provider registered: --model 'omp-gym/{base_model}' "
        f"-> 127.0.0.1:{port}"
    )
    return True


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
    if not (adapter_dir / "adapters.safetensors").is_file():
        print(
            f"no adapter found at {adapter_dir}; train one first",
            file=sys.stderr,
        )
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
    backend = subprocess.Popen(command)
    print(
        f"serving omp-gym {model_id} as --model 'omp-gym/{base_model}'; "
        "stop with Ctrl+C"
    )
    try:
        serve_shim(port, backend_port)
    finally:
        backend.terminate()
        backend.wait(timeout=10)
    return 0
