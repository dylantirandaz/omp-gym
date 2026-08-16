"""Repository configuration for the trainee model.

The user selects one open model for the whole platform in an
optional `gym.toml` file at the repository root:

    model = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

The value is an MLX-format Hugging Face repository id or a local
directory that holds an MLX model the user provides. Every verb
uses it as the default; each `--model`, `--base-model`, and
`--tokenizer` flag still overrides it per command.
"""

import tomllib
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
CONFIG_FILE = Path("gym.toml")


class ConfigError(SystemExit):
    """Raised when gym.toml exists but cannot be used."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"bad gym.toml: {reason}")


def default_model(config_file: Path = CONFIG_FILE) -> str:
    """The configured trainee model id or local path.

    A missing file gives the built-in default. A file without a
    usable `model` string is a configuration error, not a fallback.
    """
    if not config_file.is_file():
        return DEFAULT_MODEL
    try:
        raw = tomllib.loads(config_file.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(str(error)) from error
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError('set model = "<hf-repo-id-or-local-path>"')
    return model.strip()
