"""Load provider keys from a project-local env file.

The file holds KEY=VALUE lines. Blank lines and # comments are
ignored. A leading "export " is tolerated because people paste
shell lines. A malformed line stops the program; a silent skip
would hide a broken key until an episode fails strangely.
"""

from pathlib import Path


class EnvFileError(SystemExit):
    """Raised when the env file has a line that cannot be parsed."""

    def __init__(self, path: Path, line_number: int, line: str) -> None:
        super().__init__(
            f"{path}:{line_number}: expected KEY=VALUE, got: {line!r}"
        )


def load_env_file(path: Path) -> dict[str, str]:
    """Parse the env file. A missing file is an empty result."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or " " in key:
            raise EnvFileError(path, line_number, raw)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            values[key] = value
    return values
