"""Temperature conversion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvertError:
    """The value is not a valid temperature."""

    reason: str


def convert(value: str) -> str | ConvertError:
    """Convert "<degrees>F" to Celsius or "<degrees>C" to Fahrenheit.

    The result rounds to the nearest integer degree.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
