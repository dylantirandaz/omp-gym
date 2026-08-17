"""Explicit LoRA layer selection at the mlx-lm boundary."""

from collections.abc import Mapping
from typing import Literal, TypeAlias

ALL_LAYERS: Literal["all"] = "all"
LayerSelection: TypeAlias = int | Literal["all"]


def parse_layer_selection(value: object) -> LayerSelection | None:
    """Parse one layer value without accepting numeric sentinels.

    The string ``all`` is the only internal all-layer value. A
    positive integer selects the last N layers.
    """
    if value == ALL_LAYERS:
        return ALL_LAYERS
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def mlx_num_layers(selection: LayerSelection) -> int:
    """Translate the explicit selection to mlx-lm's boundary value."""
    if selection == ALL_LAYERS:
        return 0
    return selection


def adapter_layer_selection(
    config: Mapping[str, object],
) -> LayerSelection | None:
    """Read and cross-check layer selection from an adapter config.

    New configs have an explicit layer_selection field. The numeric
    num_layers field stays for mlx-lm compatibility. Old configs with
    num_layers=0 are accepted only at this external boundary.
    """
    numeric = config.get("num_layers")
    explicit = config.get("layer_selection")
    if explicit is None:
        if isinstance(numeric, int) and not isinstance(numeric, bool) and numeric == 0:
            return ALL_LAYERS
        return parse_layer_selection(numeric)
    selection = parse_layer_selection(explicit)
    if selection is None or numeric != mlx_num_layers(selection):
        return None
    return selection


def layer_config_fields(selection: LayerSelection) -> dict[str, object]:
    """Return compatible and explicit adapter config fields."""
    return {
        "num_layers": mlx_num_layers(selection),
        "layer_selection": selection,
    }
