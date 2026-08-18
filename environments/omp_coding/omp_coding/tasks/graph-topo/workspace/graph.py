"""Topological sort with a named cycle in the error value.

topo_sort(nodes, edges) orders the nodes so that every edge
(a, b) puts a before b:

- nodes is the full node list; a node can have no edges. edges may
  mention only listed nodes (the inputs in the tests are always
  consistent, no validation is needed).
- The order is deterministic: whenever several nodes are ready
  (no unsorted incoming edge), the smallest node string comes next.
- Duplicate edges count once.
- When no full order exists, return CycleError naming one cycle:
  * Take the smallest node that sits on any cycle among the
    unsortable remainder; call it the anchor. A node sits on a cycle
    when it can reach itself through unsortable nodes.
  * Walk from the anchor: at each step move to the smallest
    unsortable successor that can still reach the anchor. Stop when
    the walk returns to the anchor.
  * cycle is that walk as a tuple, anchor first, without repeating
    the anchor at the end. A self-loop on "x" gives ("x",).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CycleError:
    """The graph has no topological order."""

    cycle: tuple[str, ...]


def topo_sort(nodes: list[str], edges: list[tuple[str, str]]) -> list[str] | CycleError:
    """Return the deterministic topological order, or one named cycle.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
