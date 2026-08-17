import unittest

from graph import CycleError, topo_sort


class OrderTest(unittest.TestCase):
    def test_empty_graph(self) -> None:
        self.assertEqual(topo_sort([], []), [])

    def test_single_node(self) -> None:
        self.assertEqual(topo_sort(["a"], []), ["a"])

    def test_chain(self) -> None:
        self.assertEqual(
            topo_sort(["c", "a", "b"], [("a", "b"), ("b", "c")]),
            ["a", "b", "c"],
        )

    def test_ties_break_by_smallest_node(self) -> None:
        self.assertEqual(topo_sort(["c", "b", "a"], []), ["a", "b", "c"])

    def test_diamond(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        self.assertEqual(topo_sort(["d", "c", "b", "a"], edges), ["a", "b", "c", "d"])

    def test_ready_set_updates_as_nodes_are_taken(self) -> None:
        edges = [("b", "a"), ("b", "c")]
        self.assertEqual(topo_sort(["a", "b", "c"], edges), ["b", "a", "c"])

    def test_duplicate_edges_count_once(self) -> None:
        edges = [("a", "b"), ("a", "b")]
        self.assertEqual(topo_sort(["b", "a"], edges), ["a", "b"])

    def test_isolated_nodes_mix_into_the_order(self) -> None:
        edges = [("d", "b")]
        self.assertEqual(topo_sort(["d", "c", "b", "a"], edges), ["a", "c", "d", "b"])


class CycleTest(unittest.TestCase):
    def test_self_loop(self) -> None:
        self.assertEqual(
            topo_sort(["x"], [("x", "x")]), CycleError(cycle=("x",))
        )

    def test_two_cycle(self) -> None:
        self.assertEqual(
            topo_sort(["b", "a"], [("a", "b"), ("b", "a")]),
            CycleError(cycle=("a", "b")),
        )

    def test_cycle_starts_at_its_smallest_node(self) -> None:
        edges = [("c", "d"), ("d", "b"), ("b", "c")]
        self.assertEqual(
            topo_sort(["b", "c", "d"], edges),
            CycleError(cycle=("b", "c", "d")),
        )

    def test_nodes_before_the_cycle_still_sort(self) -> None:
        edges = [("a", "b"), ("b", "c"), ("c", "b")]
        self.assertEqual(
            topo_sort(["a", "b", "c"], edges), CycleError(cycle=("b", "c"))
        )

    def test_smallest_cyclic_node_is_the_anchor(self) -> None:
        edges = [("m", "n"), ("n", "m"), ("e", "f"), ("f", "e")]
        self.assertEqual(
            topo_sort(["m", "n", "e", "f"], edges),
            CycleError(cycle=("e", "f")),
        )

    def test_walk_prefers_the_smallest_successor_on_the_cycle(self) -> None:
        edges = [("a", "z"), ("z", "a"), ("a", "b"), ("b", "a")]
        self.assertEqual(
            topo_sort(["a", "b", "z"], edges), CycleError(cycle=("a", "b"))
        )

    def test_dead_end_branch_is_not_part_of_the_cycle(self) -> None:
        edges = [("a", "b"), ("b", "a"), ("b", "q"), ("q", "q")]
        self.assertEqual(
            topo_sort(["a", "b", "q"], edges), CycleError(cycle=("a", "b"))
        )


if __name__ == "__main__":
    unittest.main()
