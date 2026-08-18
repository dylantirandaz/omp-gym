import unittest

from config_overlay import merge_config


class ConfigOverlayTests(unittest.TestCase):
    def test_adds_overlay_key(self):
        self.assertEqual(
            merge_config({"port": 80}, {"host": "localhost"}),
            {"port": 80, "host": "localhost"},
        )

    def test_overlay_replaces_scalar(self):
        self.assertEqual(merge_config({"port": 80}, {"port": 443}), {"port": 443})

    def test_merges_nested_dictionaries(self):
        self.assertEqual(
            merge_config(
                {"server": {"host": "a", "port": 80}},
                {"server": {"port": 443}},
            ),
            {"server": {"host": "a", "port": 443}},
        )

    def test_replaces_lists(self):
        self.assertEqual(
            merge_config({"tags": ["a", "b"]}, {"tags": ["c"]}),
            {"tags": ["c"]},
        )

    def test_none_is_an_explicit_replacement(self):
        self.assertEqual(
            merge_config({"token": "secret"}, {"token": None}),
            {"token": None},
        )

    def test_type_conflict_uses_overlay(self):
        self.assertEqual(
            merge_config({"server": {"port": 80}}, {"server": "disabled"}),
            {"server": "disabled"},
        )

    def test_does_not_mutate_inputs(self):
        base = {"server": {"ports": [80]}}
        overlay = {"server": {"ports": [443]}}
        self.assertEqual(merge_config(base, overlay), {"server": {"ports": [443]}})
        self.assertEqual(base, {"server": {"ports": [80]}})
        self.assertEqual(overlay, {"server": {"ports": [443]}})

    def test_rejects_non_dictionary_input(self):
        with self.assertRaises(TypeError):
            merge_config({}, [])


if __name__ == "__main__":
    unittest.main()
