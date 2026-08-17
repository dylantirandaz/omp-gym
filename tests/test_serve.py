import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from omp_gym.serve import (
    PROVIDER_MARKER,
    ensure_provider,
    provider_yaml,
    verify_adapter,
    wait_ready,
)


class ProviderYamlTests(unittest.TestCase):
    def test_safe_dump_round_trips_tricky_characters(self) -> None:
        model_id = 'rl: "quoted" & <tagged> #hash'
        base_model = "org/model with spaces: and # marks"

        rendered = provider_yaml(8123, model_id, base_model)

        self.assertTrue(rendered.startswith(PROVIDER_MARKER))
        parsed = yaml.safe_load(rendered)
        entry = parsed["providers"]["omp-gym"]
        self.assertEqual(entry["baseUrl"], "http://127.0.0.1:8123/v1")
        self.assertEqual(entry["models"][0]["id"], base_model)
        self.assertEqual(entry["models"][0]["name"], f"omp-gym {model_id}")
        self.assertEqual(entry["models"][0]["contextWindow"], 32768)


class EnsureProviderTests(unittest.TestCase):
    def test_writes_then_skips_unchanged_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_yml = Path(tmp) / "agent" / "models.yml"

            self.assertTrue(ensure_provider(models_yml, 8123, "m", "base"))
            first_stat = models_yml.stat()
            first_content = models_yml.read_text()

            self.assertTrue(ensure_provider(models_yml, 8123, "m", "base"))
            second_stat = models_yml.stat()

        self.assertEqual(first_stat.st_mtime_ns, second_stat.st_mtime_ns)
        self.assertEqual(first_stat.st_ino, second_stat.st_ino)
        self.assertEqual(first_content.splitlines()[0], PROVIDER_MARKER)
        self.assertEqual(
            yaml.safe_load(first_content)["providers"]["omp-gym"]["baseUrl"],
            "http://127.0.0.1:8123/v1",
        )

    def test_rewrites_changed_owned_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_yml = Path(tmp) / "models.yml"
            ensure_provider(models_yml, 8123, "m", "base")

            self.assertTrue(ensure_provider(models_yml, 9123, "m", "base"))

            self.assertIn("http://127.0.0.1:9123/v1", models_yml.read_text())

    def test_foreign_file_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_yml = Path(tmp) / "models.yml"
            foreign = "providers:\n  mine:\n    baseUrl: http://x\n"
            models_yml.write_text(foreign)

            self.assertFalse(ensure_provider(models_yml, 8123, "m", "base"))

            self.assertEqual(models_yml.read_text(), foreign)

    def test_uses_a_sibling_lock_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_yml = Path(tmp) / "models.yml"

            ensure_provider(models_yml, 8123, "m", "base")

            self.assertTrue((Path(tmp) / "models.yml.lock").is_file())
            self.assertFalse((Path(tmp) / "models.yml.tmp").exists())


def _write_safetensors(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode()
    path.write_bytes(len(payload).to_bytes(8, "little") + payload)


class VerifyAdapterTests(unittest.TestCase):
    def test_missing_files_list_both_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = verify_adapter(Path(tmp))

        joined = "\n".join(problems)
        self.assertIn("adapters.safetensors", joined)
        self.assertIn("adapter_config.json", joined)

    def test_empty_weights_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapters.safetensors").write_bytes(b"")
            (adapter / "adapter_config.json").write_text("{}")

            problems = verify_adapter(adapter)

        self.assertEqual(len(problems), 1)
        self.assertIn("empty", problems[0])

    def test_unparseable_weights_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapters.safetensors").write_bytes(b"not safetensors")
            (adapter / "adapter_config.json").write_text("{}")

            problems = verify_adapter(adapter)

        self.assertEqual(len(problems), 1)
        self.assertIn("does not parse", problems[0])

    def test_unparseable_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            _write_safetensors(
                adapter / "adapters.safetensors",
                {"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
            )
            (adapter / "adapter_config.json").write_text("{oops")

            problems = verify_adapter(adapter)

        self.assertEqual(len(problems), 1)
        self.assertIn("adapter_config.json", problems[0])

    def test_valid_adapter_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            _write_safetensors(
                adapter / "adapters.safetensors",
                {"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps({"fine_tune_type": "lora"})
            )

            self.assertEqual(verify_adapter(adapter), [])


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        payload = json.dumps({"data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class WaitReadyTests(unittest.TestCase):
    def test_ready_when_models_endpoint_answers(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.assertTrue(wait_ready(server.server_address[1], timeout_seconds=5))
        finally:
            server.shutdown()
            server.server_close()

    def test_times_out_on_a_dead_port(self) -> None:
        dead = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
        dead_port = dead.server_address[1]
        dead.server_close()

        self.assertFalse(wait_ready(dead_port, timeout_seconds=1))


if __name__ == "__main__":
    unittest.main()
