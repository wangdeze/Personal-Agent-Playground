import importlib.util
import io
import json
import os
import select
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import chat


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            "base_url: https://example.test/v1\n"
            "model: test-model\n"
            "api_key_env: TEST_API_KEY\n"
            "max_turns: 2\n",
            encoding="utf-8",
        )
        self.prompt_path = self.root / "prompt.md"
        self.prompt_path.write_text("test system prompt", encoding="utf-8")
        self.history_dir = self.root / "history"

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_chat(self):
        return chat.Chat(self.config_path, self.prompt_path, self.history_dir)

    def test_config_parses_scalars_and_comments(self):
        config = chat.parse_config(self.config_path)
        self.assertEqual(config["max_turns"], 2)
        self.assertEqual(config["base_url"], "https://example.test/v1")

    def test_named_models_parse_and_switch(self):
        self.config_path.write_text(
            "default_model: provider-a\n"
            "models:\n"
            "  provider-a:\n"
            "    base_url: https://a.example/v1\n"
            "    model: model-a\n"
            "    api_key_env: A_API_KEY\n"
            "  provider-b:\n"
            "    base_url: https://b.example/v1\n"
            "    model: model-b\n"
            "    api_key_env: B_API_KEY\n"
            "max_turns: 4\n",
            encoding="utf-8",
        )
        instance = self.make_chat()

        self.assertEqual(instance.model_name, "provider-a")
        self.assertEqual(instance.config["max_turns"], 4)
        instance.select_model("provider-b")
        self.assertEqual(instance.config["base_url"], "https://b.example/v1")
        self.assertEqual(instance.config["model"], "model-b")
        self.assertEqual(instance.config["api_key_env"], "B_API_KEY")
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            instance.select_model("missing")

    def test_project_config_selects_deepseek_by_default(self):
        config = chat.parse_config(chat.CONFIG_PATH)
        chat.validate_config(config)
        self.config_path.write_text(
            chat.CONFIG_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        instance = self.make_chat()

        self.assertEqual(instance.model_name, "deepseek")
        self.assertEqual(instance.config["base_url"], "https://api.deepseek.com")
        self.assertEqual(instance.config["model"], "deepseek-flash")
        self.assertEqual(instance.config["api_key_env"], "DEEPSEEK_API_KEY")

    def test_cli_lists_and_switches_models(self):
        self.config_path.write_text(
            "default_model: provider-a\n"
            "models:\n"
            "  provider-a:\n"
            "    base_url: https://a.example/v1\n"
            "    model: model-a\n"
            "    api_key_env: A_API_KEY\n"
            "  provider-b:\n"
            "    base_url: https://b.example/v1\n"
            "    model: model-b\n"
            "    api_key_env: B_API_KEY\n",
            encoding="utf-8",
        )
        instance = self.make_chat()
        output = io.StringIO()
        with patch("builtins.input", side_effect=["/models", "/model provider-b", "/quit"]):
            with patch("sys.stdout", output):
                chat.run(instance)

        self.assertIn("provider-a: model-a (current)", output.getvalue())
        self.assertIn("Switched to provider-b (model-b).", output.getvalue())
        self.assertEqual(instance.model_name, "provider-b")

    def test_load_dotenv_sets_values_without_overriding_environment(self):
        dotenv_path = self.root / ".env"
        dotenv_path.write_text(
            "# local secrets\nDOTENV_TEST_KEY='from-file'\nexport DOTENV_OVERRIDE=from-file\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"DOTENV_OVERRIDE": "from-shell"}, clear=True):
            chat.load_dotenv(dotenv_path)
            self.assertEqual(os.environ["DOTENV_TEST_KEY"], "from-file")
            self.assertEqual(os.environ["DOTENV_OVERRIDE"], "from-shell")

    def test_session_saves_and_loads_messages_and_api_calls(self):
        instance = self.make_chat()
        instance.turns = [
            {
                "user_input": "hello",
                "api_call": {"status": "success"},
                "assistant_response": "hello back",
            }
        ]
        instance.api_calls = [instance.turns[0]["api_call"]]
        instance.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello back"},
        ]
        instance.save()

        loaded = self.make_chat()
        self.assertEqual(loaded.messages, instance.messages)
        self.assertEqual(loaded.api_calls, instance.api_calls)
        self.assertEqual(loaded.turns, instance.turns)

    def test_legacy_history_is_migrated_to_turns(self):
        self.history_dir.mkdir()
        history_path = self.history_dir / "session-legacy.json"
        history_path.write_text(
            json.dumps(
                {
                    "model": "test-model",
                    "messages": [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "reply"},
                    ],
                    "api_calls": [{"status": "success"}],
                }
            ),
            encoding="utf-8",
        )

        loaded = self.make_chat()
        migrated = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertNotIn("messages", migrated)
        self.assertNotIn("api_calls", migrated)
        self.assertEqual(
            list(migrated["turns"][0]),
            ["user_input", "api_call", "assistant_response"],
        )
        self.assertEqual(
            migrated["turns"][0],
            {
                "user_input": "first",
                "api_call": {"status": "success"},
                "assistant_response": "reply",
            },
        )
        self.assertEqual(loaded.messages[-1]["content"], "reply")

    def test_request_sends_prompt_and_persists_api_history(self):
        instance = self.make_chat()
        response_data = {"choices": [{"message": {"content": "hello back"}}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(response_data).encode()

        with patch.dict(os.environ, {"TEST_API_KEY": "test-secret"}):
            with patch("chat.urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
                answer = instance.request("hello")

        self.assertEqual(answer, "hello back")
        sent_request = urlopen.call_args.args[0]
        self.assertEqual(sent_request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(instance.messages[-1], {"role": "assistant", "content": "hello back"})
        saved = json.loads(instance.history_path.read_text(encoding="utf-8"))
        self.assertEqual(
            list(saved["turns"][0]),
            ["user_input", "api_call", "assistant_response"],
        )
        self.assertEqual(saved["turns"][0]["user_input"], "hello")
        self.assertEqual(saved["turns"][0]["api_call"]["status"], "success")
        self.assertEqual(saved["turns"][0]["assistant_response"], "hello back")
        self.assertNotIn("test-secret", instance.history_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(os.name == "posix", "requires a POSIX terminal")
    def test_long_cjk_input_can_be_corrected_after_wrapping(self):
        if importlib.util.find_spec("readline") is None:
            self.skipTest("Python readline is unavailable")

        import fcntl
        import pty
        import termios

        master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 20, 0, 0))
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import chat, json; print('RESULT:' + json.dumps(input('you> ')))",
            ],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=Path(__file__).resolve().parent,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        os.close(slave_fd)

        def read_until(marker: bytes, output: bytearray, start: int = 0) -> None:
            deadline = time.monotonic() + 5
            while marker not in output[start:] and time.monotonic() < deadline:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
                if ready:
                    try:
                        output.extend(os.read(master_fd, 65536))
                    except OSError:
                        break
            self.assertIn(marker, output[start:])

        try:
            output = bytearray()
            read_until(b"you> ", output)
            os.write(master_fd, ("你" * 30 + "XYZ" + "\x7f" * 5 + "B\n").encode())
            read_until(b"RESULT:", output)
            result_start = output.index(b"RESULT:") + len(b"RESULT:")
            read_until(b"\n", output, result_start)
            self.assertEqual(
                json.loads(output[result_start:].splitlines()[0].decode("utf-8")),
                "你" * 28 + "B",
            )
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            os.close(master_fd)

    def test_reset_starts_new_archive_and_clean_removes_old_archives(self):
        instance = self.make_chat()
        instance.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello back"},
        ]
        instance.turns = [
            {
                "user_input": "hello",
                "api_call": {"status": "success"},
                "assistant_response": "hello back",
            }
        ]
        instance.api_calls = [instance.turns[0]["api_call"]]
        instance.save()
        old_path = instance.history_path
        old_contents = old_path.read_text(encoding="utf-8")

        instance.reset()
        new_path = instance.history_path
        self.assertNotEqual(new_path, old_path)
        self.assertTrue(old_path.exists())
        self.assertEqual(old_path.read_text(encoding="utf-8"), old_contents)
        self.assertEqual(instance.turns, [])
        self.assertEqual(instance.messages, [])
        self.assertEqual(instance.api_calls, [])

        loaded = self.make_chat()
        self.assertEqual(loaded.messages, [])
        self.assertEqual(loaded.api_calls, [])
        self.assertEqual(loaded.history_path, new_path)

        instance.clean()
        self.assertFalse(old_path.exists())
        self.assertFalse(new_path.exists())
        self.assertEqual(instance.api_calls, [])

    def test_request_uses_selected_provider_and_credentials(self):
        self.config_path.write_text(
            "default_model: provider-a\n"
            "models:\n"
            "  provider-a:\n"
            "    base_url: https://a.example/v1\n"
            "    model: model-a\n"
            "    api_key_env: A_API_KEY\n"
            "  provider-b:\n"
            "    base_url: https://b.example/v1\n"
            "    model: model-b\n"
            "    api_key_env: B_API_KEY\n",
            encoding="utf-8",
        )
        instance = self.make_chat()
        instance.select_model("provider-b")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "reply"}}]}
                ).encode()

        with patch.dict(os.environ, {"B_API_KEY": "provider-b-secret"}):
            with patch("chat.urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
                instance.request("hello")

        sent_request = urlopen.call_args.args[0]
        payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(sent_request.full_url, "https://b.example/v1/chat/completions")
        self.assertEqual(payload["model"], "model-b")
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer provider-b-secret")


if __name__ == "__main__":
    unittest.main()
