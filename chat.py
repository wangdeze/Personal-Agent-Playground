#!/usr/bin/env python3
"""A small CLI for testing prompts with an OpenAI-compatible chat API."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import readline  # noqa: F401  # Enable Unicode-aware editing for input().
except ImportError:
    pass


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"
PROMPT_PATH = ROOT / "prompt.md"
HISTORY_DIR = ROOT / "history"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment variables."""
    if not path.exists():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        key, separator, value = stripped.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_config(path: Path) -> dict[str, Any]:
    """Read the scalar and nested-mapping subset of YAML used by config.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config: dict[str, Any] = {}
    parents: list[tuple[int, dict[str, Any]]] = [(-1, config)]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indentation = len(line) - len(line.lstrip())
        if "\t" in line[:indentation]:
            raise ValueError(f"Invalid config indentation at {path}:{line_number}")
        while parents[-1][0] >= indentation:
            parents.pop()
        parent = parents[-1][1]
        match = re.fullmatch(r"([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*", stripped)
        if not match:
            raise ValueError(f"Invalid config at {path}:{line_number}")
        key, raw_value = match.groups()
        if key in parent:
            raise ValueError(f"Duplicate config key at {path}:{line_number}: {key}")
        value = raw_value.split(" #", 1)[0].strip()
        if not value:
            parent[key] = {}
            parents.append((indentation, parent[key]))
        elif value.startswith(("'", '"')) and value.endswith(value[0]):
            value = value[1:-1]
            parent[key] = value
        elif value.lower() in {"true", "false"}:
            value = value.lower() == "true"
            parent[key] = value
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            parent[key] = value
    return config


def validate_config(config: dict[str, Any]) -> None:
    profiles = config.get("models")
    if profiles is not None:
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("models must contain at least one named model")
        default_model = config.get("default_model", next(iter(profiles)))
        if not isinstance(default_model, str) or default_model not in profiles:
            raise ValueError("default_model must name a configured model")
        for name, profile in profiles.items():
            if not isinstance(name, str) or not isinstance(profile, dict):
                raise ValueError("Each models entry must be a named mapping")
            validate_profile(profile, f"models.{name}")
        shared = {
            key: value for key, value in config.items() if key not in {"models", "default_model"}
        }
        validate_options(shared)
        for profile in profiles.values():
            validate_options(profile)
        return
    validate_profile(config, "config")
    validate_options(config)


def validate_profile(profile: dict[str, Any], label: str) -> None:
    required = ("base_url", "model", "api_key_env")
    missing = [key for key in required if not profile.get(key)]
    if missing:
        raise ValueError(f"Missing {label} value(s): {', '.join(missing)}")
    if any(not isinstance(profile[key], str) for key in required):
        raise ValueError(f"{label} base_url, model, and api_key_env must be strings")


def validate_options(config: dict[str, Any]) -> None:
    max_turns = config.get("max_turns", 10)
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    timeout = config.get("timeout", 60)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Chat:
    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        prompt_path: Path = PROMPT_PATH,
        history_dir: Path = HISTORY_DIR,
        env_path: Path = ENV_PATH,
    ) -> None:
        self.config_path = config_path
        self.prompt_path = prompt_path
        self.history_dir = history_dir
        self.env_path = env_path
        self.config: dict[str, Any] = {}
        self.models: dict[str, dict[str, Any]] = {}
        self.default_model = "default"
        self.model_name = "default"
        self.shared_config: dict[str, Any] = {}
        self.prompt = ""
        self.messages: list[dict[str, str]] = []
        self.api_calls: list[dict[str, Any]] = []
        self.turns: list[dict[str, Any]] = []
        self.active_turn_start = 0
        self.history_path: Path | None = None
        self.reload()
        self.load_latest()

    def reload(self) -> None:
        load_dotenv(self.env_path)
        config = parse_config(self.config_path)
        validate_config(config)
        if "models" in config:
            models = config["models"]
            default_model = config.get("default_model", next(iter(models)))
            shared_config = {
                key: value
                for key, value in config.items()
                if key not in {"models", "default_model"}
            }
        else:
            models = {"default": config}
            default_model = "default"
            shared_config = {}
        selected_model = self.model_name if self.model_name in models else default_model
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        self.models = models
        self.default_model = default_model
        self.shared_config = shared_config
        self.model_name = selected_model
        self.config = {**shared_config, **models[selected_model]}
        self.prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        if not self.prompt:
            raise ValueError(f"Prompt file is empty: {self.prompt_path}")

    def select_model(self, name: str) -> None:
        if name not in self.models:
            raise ValueError(f"Unknown model '{name}'. Use /models to list configured models.")
        self.model_name = name
        self.config = {**self.shared_config, **self.models[name]}

    def load_latest(self) -> None:
        files = sorted(self.history_dir.glob("session-*.json")) if self.history_dir.exists() else []
        if not files:
            self.start_session()
            return
        self.history_path = files[-1]
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load history {self.history_path}: {error}") from error
        if "turns" in data:
            turns = data["turns"]
            active_turn_start = data.get("active_turn_start", 0)
            if (
                not isinstance(turns, list)
                or not isinstance(active_turn_start, int)
                or isinstance(active_turn_start, bool)
                or not 0 <= active_turn_start <= len(turns)
                or any(
                    not isinstance(turn, dict)
                    or not isinstance(turn.get("user_input"), str)
                    or turn.get("api_call") is not None
                    and not isinstance(turn.get("api_call"), dict)
                    or turn.get("assistant_response") is not None
                    and not isinstance(turn.get("assistant_response"), str)
                    for turn in turns
                )
            ):
                raise ValueError(f"Invalid history format: {self.history_path}")
            self.turns = turns
            self.active_turn_start = active_turn_start
            self.api_calls = [
                turn["api_call"] for turn in turns if turn.get("api_call") is not None
            ]
            self.messages = self.messages_from_turns(turns[active_turn_start:])
        else:
            messages = data.get("messages", [])
            api_calls = data.get("api_calls", [])
            if not isinstance(messages, list) or not isinstance(api_calls, list):
                raise ValueError(f"Invalid history format: {self.history_path}")
            self.turns = self.turns_from_legacy(messages, api_calls)
            self.active_turn_start = 0
            self.api_calls = [
                turn["api_call"] for turn in self.turns if turn.get("api_call") is not None
            ]
            self.messages = self.messages_from_turns(self.turns)
            self.save()
        print(f"Loaded {self.history_path.name} ({self.turn_count} turn(s)).")

    @staticmethod
    def turns_from_legacy(
        messages: list[dict[str, Any]], api_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Invalid message in legacy history")
            if message.get("role") == "user":
                turns.append(
                    {
                        "user_input": message.get("content", ""),
                        "api_call": None,
                        "assistant_response": None,
                    }
                )
            elif message.get("role") == "assistant" and turns:
                turns[-1]["assistant_response"] = message.get("content", "")
        for turn, api_call in zip(turns, api_calls):
            if isinstance(api_call, dict):
                turn["api_call"] = api_call
        return turns

    @staticmethod
    def messages_from_turns(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in turns:
            messages.append({"role": "user", "content": turn["user_input"]})
            if turn.get("assistant_response") is not None:
                messages.append(
                    {"role": "assistant", "content": turn["assistant_response"]}
                )
        return messages

    @property
    def turn_count(self) -> int:
        return sum(message.get("role") == "user" for message in self.messages)

    def start_session(self) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.history_path = self.history_dir / f"session-{stamp}.json"
        self.messages = []
        self.api_calls = []
        self.turns = []
        self.active_turn_start = 0
        self.save()

    def save(self) -> None:
        if self.history_path is None:
            self.start_session()
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": timestamp(),
            "model": self.config["model"],
            "active_turn_start": self.active_turn_start,
            "turns": self.turns,
        }
        temporary_path = self.history_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(self.history_path)

    def request(self, user_text: str) -> str:
        api_key = os.environ.get(str(self.config["api_key_env"]))
        if not api_key:
            raise ValueError(f"Set the {self.config['api_key_env']} environment variable first.")

        turn: dict[str, Any] = {
            "user_input": user_text,
            "api_call": None,
            "assistant_response": None,
        }
        self.turns.append(turn)
        self.messages.append({"role": "user", "content": user_text})
        self.save()
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": self.prompt}, *self.messages],
        }
        if "temperature" in self.config:
            payload["temperature"] = self.config["temperature"]

        url = str(self.config["base_url"]).rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        call: dict[str, Any] = {
            "timestamp": timestamp(),
            "url": url,
            "model": self.config["model"],
            "request": payload,
        }
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 60))) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            answer = response_data["choices"][0]["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("API returned a non-text assistant response")
            call["status"] = "success"
            call["response"] = response_data
            self.messages.append({"role": "assistant", "content": answer})
            turn["assistant_response"] = answer
            return answer
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            call["status"] = "error"
            call["http_status"] = error.code
            call["error"] = body
            raise RuntimeError(f"API returned HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            call["status"] = "error"
            call["error"] = str(error.reason)
            raise RuntimeError(f"Could not reach API: {error.reason}") from error
        except (TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
            call["status"] = "error"
            call["error"] = str(error)
            raise RuntimeError(f"API request failed: {error}") from error
        finally:
            call["duration_seconds"] = round(time.monotonic() - started, 3)
            self.api_calls.append(call)
            turn["api_call"] = call
            self.save()

    def reset(self) -> None:
        previous_path = self.history_path
        self.start_session()
        print(
            f"New session started. Previous history retained in "
            f"{previous_path.name if previous_path else 'the existing archive'}."
        )

    def clean(self) -> None:
        if self.history_dir.exists():
            for path in self.history_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        self.history_path = None
        self.start_session()
        print("History archives cleaned.")


HELP = """Commands:
  /models  List configured models and the current selection
  /model   <name> Switch to a configured model
  /reset   Clear conversation messages (keep API call history)
  /clean   Delete all history archives and start a new session
  /reload  Reload config.yaml and prompt.md
  /save    Save the current session
  /help    Show this help
  /quit    Exit"""


def run(chat: Chat) -> None:
    print(
        f"Prompt testing with {chat.model_name} ({chat.config['model']}). "
        "Type /help for commands."
    )
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not text:
            continue
        if text.startswith("/"):
            command_parts = text[1:].split(maxsplit=1)
            command = command_parts[0].lower() if command_parts else ""
            argument = command_parts[1] if len(command_parts) > 1 else ""
            if command in {"quit", "exit"}:
                print("Bye.")
                return
            if command == "help":
                print(HELP)
            elif command == "models":
                for name, profile in chat.models.items():
                    selected = " (current)" if name == chat.model_name else ""
                    print(f"  {name}: {profile['model']}{selected}")
            elif command == "model":
                name = argument.strip()
                if not name:
                    print("Usage: /model <name>")
                else:
                    try:
                        chat.select_model(name)
                        print(f"Switched to {chat.model_name} ({chat.config['model']}).")
                    except ValueError as error:
                        print(error)
            elif command == "save":
                chat.save()
                print(f"Saved {chat.history_path.name if chat.history_path else 'session'}.")
            elif command == "reset":
                chat.reset()
            elif command == "clean":
                chat.clean()
            elif command == "reload":
                try:
                    chat.reload()
                    print("Config and prompt reloaded.")
                except (OSError, ValueError) as error:
                    print(f"Reload failed: {error}")
            else:
                print(f"Unknown command: {text}. Type /help.")
            continue
        if chat.turn_count >= chat.config.get("max_turns", 10):
            print(f"Turn limit reached ({chat.config.get('max_turns', 10)}). Use /reset to start over.")
            continue
        try:
            print(f"assistant> {chat.request(text)}")
        except (RuntimeError, ValueError) as error:
            print(f"Error: {error}")


def main() -> int:
    try:
        run(Chat())
    except (OSError, ValueError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
