"""Settings plumbing: what the MCP subprocess inherits, and .env loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from citewise.config import load_env_file, server_environment


def test_server_gets_settings_and_system_variables_but_not_the_claude_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("CITEWISE_INDEX_PATH", "/tmp/index")
    monkeypatch.setenv("NCBI_EMAIL", "me@example.com")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")  # Windows subprocesses need it
    monkeypatch.setenv("UNRELATED_SECRET", "x")
    env = server_environment({"EXTRA": "1"})
    assert env["CITEWISE_INDEX_PATH"] == "/tmp/index"
    assert env["NCBI_EMAIL"] == "me@example.com"
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["PATH"] == os.environ["PATH"]
    assert env["EXTRA"] == "1"
    assert "ANTHROPIC_API_KEY" not in env
    assert "UNRELATED_SECRET" not in env


def test_env_file_is_loaded_but_never_overrides_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CITEWISE_TEST_FROM_FILE=file\nCITEWISE_TEST_BOTH=file\n")
    monkeypatch.delenv("CITEWISE_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("CITEWISE_TEST_BOTH", "shell")
    load_env_file(str(env_file))
    assert os.environ["CITEWISE_TEST_FROM_FILE"] == "file"
    assert os.environ["CITEWISE_TEST_BOTH"] == "shell"
    monkeypatch.delenv("CITEWISE_TEST_FROM_FILE")
    load_env_file(str(tmp_path / "missing.env"))  # no file: nothing happens


def test_client_code_does_not_load_the_mcp_server() -> None:
    """Building the server object switches on its INFO logging, which floods the client's terminal."""
    import subprocess
    import sys

    code = (
        "import logging, sys, citewise.cli, citewise.eval.cli, citewise.api, citewise.toolbox; "
        "print('citewise.mcp_server' in sys.modules, len(logging.getLogger().handlers))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False 0"
