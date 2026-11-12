"""Settings, read from environment variables so the same code runs locally, in Docker and in CI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def _prices() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("CITEWISE_PRICES", "").strip()
    if not raw:
        return {}
    return {model: (float(pair[0]), float(pair[1])) for model, pair in json.loads(raw).items()}


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # Models. The main model researches and writes the answer; the fast model triages and checks.
    model: str = field(default_factory=lambda: os.environ.get("CITEWISE_MODEL", "claude-sonnet-5-5"))
    fast_model: str = field(default_factory=lambda: os.environ.get("CITEWISE_FAST_MODEL", "claude-haiku-5-5"))
    temperature: float | None = field(default_factory=lambda: _float("CITEWISE_TEMPERATURE"))
    # json_schema uses Anthropic's native structured outputs, which the newest models need (they do
    # not accept forced tool choice). function_calling is the older tool-based method.
    structured_output: str = field(
        default_factory=lambda: os.environ.get("CITEWISE_STRUCTURED_OUTPUT", "json_schema")
    )
    # Room for adaptive thinking plus the answer. Only tokens actually used are billed.
    max_tokens: int = field(default_factory=lambda: _int("CITEWISE_MAX_TOKENS", 8192))
    effort: str | None = field(default_factory=lambda: os.environ.get("CITEWISE_EFFORT") or None)
    anthropic_base_url: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL") or None
    )

    # Agent limits.
    max_tool_calls: int = field(default_factory=lambda: _int("CITEWISE_MAX_TOOL_CALLS", 6))
    max_answer_attempts: int = field(default_factory=lambda: _int("CITEWISE_MAX_ATTEMPTS", 2))
    # strict: abstain unless the documents answer the question. best_supported: commit to the
    # verdict the evidence leans toward, saying when it is indirect.
    verdict_policy: str = field(default_factory=lambda: os.environ.get("CITEWISE_VERDICT_POLICY", "strict"))
    check_support: bool = field(default_factory=lambda: os.environ.get("CITEWISE_CHECK_SUPPORT", "1") != "0")

    # Knowledge service (run by the MCP server).
    index_path: str = field(default_factory=lambda: os.environ.get("CITEWISE_INDEX_PATH", ".citewise/index"))
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("CITEWISE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    fake_embeddings: bool = field(
        default_factory=lambda: os.environ.get("CITEWISE_FAKE_EMBEDDINGS", "0") == "1"
    )
    offline_fixture: str | None = field(
        default_factory=lambda: os.environ.get("CITEWISE_OFFLINE_FIXTURE") or None
    )
    ncbi_api_key: str | None = field(default_factory=lambda: os.environ.get("NCBI_API_KEY") or None)
    ncbi_email: str | None = field(default_factory=lambda: os.environ.get("NCBI_EMAIL") or None)

    # Cost reporting is optional. CITEWISE_PRICES is JSON mapping a model to the USD you pay per
    # million input and output tokens, e.g. {"<model id>": [input_usd, output_usd]}. Take the
    # numbers from the current price list.
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: _prices())

    # Public demo protection.
    rate_limit_per_minute: int = field(default_factory=lambda: _int("CITEWISE_RATE_LIMIT_PER_MINUTE", 5))
    daily_token_budget: int = field(default_factory=lambda: _int("CITEWISE_DAILY_TOKEN_BUDGET", 2_000_000))


# Environment variables the MCP server subprocess may see, on top of the MCP SDK's platform defaults.
# The Claude API key is not among them: the knowledge service has no reason to hold it.
SERVER_ENV_PREFIXES = ("CITEWISE_", "NCBI_", "HF_", "FASTEMBED_")
SERVER_ENV_NAMES = frozenset(
    {
        # all platforms
        "PATH",
        "HOME",
        "LANG",
        "TMPDIR",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        # Windows: Python and its network stack need these to start a subprocess
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "USERNAME",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
    }
)


def server_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    from mcp.client.stdio import get_default_environment

    env = get_default_environment()
    env.update(
        {
            key: value
            for key, value in os.environ.items()
            if key.upper() in SERVER_ENV_NAMES or key.startswith(SERVER_ENV_PREFIXES)
        }
    )
    env.update(extra or {})
    return env


def load_env_file(path: str = ".env") -> None:
    """Read settings from a .env file in the working folder, if there is one.

    Variables already set in the environment win, so CI secrets and shell exports are never
    overridden by a stray file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(path, override=False)
