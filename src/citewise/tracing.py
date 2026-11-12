"""Per-request traces: time per step, tokens per model, tool calls and estimated cost.

A trace lives in a context variable, so concurrent requests each get their own. Every request also
writes one JSON log line. If Langfuse keys are set and the `tracing` extra is installed, model calls
are sent to Langfuse as well.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("citewise")

current_trace: ContextVar[Trace | None] = ContextVar("citewise_trace", default=None)


@dataclass
class Trace:
    started: float = field(default_factory=time.perf_counter)
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        entry = self.usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        entry["input_tokens"] += int(input_tokens)
        entry["output_tokens"] += int(output_tokens)
        entry["calls"] += 1

    @contextmanager
    def step(self, name: str) -> Iterator[dict[str, Any]]:
        start = time.perf_counter()
        info: dict[str, Any] = {"step": name}
        try:
            yield info
        finally:
            info["ms"] = round((time.perf_counter() - start) * 1000)
            self.steps.append(info)

    @property
    def total_tokens(self) -> int:
        return sum(u["input_tokens"] + u["output_tokens"] for u in self.usage.values())

    def cost_usd(self, prices: dict[str, tuple[float, float]]) -> float | None:
        if not self.usage or any(model not in prices for model in self.usage):
            return None
        return round(
            sum(
                u["input_tokens"] / 1e6 * prices[m][0] + u["output_tokens"] / 1e6 * prices[m][1]
                for m, u in self.usage.items()
            ),
            6,
        )

    def summary(self, prices: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
        return {
            "total_ms": round((time.perf_counter() - self.started) * 1000),
            "steps": self.steps,
            "usage": self.usage,
            "total_tokens": self.total_tokens,
            "tool_calls": len(self.tool_calls),
            "cost_usd": self.cost_usd(prices or {}),
        }


def record_usage(model: str, message: Any) -> None:
    """Add a model response's token usage to the current trace, if there is one."""
    trace = current_trace.get()
    usage = getattr(message, "usage_metadata", None)
    if trace is not None and usage:
        trace.add_usage(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def log_event(event: str, **fields: Any) -> None:
    log.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def langchain_callbacks() -> list[Any]:
    """A Langfuse callback handler when configured, otherwise nothing."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse >= 3
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # langfuse 2
        except ImportError:
            log.warning("LANGFUSE keys are set but langfuse is not installed: pip install .[tracing]")
            return []
    return [CallbackHandler()]
