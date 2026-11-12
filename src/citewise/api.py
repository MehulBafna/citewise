"""HTTP API and demo page. POST /ask runs the agent; GET / serves a small web page.

Protection for a demo that spends real API credits: a per-client rate limit and a daily token
budget. Both are in memory, which is enough for a single-process demo.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from citewise.config import Settings, load_env_file

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.calls: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        window = self.calls[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


class DailyBudget:
    def __init__(self, tokens_per_day: int) -> None:
        self.limit = tokens_per_day
        self.day = datetime.now(UTC).date()
        self.used = 0

    def _roll(self) -> None:
        today = datetime.now(UTC).date()
        if today != self.day:
            self.day, self.used = today, 0

    def exhausted(self) -> bool:
        self._roll()
        return self.used >= self.limit

    def add(self, tokens: int) -> None:
        self._roll()
        self.used += tokens


AgentFactory = Callable[[], Awaitable[tuple[Any, Callable[[], Awaitable[None]]]]]


async def default_agent_factory() -> tuple[Any, Callable[[], Awaitable[None]]]:
    from citewise.agent import CitewiseAgent
    from citewise.llm import ClaudeLLM
    from citewise.toolbox import MCPToolbox

    settings = Settings()
    toolbox = MCPToolbox()
    await toolbox.__aenter__()

    async def close() -> None:
        await toolbox.__aexit__(None, None, None)

    return CitewiseAgent(ClaudeLLM(settings), toolbox, settings), close


def create_app(agent_factory: AgentFactory = default_agent_factory) -> FastAPI:
    settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent, close = await agent_factory()
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        app.state.budget = DailyBudget(settings.daily_token_budget)
        try:
            yield
        finally:
            await close()

    app = FastAPI(title="Citewise", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": settings.model, "fast_model": settings.fast_model}

    @app.post("/ask")
    async def ask(body: AskRequest, request: Request) -> dict[str, Any]:
        client = request.client.host if request.client else "unknown"
        if not app.state.limiter.allow(client):
            raise HTTPException(429, "Too many questions. Please wait a minute.")
        if app.state.budget.exhausted():
            raise HTTPException(429, "The demo's daily token budget is used up. Try tomorrow.")
        result = await app.state.agent.run(body.question)
        app.state.budget.add(int(result.trace.get("total_tokens", 0)))
        return result.to_dict()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


load_env_file()  # before the app reads its settings
app = create_app()


def main() -> None:
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        "citewise.api:app",
        host=os.environ.get("CITEWISE_HOST", "127.0.0.1"),
        port=int(os.environ.get("CITEWISE_PORT", "8000")),
    )
