"""Shared test helpers: an offline knowledge service, a scripted fake LLM and in-process tools.

No test needs network access or an API key.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from citewise import mcp_server
from citewise.config import Settings
from citewise.schemas import Answer, Claim, ClaimCheck, QuickAnswer, SupportReport, Triage
from citewise.toolbox import model_facing_schema
from citewise.tracing import current_trace

FIXTURES = Path(__file__).parent / "fixtures"
PAPERS = FIXTURES / "papers.json"


@pytest.fixture(autouse=True)
def offline_knowledge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a fresh in-memory index over the synthetic papers."""
    monkeypatch.setenv("CITEWISE_OFFLINE_FIXTURE", str(PAPERS))
    monkeypatch.setenv("CITEWISE_FAKE_EMBEDDINGS", "1")
    monkeypatch.setenv("CITEWISE_INDEX_PATH", ":memory:")
    monkeypatch.delenv("CITEWISE_PRICES", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    mcp_server.knowledge.cache_clear()
    yield
    mcp_server.knowledge.cache_clear()


def settings(**overrides: Any) -> Settings:
    base = Settings()
    return Settings(**{**base.__dict__, **overrides})


def _usage(model: str, inp: int = 100, out: int = 20) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.add_usage(model, inp, out)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


AnswerSpec = Answer | Callable[[str, str | None], Answer]


class FakeLLM:
    """A scripted stand-in for Claude that records what it was asked."""

    def __init__(
        self,
        *,
        triage: Triage | None = None,
        research: list[AIMessage] | Callable[[int], AIMessage] | None = None,
        answers: list[AnswerSpec] | None = None,
        support: Callable[[list[tuple[str, str]]], SupportReport] | None = None,
        verdict: str = "yes",
    ) -> None:
        self.triage_result = triage or Triage(
            category="research_question", reason="research", search_queries=["vitamin D fracture"]
        )
        self.research_script = research
        self.answers = list(answers or [])
        self.support = support
        self.verdict = verdict
        self.calls: dict[str, list[Any]] = defaultdict(list)

    async def triage(self, question: str) -> Triage:
        self.calls["triage"].append(question)
        _usage("fake-fast")
        return self.triage_result

    async def research(self, messages: list, tools: list[dict]) -> AIMessage:
        self.calls["research"].append({"messages": list(messages), "tools": tools})
        _usage("fake-main")
        step = len(self.calls["research"]) - 1
        if callable(self.research_script):
            return self.research_script(step)
        script = self.research_script or []
        return script[step] if step < len(script) else AIMessage(content="I have enough evidence.")

    async def answer(self, question: str, evidence: str, feedback: str | None) -> Answer:
        self.calls["answer"].append({"evidence": evidence, "feedback": feedback})
        _usage("fake-main", 400, 120)
        spec = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return spec(evidence, feedback) if callable(spec) else spec

    async def check_support(self, claims: list[tuple[str, str]]) -> SupportReport:
        self.calls["check_support"].append(claims)
        _usage("fake-fast")
        if self.support:
            return self.support(claims)
        return SupportReport(
            checks=[ClaimCheck(index=i, supported=True, reason="ok") for i in range(len(claims))]
        )

    async def closed_book(self, question: str) -> QuickAnswer:
        self.calls["closed_book"].append(question)
        _usage("fake-main")
        return QuickAnswer(reasoning="from memory", verdict=self.verdict)

    async def with_context(self, question: str, contexts: list[str]) -> QuickAnswer:
        self.calls["with_context"].append(question)
        _usage("fake-main")
        return QuickAnswer(reasoning="from context", verdict=self.verdict)


class InProcessToolbox:
    """Calls the real MCP server tool functions directly, without a subprocess."""

    def __init__(self, inject: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.inject = inject or []  # extra records appended to every result (simulates a bad server)

    def specs(self) -> list[dict[str, Any]]:
        specs = []
        for tool in mcp_server.mcp._tool_manager.list_tools():
            specs.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": model_facing_schema(tool.parameters),
                }
            )
        return specs

    async def call(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append((name, dict(args)))
        raw = getattr(mcp_server, name)(**args)
        if self.inject:
            raw = json.dumps(json.loads(raw) + self.inject)
        return raw


def good_answer(
    *pmids: str, verdict: str = "no", primary: str | None = None, study: str | None = None
) -> Answer:
    return Answer(
        primary_pmid=primary,
        study_verdict=study or ("no_matching_study" if primary is None else verdict),
        verdict=verdict,
        summary="Vitamin D alone did not reduce hip fractures.",
        claims=[Claim(statement="Vitamin D alone did not reduce hip fractures.", pmids=list(pmids))],
    )
