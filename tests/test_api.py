"""The HTTP API: answers, rate limiting and the daily token budget."""

from __future__ import annotations

from fastapi.testclient import TestClient

from citewise.agent import AgentResult
from citewise.api import DailyBudget, RateLimiter, create_app
from tests.conftest import good_answer


class StubAgent:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def run(self, question: str) -> AgentResult:
        self.questions.append(question)
        return AgentResult(
            question=question,
            status="answered",
            answer=good_answer("90000001"),
            refusal_reason=None,
            evidence={"90000001": {"title": "T", "year": 2014, "journal": "J", "passages": {}}},
            trace={
                "total_ms": 5,
                "total_tokens": 600,
                "tool_calls": 2,
                "cost_usd": None,
                "steps": [],
            },
        )


def _client(monkeypatch, **env: str) -> tuple[TestClient, StubAgent]:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    agent = StubAgent()

    async def factory():
        async def close() -> None:
            return None

        return agent, close

    return TestClient(create_app(factory)), agent


def test_ask_returns_answer_sources_and_disclaimer(monkeypatch) -> None:
    client, agent = _client(monkeypatch)
    with client:
        assert client.get("/health").json()["status"] == "ok"
        body = client.post("/ask", json={"question": "Does vitamin D prevent fractures?"}).json()
        assert body["answer"]["verdict"] == "no"
        assert body["sources"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/90000001/"
        assert "Not medical advice" in body["disclaimer"]
        assert "Citewise" in client.get("/").text
    assert agent.questions == ["Does vitamin D prevent fractures?"]


def test_rate_limit_per_client(monkeypatch) -> None:
    client, _ = _client(monkeypatch, CITEWISE_RATE_LIMIT_PER_MINUTE="2")
    with client:
        codes = [client.post("/ask", json={"question": "q is long enough"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_daily_budget_stops_the_demo(monkeypatch) -> None:
    client, _ = _client(monkeypatch, CITEWISE_DAILY_TOKEN_BUDGET="1000", CITEWISE_RATE_LIMIT_PER_MINUTE="100")
    with client:
        codes = [client.post("/ask", json={"question": "q is long enough"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]  # 600 + 600 tokens passes 1000 after the second question


def test_limiter_window_and_budget_units() -> None:
    limiter = RateLimiter(1)
    assert limiter.allow("a", now=0) and not limiter.allow("a", now=30) and limiter.allow("a", now=61)
    assert limiter.allow("b", now=30)
    budget = DailyBudget(10)
    budget.add(9)
    assert not budget.exhausted()
    budget.add(1)
    assert budget.exhausted()


def test_rejects_empty_questions(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    with client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
