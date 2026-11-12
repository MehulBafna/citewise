"""Integration tests on the real code paths, still offline.

1. The MCP server runs as a subprocess and the agent talks to it over stdio via
   langchain-mcp-adapters, exactly as in production.
2. The Claude client (ChatAnthropic, tool calling, structured output) runs against a local mock of
   the Anthropic Messages API, so request and response handling are tested without an API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from citewise.agent import CitewiseAgent
from citewise.guardrails import EvidencePolicy
from citewise.llm import ClaudeLLM
from citewise.toolbox import MCPToolbox
from tests.conftest import FakeLLM, InProcessToolbox, good_answer, settings, tool_call


async def test_agent_over_a_real_mcp_stdio_session() -> None:
    llm = FakeLLM(
        research=[tool_call("search_pubmed", {"query": "vitamin D hip fracture"})],
        answers=[good_answer("90000003")],
    )
    policy = EvidencePolicy(exclude_pmids=frozenset({"90000001"}), max_year=2016)
    async with MCPToolbox() as toolbox:
        assert sorted(toolbox.tools) == ["get_abstracts", "search_evidence", "search_pubmed"]
        for spec in toolbox.specs():
            assert not {"exclude_pmids", "max_year"} & set(spec["input_schema"]["properties"])
        result = await CitewiseAgent(llm, toolbox, settings()).run(
            "Does vitamin D supplementation reduce hip fracture risk?", policy
        )
    assert result.status == "answered"
    assert [s["pmid"] for s in result.sources()] == ["90000003"]
    assert "90000001" not in result.evidence and "90000004" not in result.evidence


# --- a mock Anthropic Messages API -----------------------------------------------------------


def _tool_use(name: str, payload: dict[str, Any], call_id: str = "toolu_s") -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": payload}


STRUCTURED = {
    "Triage": {
        "category": "research_question",
        "reason": "A question about published trials.",
        "search_queries": ["vitamin D hip fracture"],
    },
    "Answer": {
        "verdict": "no",
        "summary": "Vitamin D alone did not reduce fractures.",
        "claims": [
            {
                "statement": "Vitamin D alone did not reduce hip or total fractures.",
                "pmids": ["90000003"],
            }
        ],
    },
    "SupportReport": {"checks": [{"index": 0, "supported": True, "reason": "Stated directly."}]},
    "QuickAnswer": {"reasoning": "Trials were negative.", "verdict": "no"},
}


SCHEMA_BY_FIELDS = {
    frozenset({"category", "reason", "search_queries"}): "Triage",
    frozenset({"verdict", "summary", "claims"}): "Answer",
    frozenset({"checks"}): "SupportReport",
    frozenset({"reasoning", "verdict"}): "QuickAnswer",
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        forced = (body.get("tool_choice") or {}).get("name")
        output_format = (body.get("output_config") or {}).get("format")
        if output_format:  # native structured output: reply with JSON text matching the schema
            name = SCHEMA_BY_FIELDS[frozenset(output_format["schema"]["properties"])]
            self.server.structured.append(name)  # type: ignore[attr-defined]
            content, stop = [{"type": "text", "text": json.dumps(STRUCTURED[name])}], "end_turn"
        elif forced:  # the older forced-tool method
            self.server.structured.append(forced)  # type: ignore[attr-defined]
            content, stop = [_tool_use(forced, STRUCTURED[forced])], "tool_use"
        else:
            answered = any(
                isinstance(m["content"], list)
                and any(block.get("type") == "tool_result" for block in m["content"])
                for m in body["messages"]
            )
            if answered:
                content, stop = [{"type": "text", "text": "Enough evidence."}], "end_turn"
            else:
                content = [
                    {"type": "text", "text": "Searching PubMed."},
                    _tool_use("search_pubmed", {"query": "vitamin D hip fracture"}, "toolu_r1"),
                ]
                stop = "tool_use"
        reply = {
            "id": f"msg_{len(self.server.requests)}",  # type: ignore[attr-defined]
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 30},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []  # type: ignore[attr-defined]
    server.structured = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    yield server
    server.shutdown()


def _claude(server, **overrides: Any) -> tuple[ClaudeLLM, Any]:
    # The real model names matter: langchain-anthropic changes behaviour per model.
    values = {
        "anthropic_base_url": f"http://127.0.0.1:{server.server_address[1]}",
        "model": "claude-sonnet-5-5",
        "fast_model": "claude-haiku-5-5",
        **overrides,
    }
    s = settings(**values)
    return ClaudeLLM(s), s


async def test_the_real_claude_code_path_against_a_mock_api(mock_anthropic) -> None:
    llm, s = _claude(mock_anthropic)
    result = await CitewiseAgent(llm, InProcessToolbox(), s).run(
        "Does vitamin D supplementation reduce hip fracture risk?"
    )
    assert result.status == "answered", result
    assert result.answer.verdict == "no"
    assert [x["pmid"] for x in result.sources()] == ["90000003"]

    requests = mock_anthropic.requests
    # structured outputs use Anthropic's native output format, never forced tool choice
    assert mock_anthropic.structured == ["Triage", "Answer", "SupportReport"]
    assert not any(r.get("tool_choice") for r in requests)
    assert all(r["max_tokens"] == 8192 for r in requests)
    research = [r for r in requests if r.get("tools")]
    assert len(research) == 2
    tool_names = {t["name"] for t in research[0]["tools"]}
    assert tool_names == {"search_pubmed", "get_abstracts", "search_evidence"}
    for tool in research[0]["tools"]:
        assert "exclude_pmids" not in tool["input_schema"]["properties"]
    # the second research request carries the tool result for the first call
    results = [
        block
        for m in research[1]["messages"]
        if isinstance(m["content"], list)
        for block in m["content"]
        if block.get("type") == "tool_result"
    ]
    assert results and results[0]["tool_use_id"] == "toolu_r1"
    assert "90000003" in json.dumps(results[0]["content"])
    # the system prompt went in the system field, and the models were the configured ones
    assert "untrusted" in json.dumps(research[0]["system"])
    assert {r["model"] for r in requests} == {"claude-sonnet-5-5", "claude-haiku-5-5"}

    usage = result.trace["usage"]
    assert usage["claude-sonnet-5-5"] == {"input_tokens": 360, "output_tokens": 90, "calls": 3}
    assert usage["claude-haiku-5-5"] == {"input_tokens": 240, "output_tokens": 60, "calls": 2}


async def test_baseline_modes_against_the_mock_api(mock_anthropic) -> None:
    llm, _ = _claude(mock_anthropic)
    assert (await llm.closed_book("Does vitamin D prevent fractures?")).verdict == "no"
    reply = await llm.with_context("Does vitamin D prevent fractures?", ["Trial text."])
    assert reply.verdict == "no"
    assert "Trial text." in json.dumps(mock_anthropic.requests[-1]["messages"])


async def test_the_older_forced_tool_method_still_works_where_supported(mock_anthropic) -> None:
    llm, _ = _claude(mock_anthropic, model="claude-haiku-5-5", structured_output="function_calling")
    assert (await llm.closed_book("Does vitamin D prevent fractures?")).verdict == "no"
    assert mock_anthropic.requests[-1]["tool_choice"]["name"] == "QuickAnswer"
