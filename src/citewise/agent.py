"""The research agent, as a LangGraph state graph.

    guard_input -> triage -> research <-> tools -> gather -> answer -> verify -> END
         |           |                                          ^         |
         +-----------+--> END (refused)                         +---------+ (retry with feedback)

- guard_input: deterministic input checks (length, prompt injection). No model call.
- triage:      the fast model refuses personal medical advice and off-topic requests, and suggests
               PubMed queries.
- research:    the main model calls MCP tools (search PubMed, read abstracts, search the index)
               within a tool budget.
- tools:       runs the calls with the evidence policy injected, and filters every result again.
- gather:      a deterministic retrieval step: semantic search of the index for the question, so the
               answer is grounded even if the model searched poorly.
- answer:      the main model answers from the evidence only, citing PMIDs per claim.
- verify:      citation checks, then a support check by the fast model. Failures go back to answer
               once with feedback; claims that still fail are removed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from citewise.config import Settings
from citewise.guardrails import (
    DISCLAIMER,
    POLICY_PARAMS,
    REFUSALS,
    EvidencePolicy,
    check_input,
    validate_answer,
    wrap_document,
)
from citewise.llm import LLM, RESEARCH_SYSTEM
from citewise.schemas import Answer
from citewise.toolbox import Toolbox
from citewise.tracing import Trace, current_trace, log_event

MAX_DOCUMENTS = 12
MAX_DOC_CHARS = 2500
MAX_CITED_CHARS = 1500
GATHER_K = 10
TOOL_TEXT_LIMIT = 4000
ANSWERED_VERDICTS = ("yes", "no", "maybe", "not_applicable")


class State(TypedDict, total=False):
    question: str
    policy: EvidencePolicy
    status: str  # running, refused or answered
    refusal_reason: str
    search_queries: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_used: int
    evidence: dict[str, dict[str, Any]]  # pmid -> title, year, journal, passages {section: text}
    gather_order: list[str]
    answer: Answer
    attempts: int
    feedback: str | None
    removed_claims: list[dict[str, Any]]


def _step(name: str) -> AbstractContextManager[dict[str, Any]]:
    trace = current_trace.get()
    return trace.step(name) if trace is not None else nullcontext({})


def parse_items(raw: str) -> list[dict[str, Any]] | None:
    """Tool output as a list of records, or None if it is not one (e.g. an error message)."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    return None


def merge_evidence(
    evidence: dict[str, dict[str, Any]], items: list[dict[str, Any]], policy: EvidencePolicy
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Add tool results to the evidence, dropping anything the policy does not allow."""
    merged = {pmid: {**doc, "passages": dict(doc["passages"])} for pmid, doc in evidence.items()}
    kept = []
    for item in items:
        pmid = str(item.get("pmid", ""))
        year = item.get("year")
        if not pmid or not policy.allows(pmid, year):
            continue
        kept.append(item)
        doc = merged.setdefault(
            pmid,
            {
                "title": item.get("title", ""),
                "year": year,
                "journal": item.get("journal", ""),
                "passages": {},
            },
        )
        if item.get("abstract"):
            doc["passages"]["ABSTRACT"] = item["abstract"]
        elif item.get("text"):
            doc["passages"][item.get("section") or "PASSAGE"] = item["text"]
        elif item.get("snippet") and not doc["passages"]:
            doc["passages"]["SNIPPET"] = item["snippet"]
    return merged, kept


def document_body(doc: dict[str, Any]) -> str:
    passages = doc["passages"]
    if "ABSTRACT" in passages:
        return passages["ABSTRACT"]
    parts = {k: v for k, v in passages.items() if k != "SNIPPET"} or passages
    return "\n".join(f"{section}: {text}" for section, text in parts.items())


@dataclass
class AgentResult:
    question: str
    status: str
    answer: Answer | None
    refusal_reason: str | None
    evidence: dict[str, dict[str, Any]]
    removed_claims: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def refusal(self) -> str | None:
        return REFUSALS.get(self.refusal_reason or "", None)

    def sources(self) -> list[dict[str, Any]]:
        if self.answer is None:
            return []
        cited = dict.fromkeys(pmid for claim in self.answer.claims for pmid in claim.pmids)
        return [
            {
                "pmid": pmid,
                "title": self.evidence[pmid]["title"],
                "year": self.evidence[pmid]["year"],
                "journal": self.evidence[pmid]["journal"],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
            for pmid in cited
            if pmid in self.evidence
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "refusal": self.refusal,
            "answer": self.answer.model_dump() if self.answer else None,
            "sources": self.sources(),
            "removed_claims": self.removed_claims,
            "tool_calls": self.tool_calls,
            "trace": self.trace,
            "disclaimer": DISCLAIMER,
        }


def describe_tool_call(name: str, args: dict[str, Any]) -> str:
    if name == "search_pubmed":
        return f"Searching PubMed: {args.get('query', '')}"
    if name == "get_abstracts":
        return f"Reading {len(args.get('pmids') or [])} abstract(s)"
    if name == "search_evidence":
        return "Searching the evidence index for the most relevant passages"
    return f"Calling {name}"


class CitewiseAgent:
    def __init__(
        self,
        llm: LLM,
        toolbox: Toolbox,
        settings: Settings | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.settings = settings or Settings()
        self.progress = progress or (lambda message: None)
        self.graph = self._build()

    # graph ----------------------------------------------------------------------------------

    def _build(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("guard_input", self.guard_input)
        graph.add_node("triage", self.triage)
        graph.add_node("research", self.research)
        graph.add_node("tools", self.tools)
        graph.add_node("gather", self.gather)
        graph.add_node("answer", self.answer)
        graph.add_node("verify", self.verify)
        graph.add_edge(START, "guard_input")
        graph.add_conditional_edges("guard_input", self._unless_refused("triage"), ["triage", END])
        graph.add_conditional_edges("triage", self._unless_refused("research"), ["research", END])
        graph.add_conditional_edges("research", self._after_research, ["tools", "gather"])
        graph.add_edge("tools", "research")
        graph.add_edge("gather", "answer")
        graph.add_edge("answer", "verify")
        graph.add_conditional_edges("verify", self._after_verify, ["answer", END])
        return graph.compile()

    @staticmethod
    def _unless_refused(next_node: str) -> Any:
        return lambda state: END if state.get("status") == "refused" else next_node

    def _after_research(self, state: State) -> str:
        last = state["messages"][-1]
        wants_tools = bool(getattr(last, "tool_calls", None))
        within_budget = state.get("tool_calls_used", 0) < self.settings.max_tool_calls
        return "tools" if wants_tools and within_budget else "gather"

    @staticmethod
    def _after_verify(state: State) -> str:
        return END if state.get("status") == "answered" else "answer"

    # nodes ----------------------------------------------------------------------------------

    async def guard_input(self, state: State) -> dict[str, Any]:
        with _step("guard_input"):
            check = check_input(state["question"])
        if not check.allowed:
            return {"status": "refused", "refusal_reason": check.reason}
        return {"status": "running"}

    async def triage(self, state: State) -> dict[str, Any]:
        self.progress("Checking the question")
        with _step("triage") as info:
            triage = await self.llm.triage(state["question"])
            info["category"] = triage.category
        if triage.category != "research_question":
            return {"status": "refused", "refusal_reason": triage.category}
        queries = [q.strip() for q in triage.search_queries if q.strip()][:3]
        hint = f"\nSuggested searches: {'; '.join(queries)}" if queries else ""
        return {
            "search_queries": queries,
            "messages": [
                SystemMessage(RESEARCH_SYSTEM),
                HumanMessage(f"Question: {state['question']}{hint}"),
            ],
            "tool_calls_used": 0,
            "evidence": {},
            "attempts": 0,
        }

    async def research(self, state: State) -> dict[str, Any]:
        self.progress("Planning the next search" if state.get("tool_calls_used") else "Planning the research")
        with _step("research"):
            reply = await self.llm.research(state["messages"], self.toolbox.specs())
        return {"messages": [reply]}

    async def _call_tool(
        self, name: str, model_args: dict[str, Any], state: State, evidence: dict
    ) -> tuple[str, dict, list[dict[str, Any]]]:
        policy = state["policy"]
        args = {k: v for k, v in model_args.items() if k not in POLICY_PARAMS}
        self.progress(describe_tool_call(name, args))
        with _step(f"tool:{name}") as info:
            raw = await self.toolbox.call(name, {**args, **policy.tool_args()})
            items = parse_items(raw)
            if items is None:
                info["error"] = True
                content = raw[:TOOL_TEXT_LIMIT]
                kept: list[dict[str, Any]] = []
            else:
                evidence, kept = merge_evidence(evidence, items, policy)
                info["results"] = len(kept)
                info["dropped_by_policy"] = len(items) - len(kept)
                content = json.dumps(kept, ensure_ascii=False)
        trace = current_trace.get()
        if trace is not None:
            trace.tool_calls.append({"name": name, "args": args, "results": len(kept)})
        return content, evidence, kept

    async def tools(self, state: State) -> dict[str, Any]:
        used = state.get("tool_calls_used", 0)
        evidence = state.get("evidence", {})
        messages = []
        for call in state["messages"][-1].tool_calls:
            if used >= self.settings.max_tool_calls:
                content = "Tool budget used up. Write your note now."
            else:
                content, evidence, _ = await self._call_tool(call["name"], call["args"], state, evidence)
                used += 1
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": messages, "tool_calls_used": used, "evidence": evidence}

    async def gather(self, state: State) -> dict[str, Any]:
        evidence = state.get("evidence", {})
        if state.get("tool_calls_used", 0) == 0:  # the model did not search: search for it
            for query in state.get("search_queries") or [state["question"]]:
                _, evidence, _ = await self._call_tool(
                    "search_pubmed", {"query": query, "max_results": 8}, state, evidence
                )
        _, evidence, kept = await self._call_tool(
            "search_evidence", {"question": state["question"], "k": GATHER_K}, state, evidence
        )
        order = list(dict.fromkeys(str(item["pmid"]) for item in kept))
        return {"evidence": evidence, "gather_order": order}

    def _format_evidence(self, evidence: dict[str, dict], order: list[str]) -> str:
        pmids = list(dict.fromkeys([p for p in order if p in evidence] + list(evidence)))
        return "\n\n".join(
            wrap_document(
                pmid,
                evidence[pmid]["title"],
                evidence[pmid]["year"],
                document_body(evidence[pmid])[:MAX_DOC_CHARS],
            )
            for pmid in pmids[:MAX_DOCUMENTS]
        )

    async def answer(self, state: State) -> dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        evidence = state.get("evidence", {})
        if not evidence:
            empty = Answer(
                verdict="insufficient_evidence",
                summary="No relevant evidence was found in PubMed.",
                claims=[],
            )
            return {"answer": empty, "attempts": attempts}
        if attempts == 1:
            self.progress(f"Writing the answer from {min(len(evidence), MAX_DOCUMENTS)} papers")
        else:
            self.progress("Rewriting the answer to fix its citations")
        with _step("answer") as info:
            info["attempt"] = attempts
            answer = await self.llm.answer(
                state["question"],
                self._format_evidence(evidence, state.get("gather_order", [])),
                state.get("feedback"),
            )
        return {"answer": answer, "attempts": attempts}

    def _cited_text(self, pmids: list[str], evidence: dict[str, dict]) -> str:
        return "\n".join(
            f"[PMID {pmid}] {document_body(evidence[pmid])[:MAX_CITED_CHARS]}"
            for pmid in pmids
            if pmid in evidence
        )

    async def verify(self, state: State) -> dict[str, Any]:
        answer = state["answer"]
        evidence = state.get("evidence", {})
        self.progress("Checking every citation against its source")
        with _step("verify") as info:
            problems = validate_answer(answer, set(evidence), state["policy"])
            bad = {p.index for p in problems}
            notes = [f"Claim {p.index}: {p.problem}" for p in problems]
            if answer.verdict in ANSWERED_VERDICTS and not answer.claims:
                notes.append("The verdict is not backed by any cited claim.")
            to_check = [(i, c) for i, c in enumerate(answer.claims) if i not in bad]
            if self.settings.check_support and to_check:
                report = await self.llm.check_support(
                    [(c.statement, self._cited_text(c.pmids, evidence)) for _, c in to_check]
                )
                for check in report.checks:
                    if 0 <= check.index < len(to_check) and not check.supported:
                        original = to_check[check.index][0]
                        bad.add(original)
                        notes.append(f"Claim {original}: not supported ({check.reason})")
            info["problems"] = len(notes)

        if not notes:
            return {"status": "answered", "feedback": None}
        if state.get("attempts", 0) < self.settings.max_answer_attempts:
            return {"feedback": "\n".join(notes)}

        kept = [c for i, c in enumerate(answer.claims) if i not in bad]
        removed = [
            {"statement": c.statement, "pmids": c.pmids} for i, c in enumerate(answer.claims) if i in bad
        ]
        if kept:
            final = Answer(
                verdict=answer.verdict,
                summary=" ".join(c.statement for c in kept),
                claims=kept,
            )
        else:
            final = Answer(
                verdict="insufficient_evidence",
                summary="The evidence found did not support a verifiable answer.",
                claims=[],
            )
        return {"answer": final, "status": "answered", "removed_claims": removed, "feedback": None}

    # entry point ----------------------------------------------------------------------------

    async def run(self, question: str, policy: EvidencePolicy | None = None) -> AgentResult:
        trace = Trace()
        token = current_trace.set(trace)
        try:
            state = await self.graph.ainvoke(
                {"question": question, "policy": policy or EvidencePolicy()},
                config={"recursion_limit": 60},
            )
        finally:
            current_trace.reset(token)
        result = AgentResult(
            question=question,
            status=state.get("status", "answered"),
            answer=state.get("answer"),
            refusal_reason=state.get("refusal_reason"),
            evidence=state.get("evidence", {}),
            removed_claims=state.get("removed_claims", []),
            tool_calls=trace.tool_calls,
            trace=trace.summary(self.settings.prices),
        )
        log_event(
            "agent_run",
            status=result.status,
            refusal_reason=result.refusal_reason,
            verdict=result.answer.verdict if result.answer else None,
            claims=len(result.answer.claims) if result.answer else 0,
            removed_claims=len(result.removed_claims),
            tool_calls=len(trace.tool_calls),
            total_tokens=trace.total_tokens,
            total_ms=result.trace["total_ms"],
        )
        return result
