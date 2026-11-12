"""Guardrails: checks on what comes in, on what the tools may read, and on what goes out.

Input   - length limits and prompt-injection patterns, before any model is called; then a model
          triage refuses personal medical advice and off-topic requests.
Tools   - an evidence policy (excluded PMIDs, latest publication year) applied to every tool call
          and again to every tool result. Tool output is wrapped as untrusted documents.
Output  - every claim must cite at least one PMID, every cited PMID must be in the evidence the
          agent actually retrieved, and no excluded paper may be cited. A model check then asks
          whether the cited passages support each claim. Claims that fail are removed, never shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from citewise.schemas import Answer

MIN_CHARS = 8
MAX_CHARS = 1000

DISCLAIMER = "Research summary from published abstracts. Not medical advice."
REFUSALS = {
    "too_short": "Please ask a complete research question.",
    "too_long": f"Please keep the question under {MAX_CHARS} characters.",
    "prompt_injection": "This request looks like an attempt to change the assistant's instructions.",
    "personal_medical_advice": (
        "I can summarise published research, but I can't advise on an individual's diagnosis, "
        "treatment or dosing. Please ask a doctor or pharmacist."
    ),
    "off_topic": "I only answer questions about biomedical research literature.",
}

INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions|prompt|rules)\b",
    r"\bdisregard\b.{0,40}\b(instructions|prompt|rules)\b",
    r"\b(system|developer)\s+prompt\b",
    r"\byou are now\b",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"</?\s*(system|assistant|document)\s*>",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class InputCheck:
    allowed: bool
    reason: str | None = None


def check_input(question: str) -> InputCheck:
    text = question.strip()
    if len(text) < MIN_CHARS:
        return InputCheck(False, "too_short")
    if len(text) > MAX_CHARS:
        return InputCheck(False, "too_long")
    if _INJECTION.search(text):
        return InputCheck(False, "prompt_injection")
    return InputCheck(True)


# Tool parameters set from the evidence policy. The agent fills them in and hides them from the model.
POLICY_PARAMS = ("exclude_pmids", "max_year")


@dataclass(frozen=True)
class EvidencePolicy:
    """What the agent may read. The evaluation uses it to keep the answer's own paper out of reach."""

    exclude_pmids: frozenset[str] = field(default_factory=frozenset)
    max_year: int | None = None

    def tool_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.exclude_pmids:
            args["exclude_pmids"] = sorted(self.exclude_pmids)
        if self.max_year is not None:
            args["max_year"] = self.max_year
        return args

    def allows(self, pmid: str, year: int | None) -> bool:
        if pmid in self.exclude_pmids:
            return False
        return self.max_year is None or (year is not None and year <= self.max_year)


def wrap_document(pmid: str, title: str, year: int | None, body: str) -> str:
    """Present retrieved text as data. Tags inside the text are neutralised so it cannot break out."""
    safe = re.sub(r"<\s*/?\s*document", "[document", body, flags=re.IGNORECASE)
    safe_title = re.sub(r"[<>]", "", title)
    return f'<document pmid="{pmid}" year="{year or "unknown"}" title="{safe_title}">\n{safe}\n</document>'


@dataclass(frozen=True)
class ClaimProblem:
    index: int
    problem: str


def validate_answer(answer: Answer, evidence_pmids: set[str], policy: EvidencePolicy) -> list[ClaimProblem]:
    """Deterministic citation checks. Empty if every claim is properly cited."""
    problems = []
    for index, claim in enumerate(answer.claims):
        if not claim.pmids:
            problems.append(ClaimProblem(index, "cites no PMID"))
            continue
        for pmid in claim.pmids:
            if pmid in policy.exclude_pmids:
                problems.append(ClaimProblem(index, f"cites excluded PMID {pmid}"))
            elif pmid not in evidence_pmids:
                problems.append(ClaimProblem(index, f"cites PMID {pmid}, which is not in the evidence"))
    return problems
