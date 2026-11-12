"""Structured outputs the models must return. Pydantic validates every response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["yes", "no", "maybe", "not_applicable", "insufficient_evidence"]
StudyVerdict = Literal["yes", "no", "maybe", "no_matching_study"]
STUDY_ANSWERS = ("yes", "no", "maybe")


class Triage(BaseModel):
    """How to handle an incoming question."""

    category: Literal["research_question", "personal_medical_advice", "off_topic"] = Field(
        description=(
            "research_question: answerable from published biomedical research. "
            "personal_medical_advice: asks what a specific person should do about their own health "
            "(diagnosis, treatment, dosing). off_topic: anything else."
        )
    )
    reason: str = Field(description="One short sentence explaining the category.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="For research questions: 1 to 3 short PubMed keyword queries.",
    )


class Claim(BaseModel):
    statement: str = Field(description="One factual statement drawn from the evidence.")
    pmids: list[str] = Field(description="PMIDs of the evidence documents that support it.")


class Answer(BaseModel):
    """An answer grounded in the evidence documents, with two verdicts: what the study the question
    describes found, and what the evidence as a whole suggests."""

    primary_pmid: str | None = Field(
        description=(
            "PMID of the document that is the study the question describes (same intervention or "
            "exposure, outcome and population). null if no document is that study."
        )
    )
    study_verdict: StudyVerdict = Field(
        description=(
            "What that study found, from its own results: yes, no, or maybe if its results are "
            "mixed or conditional. no_matching_study if primary_pmid is null."
        )
    )
    verdict: Verdict = Field(
        description=(
            "What the evidence as a whole suggests. For a yes/no question: yes, no, or maybe "
            "(evidence mixed or conditional). "
            "not_applicable for other questions. insufficient_evidence if the documents do not "
            "answer the question."
        )
    )
    summary: str = Field(description="One or two sentences that only restate the cited claims.")
    claims: list[Claim] = Field(description="The supporting claims, each with its citations.")


class QuickAnswer(BaseModel):
    """A yes/no/maybe answer without citations, used for the evaluation baselines."""

    reasoning: str = Field(description="Brief reasoning, at most four sentences.")
    verdict: Literal["yes", "no", "maybe"]


class ClaimCheck(BaseModel):
    index: int = Field(description="The claim's number as given.")
    supported: bool = Field(description="True only if the cited passages state or imply it.")
    reason: str = Field(description="One short sentence.")


class SupportReport(BaseModel):
    checks: list[ClaimCheck]
