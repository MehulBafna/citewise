"""The paper record shared by the PubMed client, the index and the MCP tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Section:
    label: str  # e.g. BACKGROUND, METHODS, RESULTS, CONCLUSIONS, or ABSTRACT if unstructured
    text: str


@dataclass(frozen=True)
class Paper:
    pmid: str
    title: str
    sections: tuple[Section, ...] = field(default_factory=tuple)
    journal: str = ""
    year: int | None = None

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def abstract(self) -> str:
        return "\n".join(f"{s.label}: {s.text}" if s.label != "ABSTRACT" else s.text for s in self.sections)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = [asdict(s) for s in self.sections]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        return cls(
            pmid=str(data["pmid"]),
            title=data.get("title", ""),
            sections=tuple(Section(s["label"], s["text"]) for s in data.get("sections", [])),
            journal=data.get("journal", ""),
            year=int(data["year"]) if data.get("year") else None,
        )
