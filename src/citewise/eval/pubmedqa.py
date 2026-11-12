"""PubMedQA (Jin et al., 2019): 1,000 expert-labelled yes/no/maybe questions, each from one paper.

Each question is the title of a PubMed paper. The labelled set gives the abstract without its
conclusion, the conclusion itself, and the yes/no/maybe answer. The official test split is 500 of
the 1,000 questions.

Three settings are compared:

- closed_book:  Claude answers from its own knowledge, with no documents.
- with_context: Claude answers from the paper's own abstract (minus the conclusion). This is the
                standard "reasoning required" PubMedQA setting, and an upper reference point.
- agent:        the full agent, which must find evidence in PubMed itself. The question's own paper
                is excluded, and so is everything published in or after its year, so the agent can
                neither read the answer nor read later papers that cite it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

from citewise.guardrails import EvidencePolicy

DATA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"
FILES = ("ori_pqal.json", "test_ground_truth.json")


@dataclass(frozen=True)
class Item:
    pmid: str
    question: str
    contexts: tuple[str, ...]
    year: int | None
    label: str
    long_answer: str


def download(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        for name in FILES:
            target = data_dir / name
            if not target.exists():
                response = http.get(DATA_URL + name)
                response.raise_for_status()
                target.write_bytes(response.content)


def load_items(data_dir: str | Path, split: str = "test") -> list[Item]:
    data_dir = Path(data_dir)
    raw = json.loads((data_dir / "ori_pqal.json").read_text(encoding="utf-8"))
    if split == "test":
        wanted = set(json.loads((data_dir / "test_ground_truth.json").read_text(encoding="utf-8")))
    elif split == "all":
        wanted = set(raw)
    else:
        raise ValueError(f"unknown split {split!r}; use test or all")
    items = []
    for pmid in sorted(wanted):
        entry = raw[pmid]
        year = entry.get("YEAR")
        items.append(
            Item(
                pmid=str(pmid),
                question=entry["QUESTION"],
                contexts=tuple(entry["CONTEXTS"]),
                year=int(year) if year else None,
                label=entry["final_decision"],
                long_answer=entry.get("LONG_ANSWER", ""),
            )
        )
    return items


def sample(items: list[Item], limit: int | None, seed: int = 0) -> list[Item]:
    """A random subset. Samples are nested: --limit 100 contains every question of --limit 30
    with the same seed, so a larger run reuses the answers already saved."""
    if not limit or limit >= len(items):
        return items
    order = list(items)
    random.Random(seed).shuffle(order)
    return sorted(order[:limit], key=lambda item: item.pmid)


def to_label(verdict: str) -> str:
    """Map an agent verdict onto PubMedQA's labels. Anything not yes or no counts as maybe."""
    return verdict if verdict in ("yes", "no") else "maybe"


def leakage_policy(item: Item, *, same_year_ok: bool = False) -> EvidencePolicy:
    if item.year is None:
        max_year = None
    else:
        max_year = item.year if same_year_ok else item.year - 1
    return EvidencePolicy(exclude_pmids=frozenset({item.pmid}), max_year=max_year)
