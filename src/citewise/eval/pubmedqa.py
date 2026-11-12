"""PubMedQA (Jin et al., 2019): 1,000 expert-labelled yes/no/maybe questions, each from one paper.

Each question is the title of a PubMed paper. The labelled set gives the abstract without its
conclusion, the conclusion itself, and the yes/no/maybe answer. The official test split is 500 of
the 1,000 questions; the other 500 form the dev split, where every design choice is tuned so the
test result stays honest.

Three settings are compared:

- closed_book:  Claude answers from its own knowledge, with no documents.
- with_context: Claude answers from the paper's own abstract (minus the conclusion). This is the
                standard "reasoning required" PubMedQA setting, and an upper reference point.
- find_and_read: the full agent, which must find the question's own paper in PubMed by itself. It
                sees that paper only as the abstract without the conclusion (as with_context does),
                and every other paper must be from earlier years. This is the retrieval setting.
- agent:        the full agent with the question's own paper excluded and only earlier papers
                allowed: it must forecast the study's result from prior literature. The hard case.
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
    labels: tuple[str, ...]
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
    test = set(json.loads((data_dir / "test_ground_truth.json").read_text(encoding="utf-8")))
    if split == "test":
        wanted = test
    elif split == "dev":  # the labelled questions outside the official test set: tune here
        wanted = set(raw) - test
    elif split == "all":
        wanted = set(raw)
    else:
        raise ValueError(f"unknown split {split!r}; use test, dev or all")
    items = []
    for pmid in sorted(wanted):
        entry = raw[pmid]
        year = entry.get("YEAR")
        items.append(
            Item(
                pmid=str(pmid),
                question=entry["QUESTION"],
                contexts=tuple(entry["CONTEXTS"]),
                labels=tuple(entry.get("LABELS") or ()),
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


def abstract_without_conclusion(item: Item) -> str:
    """The paper's abstract minus its conclusion, exactly as PubMedQA's reasoning-required setting."""
    if len(item.labels) == len(item.contexts):
        return "\n".join(f"{label}: {text}" for label, text in zip(item.labels, item.contexts, strict=True))
    return "\n".join(item.contexts)


def find_and_read_policy(item: Item) -> EvidencePolicy:
    """The agent may find the source paper, but only sees its abstract without the conclusion.
    Every other paper must be from earlier years, so later papers cannot give the answer away."""
    return EvidencePolicy(
        max_year=item.year - 1 if item.year else None,
        masked={item.pmid: abstract_without_conclusion(item)},
    )
