"""A free check before a paid run: can a plain PubMed search find each question's source paper?

No model is called. Each question is searched twice, as written and as a short keyword query (the
kind of query the agent writes), and the report says how often the source paper appears in the
top 1, 5 and 10 results. If it is found for most questions, the find_and_read setting is worth
running; if not, the agent would mostly be answering without the paper.
"""

from __future__ import annotations

from typing import Any

from citewise.eval.pubmedqa import Item
from citewise.pubmed import PaperSource
from citewise.queries import keyword_query  # noqa: F401  (re-exported)

CUTOFFS = (1, 5, 10)


def rank_of(pmid: str, results: list[str]) -> int | None:
    return results.index(pmid) + 1 if pmid in results else None


def check_retrieval(items: list[Item], source: PaperSource, *, depth: int = 10) -> dict[str, Any]:
    rows = []
    errors = 0

    def search(query: str) -> list[str]:
        nonlocal errors
        if not query:
            return []
        try:
            return source.search(query, max_results=depth)
        except Exception:  # a failed search counts as not found, and is reported
            errors += 1
            return []

    for item in items:
        keywords = keyword_query(item.question)
        rows.append(
            {
                "pmid": item.pmid,
                "question_rank": rank_of(item.pmid, search(item.question)),
                "keyword_rank": rank_of(item.pmid, search(keywords)),
                "keyword_query": keywords,
            }
        )

    def rate(key: str, cutoff: int) -> float:
        hits = 0
        for row in rows:
            ranks = [row["question_rank"], row["keyword_rank"]] if key == "either" else [row[key]]
            if any(r is not None and r <= cutoff for r in ranks):
                hits += 1
        return hits / len(rows) if rows else float("nan")

    summary = {
        key: {f"top_{c}": rate(key, c) for c in CUTOFFS}
        for key in ("question_rank", "keyword_rank", "either")
    }
    return {"questions": len(rows), "failed_searches": errors, "summary": summary, "rows": rows}


def render_retrieval(result: dict[str, Any]) -> str:
    names = {
        "question_rank": "Question as written",
        "keyword_rank": "Keyword query",
        "either": "Either query",
    }
    lines = [
        f"# Can PubMed search find the source paper? ({result['questions']} questions, no model calls)",
        "",
        "| Query | Top 1 | Top 5 | Top 10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in names.items():
        s = result["summary"][key]
        lines.append(
            f"| {label} | {s['top_1'] * 100:.0f}% | {s['top_5'] * 100:.0f}% | {s['top_10'] * 100:.0f}% |"
        )
    either = result["summary"]["either"]["top_10"]
    advice = (
        "Most source papers are findable: the find_and_read setting is worth running."
        if either >= 0.7
        else "Many source papers are not found by a plain search: expect find_and_read to suffer."
    )
    lines += ["", advice]
    if result.get("failed_searches"):
        lines.append(f"{result['failed_searches']} searches failed (network or PubMed errors): run it again.")
    return "\n".join(lines) + "\n"
