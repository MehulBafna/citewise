"""The PubMedQA harness: data loading, leakage policy, metrics, resumable runs and the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citewise.agent import CitewiseAgent
from citewise.eval.cli import parse_gates
from citewise.eval.metrics import accuracy, bootstrap_ci, confusion, macro_f1, paired_difference
from citewise.eval.pubmedqa import leakage_policy, load_items, sample, to_label
from citewise.eval.report import build_report, render_markdown
from citewise.eval.runner import run_mode
from tests.conftest import FakeLLM, InProcessToolbox, good_answer, settings


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    def entry(question: str, year: str | None, label: str) -> dict:
        return {
            "QUESTION": question,
            "CONTEXTS": ["Context one.", "Context two."],
            "LABELS": ["BACKGROUND", "RESULTS"],
            "MESHES": [],
            "YEAR": year,
            "final_decision": label,
            "LONG_ANSWER": "Conclusion.",
        }

    raw = {
        "90000001": entry("Does vitamin D reduce hip fracture risk in adults over 65?", "2014", "no"),
        "90000005": entry("Does exercise prevent falls in older adults?", "2015", "yes"),
        "90000007": entry("Do statins reduce mortality in primary prevention?", None, "maybe"),
        "1": entry("Is this question outside the test split?", "2001", "yes"),
    }
    (tmp_path / "ori_pqal.json").write_text(json.dumps(raw))
    (tmp_path / "test_ground_truth.json").write_text(
        json.dumps({"90000001": "no", "90000005": "yes", "90000007": "maybe"})
    )
    return tmp_path


def test_loading_the_test_split(data_dir: Path) -> None:
    items = load_items(data_dir, "test")
    assert [i.pmid for i in items] == ["90000001", "90000005", "90000007"]
    assert items[0].year == 2014 and items[2].year is None
    assert len(load_items(data_dir, "all")) == 4
    assert sample(items, 2, seed=1) == sample(items, 2, seed=1)
    assert len(sample(items, 2)) == 2 and sample(items, None) == items
    assert set(sample(items, 1, seed=3)) <= set(sample(items, 2, seed=3))  # nested samples


def test_leakage_policy_excludes_the_source_paper_and_its_year(data_dir: Path) -> None:
    item = load_items(data_dir)[0]
    assert leakage_policy(item).exclude_pmids == frozenset({"90000001"})
    assert leakage_policy(item).max_year == 2013
    assert leakage_policy(item, same_year_ok=True).max_year == 2014
    assert leakage_policy(load_items(data_dir)[2]).max_year is None


def test_verdict_mapping() -> None:
    assert [to_label(v) for v in ("yes", "no", "maybe", "insufficient_evidence", "not_applicable")] == [
        "yes",
        "no",
        "maybe",
        "maybe",
        "maybe",
    ]


def test_metrics_on_known_values() -> None:
    y_true = ["yes", "yes", "no", "maybe"]
    y_pred = ["yes", "no", "no", "yes"]
    assert accuracy(y_true, y_pred) == 0.5
    # yes: p=1/2 r=1/2 f1=.5 ; no: p=1/2 r=1 f1=2/3 ; maybe: 0
    assert macro_f1(y_true, y_pred) == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert confusion(y_true, y_pred)["yes"] == {"yes": 1, "no": 1, "maybe": 0}
    low, high = bootstrap_ci(y_true * 25, y_pred * 25, accuracy, resamples=500)
    assert low < 0.5 < high
    better = paired_difference(y_true * 25, y_true * 25, y_pred * 25, accuracy, resamples=500)
    assert better["difference"] == 0.5 and better["ci95_low"] > 0


async def test_runs_resume_and_the_agent_never_sees_the_source_paper(data_dir: Path, tmp_path: Path) -> None:
    items = load_items(data_dir)
    llm = FakeLLM(answers=[good_answer("90000003", verdict="no")], verdict="yes")
    agent = CitewiseAgent(llm, InProcessToolbox(), settings())
    out = tmp_path / "results"

    closed = await run_mode("closed_book", items, llm, None, out / "closed_book.jsonl", progress=False)
    agent_records = await run_mode("agent", items, llm, agent, out / "agent.jsonl", progress=False)
    assert len(closed) == 3 and all(r["error"] is None for r in closed)
    assert all(r["tokens"] > 0 for r in closed + agent_records)
    assert not any(r["leaked"] for r in agent_records)
    first = next(r for r in agent_records if r["pmid"] == "90000001")
    assert "90000001" not in first["evidence_pmids"]

    calls_before = len(llm.calls["closed_book"])
    again = await run_mode("closed_book", items, llm, None, out / "closed_book.jsonl", progress=False)
    assert len(llm.calls["closed_book"]) == calls_before  # nothing re-run
    assert again == closed

    report = build_report(
        {"closed_book": closed, "agent": agent_records},
        {"split": "test", "items": 3, "seed": 0, "model": "fake", "fast_model": "fake"},
    )
    assert report["modes"]["closed_book"]["accuracy"] == pytest.approx(1 / 3)
    assert report["modes"]["agent"]["leaked_items"] == 0
    assert "agent_minus_closed_book" in report["paired_accuracy"]
    text = render_markdown(report)
    assert "| closed_book | 3/3 |" in text and "must be 0" in text


async def test_a_failing_item_is_recorded_and_retried_next_time(data_dir: Path, tmp_path: Path) -> None:
    class Flaky(FakeLLM):
        fail = True

        async def closed_book(self, question: str):
            if self.fail:
                raise TimeoutError("slow API")
            return await super().closed_book(question)

    llm = Flaky()
    path = tmp_path / "cb.jsonl"
    records = await run_mode("closed_book", load_items(data_dir), llm, None, path, progress=False)
    assert all(r["error"] == "TimeoutError: slow API" for r in records)
    llm.fail = False
    records = await run_mode("closed_book", load_items(data_dir), llm, None, path, progress=False)
    assert all(r["error"] is None for r in records)


def test_quality_gates_parse() -> None:
    assert parse_gates(["agent:0.6", "closed_book:0.5"]) == {"agent": 0.6, "closed_book": 0.5}
    with pytest.raises(SystemExit):
        parse_gates(["nonsense"])
