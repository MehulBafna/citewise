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


def test_dev_split_is_everything_labelled_outside_the_test_set(data_dir: Path) -> None:
    assert [i.pmid for i in load_items(data_dir, "dev")] == ["1"]
    test = {i.pmid for i in load_items(data_dir, "test")}
    assert not test & {i.pmid for i in load_items(data_dir, "dev")}


def test_committed_answers_are_measured_separately() -> None:
    from citewise.eval.report import summarize_mode

    records = [
        {"pmid": "1", "label": "yes", "pred": "yes", "error": None},
        {"pmid": "2", "label": "no", "pred": "no", "error": None},
        {"pmid": "3", "label": "yes", "pred": "maybe", "error": None},
        {"pmid": "4", "label": "no", "pred": "yes", "error": None},
    ]
    s = summarize_mode(records)
    assert s["accuracy"] == 0.5
    assert s["committed"] == 3 and s["coverage"] == 0.75
    assert s["accuracy_when_committed"] == pytest.approx(2 / 3)


def test_two_runs_are_compared_on_their_common_questions() -> None:
    from citewise.eval.report import compare_runs

    a = [{"pmid": p, "label": "yes", "pred": "yes", "error": None} for p in "123"]
    b = [{"pmid": p, "label": "yes", "pred": "maybe", "error": None} for p in "1234"]
    text = compare_runs(a, b, "strict", "best")
    assert "Comparison on 3 questions" in text
    assert "| strict | 100.0%" in text and "| best | 0.0%" in text
    assert "strict minus best: +100.0 points" in text


def test_each_agent_configuration_gets_its_own_results_file(tmp_path: Path) -> None:
    from citewise.eval.cli import agent_tag, results_file

    s = settings(verdict_policy="best_supported", max_tool_calls=4)
    assert agent_tag(s, same_year_ok=False) == "best_supported-t4-a2"
    assert agent_tag(s, same_year_ok=True) == "best_supported-t4-a2-sameyear"
    assert results_file(tmp_path, "agent", "best_supported-t4").name == "agent-best_supported-t4.jsonl"
    assert results_file(tmp_path, "closed_book", "best_supported-t4").name == "closed_book.jsonl"


def test_find_and_read_policy_and_leak_check(data_dir: Path) -> None:
    from citewise.eval.pubmedqa import find_and_read_policy
    from citewise.eval.runner import leaked

    item = load_items(data_dir)[0]  # 90000001, 2014, contexts with BACKGROUND/RESULTS labels
    policy = find_and_read_policy(item)
    assert policy.max_year == 2013 and set(policy.masked) == {"90000001"}
    assert policy.masked["90000001"] == "BACKGROUND: Context one.\nRESULTS: Context two."

    def doc(year: int, text: str) -> dict:
        return {"title": "t", "year": year, "journal": "j", "passages": {"ABSTRACT": text}}

    clean = {"90000001": doc(2014, "masked text"), "90000002": doc(2012, "older paper")}
    assert not leaked(item, clean, policy)
    assert leaked(item, {**clean, "90000005": doc(2015, "a newer paper")}, policy)
    assert leaked(item, {"90000001": doc(2014, "it says Conclusion. right here")}, policy)


async def test_find_and_read_runs_and_reports_the_source_found_rate(data_dir: Path, tmp_path: Path) -> None:
    items = load_items(data_dir)
    llm = FakeLLM(answers=[good_answer("90000002", verdict="no")])
    agent = CitewiseAgent(llm, InProcessToolbox(), settings())
    records = await run_mode("find_and_read", items, llm, agent, tmp_path / "f.jsonl", progress=False)
    assert all(r["error"] is None and not r["leaked"] for r in records)
    assert {r["pmid"]: r["found_source"] for r in records}["90000001"] is True
    report = build_report(
        {"find_and_read": records},
        {"split": "test", "items": 3, "seed": 0, "model": "fake", "fast_model": "fake"},
    )
    assert 0 < report["modes"]["find_and_read"]["source_found_rate"] <= 1
    assert "found the source paper" in render_markdown(report)


def test_retrieval_check_counts_where_the_source_paper_ranks(data_dir: Path) -> None:
    from citewise.eval.retrieval_check import check_retrieval, keyword_query, render_retrieval
    from citewise.pubmed import OfflinePubMed
    from tests.conftest import PAPERS

    assert keyword_query("Does vitamin D reduce hip fracture risk in adults over 65?") == (
        "vitamin reduce hip fracture risk adults over 65"
    )
    result = check_retrieval(load_items(data_dir), OfflinePubMed(PAPERS))
    assert result["questions"] == 3
    ranks = {row["pmid"]: row for row in result["rows"]}
    assert ranks["90000001"]["keyword_rank"] is not None
    assert result["summary"]["either"]["top_10"] > 0
    assert "Top 10" in render_retrieval(result)


async def test_scoring_uses_the_study_verdict_when_there_is_one(data_dir: Path, tmp_path: Path) -> None:
    items = [i for i in load_items(data_dir) if i.pmid == "90000001"]  # label: no
    # the study says no (correct); the wider evidence says maybe
    answer = good_answer("90000001", verdict="maybe", primary="90000001", study="no")
    llm = FakeLLM(answers=[answer])
    agent = CitewiseAgent(llm, InProcessToolbox(), settings())
    [record] = await run_mode("find_and_read", items, llm, agent, tmp_path / "s.jsonl", progress=False)
    assert record["study_verdict"] == "no" and record["verdict"] == "maybe"
    assert record["pred"] == "no" and record["primary_is_source"] is True
