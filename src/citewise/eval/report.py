"""Turn per-item records into a report: metrics with 95% intervals, paired comparisons, behaviour."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from citewise.eval.metrics import accuracy, bootstrap_ci, confusion, macro_f1, paired_difference

COMPARISONS = (
    ("find_and_read", "with_context"),
    ("find_and_read", "closed_book"),
    ("agent", "closed_book"),
    ("with_context", "closed_book"),
    ("agent", "with_context"),
)


def _ok(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("error") is None and r.get("pred") is not None]


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 3) if values else None


def committed_metrics(ok: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """How often a setting commits to yes or no, and how accurate those committed answers are.

    A system allowed to abstain should be judged on both: accuracy alone punishes every "maybe",
    and accuracy-when-committed alone rewards answering only the easy questions.
    """
    committed = [r for r in ok if r["pred"] in ("yes", "no")]
    y_true = [r["label"] for r in committed]
    y_pred = [r["pred"] for r in committed]
    low, high = bootstrap_ci(y_true, y_pred, accuracy, seed=seed)
    return {
        "committed": len(committed),
        "coverage": len(committed) / len(ok) if ok else float("nan"),
        "accuracy_when_committed": accuracy(y_true, y_pred),
        "accuracy_when_committed_ci95": [low, high],
    }


def summarize_mode(records: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    ok = _ok(records)
    y_true = [r["label"] for r in ok]
    y_pred = [r["pred"] for r in ok]
    acc_low, acc_high = bootstrap_ci(y_true, y_pred, accuracy, seed=seed)
    f1_low, f1_high = bootstrap_ci(y_true, y_pred, macro_f1, seed=seed)
    summary: dict[str, Any] = {
        "items": len(records),
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci95": [acc_low, acc_high],
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci95": [f1_low, f1_high],
        "confusion": confusion(y_true, y_pred),
        "mean_tokens": _mean([r.get("tokens", 0) for r in ok]),
        "mean_seconds": _mean([r.get("ms", 0) / 1000 for r in ok]),
        **committed_metrics(ok, seed=seed),
    }
    if ok and "tool_calls" in ok[0]:
        summary.update(
            mean_tool_calls=_mean([r.get("tool_calls", 0) for r in ok]),
            mean_claims=_mean([r.get("claims", 0) for r in ok]),
            answers_with_removed_claims=_mean([1.0 if r.get("removed_claims") else 0.0 for r in ok]),
            insufficient_evidence_rate=_mean(
                [1.0 if r.get("verdict") == "insufficient_evidence" else 0.0 for r in ok]
            ),
            leaked_items=sum(1 for r in ok if r.get("leaked")),
            source_found_rate=_mean([1.0 if r.get("found_source") else 0.0 for r in ok]),
            study_verdict_rate=_mean(
                [1.0 if r.get("study_verdict") in ("yes", "no", "maybe") else 0.0 for r in ok]
            ),
            primary_is_source_rate=_mean([1.0 if r.get("primary_is_source") else 0.0 for r in ok]),
        )
    return summary


def build_report(
    records_by_mode: dict[str, list[dict[str, Any]]], meta: dict[str, Any], *, seed: int = 0
) -> dict[str, Any]:
    modes = {mode: summarize_mode(records, seed=seed) for mode, records in records_by_mode.items()}
    paired = {}
    for a, b in COMPARISONS:
        if a not in records_by_mode or b not in records_by_mode:
            continue
        ra = {r["pmid"]: r for r in _ok(records_by_mode[a])}
        rb = {r["pmid"]: r for r in _ok(records_by_mode[b])}
        common = sorted(set(ra) & set(rb))
        y_true = [ra[p]["label"] for p in common]
        result = paired_difference(
            y_true,
            [ra[p]["pred"] for p in common],
            [rb[p]["pred"] for p in common],
            accuracy,
            seed=seed,
        )
        paired[f"{a}_minus_{b}"] = {"items": len(common), **result}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **meta,
        "modes": modes,
        "paired_accuracy": paired,
    }


def _pct(value: float | None) -> str:
    return "-" if value is None or value != value else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PubMedQA evaluation",
        "",
        f"Generated {report['generated_at']}. Split: {report.get('split')}, "
        f"{report.get('items')} questions (seed {report.get('seed')}). "
        f"Main model: `{report.get('model')}`, fast model: `{report.get('fast_model')}`. "
        f"Agent: verdict policy `{report.get('verdict_policy', 'strict')}`, "
        f"tool budget {report.get('max_tool_calls', '-')}.",
        "",
        "| Setting | Completed | Accuracy (95% CI) | Macro-F1 (95% CI) | Mean tokens | Mean s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, s in report["modes"].items():
        lines.append(
            f"| {mode} | {s['completed']}/{s['items']} | {_pct(s['accuracy'])} "
            f"({_pct(s['accuracy_ci95'][0])} to {_pct(s['accuracy_ci95'][1])}) | "
            f"{_pct(s['macro_f1'])} ({_pct(s['macro_f1_ci95'][0])} to "
            f"{_pct(s['macro_f1_ci95'][1])}) | {s['mean_tokens'] or '-'} | {s['mean_seconds'] or '-'} |"
        )
    lines += [
        "",
        "Committed answers (yes or no, as opposed to maybe):",
        "",
        "| Setting | Commits to yes/no | Accuracy when committed (95% CI) |",
        "| --- | ---: | ---: |",
    ]
    for mode, s in report["modes"].items():
        ci = s["accuracy_when_committed_ci95"]
        lines.append(
            f"| {mode} | {_pct(s['coverage'])} ({s['committed']}/{s['completed']}) | "
            f"{_pct(s['accuracy_when_committed'])} ({_pct(ci[0])} to {_pct(ci[1])}) |"
        )
    if report["paired_accuracy"]:
        lines += ["", "Paired accuracy differences on the same questions:", ""]
        for name, p in report["paired_accuracy"].items():
            if not p["items"]:
                lines.append(f"- {name.replace('_minus_', ' minus ')}: no questions completed in both.")
                continue
            verdict = (
                "the data cannot tell them apart"
                if p["ci95_low"] <= 0 <= p["ci95_high"]
                else "a clear difference"
            )
            lines.append(
                f"- {name.replace('_minus_', ' minus ')}: {p['difference'] * 100:+.1f} points "
                f"(95% CI {p['ci95_low'] * 100:+.1f} to {p['ci95_high'] * 100:+.1f}, "
                f"{p['items']} questions): {verdict}."
            )
    for mode in ("find_and_read", "agent"):
        agent = report["modes"].get(mode)
        if not agent or "mean_tool_calls" not in agent:
            continue
        lines += [
            "",
            f"Agent behaviour ({mode}):",
            "",
            f"- Mean tool calls: {agent['mean_tool_calls']}, mean cited claims: {agent['mean_claims']}",
            f"- Answers where the verifier removed a claim: {_pct(agent['answers_with_removed_claims'])}",
            f"- Answers of insufficient_evidence (scored as maybe): "
            f"{_pct(agent['insufficient_evidence_rate'])}",
        ]
        lines.append(f"- Answers with a study verdict: {_pct(agent.get('study_verdict_rate'))}")
        if mode == "find_and_read":
            lines.append(
                f"- Questions where the agent found the source paper: {_pct(agent['source_found_rate'])}"
            )
            lines.append(
                f"- Questions where it named the source paper as the study asked about: "
                f"{_pct(agent.get('primary_is_source_rate'))}"
            )
        lines.append(
            f"- Questions where forbidden text reached the evidence: {agent['leaked_items']} (must be 0)"
        )
    lines += [
        "",
        "Settings: closed_book answers from model knowledge only. with_context answers from the "
        "paper's own abstract without its conclusion (the standard PubMedQA setting). find_and_read "
        "must find that paper in PubMed itself and sees only the same abstract without conclusion; "
        "other papers must be from earlier years. agent has the source paper excluded and only "
        "earlier papers allowed, so it must forecast the result from prior literature.",
    ]
    return "\n".join(lines) + "\n"


def compare_runs(
    a: list[dict[str, Any]], b: list[dict[str, Any]], a_name: str, b_name: str, *, seed: int = 0
) -> str:
    """Two runs of one setting (e.g. two agent configurations) on the questions both completed."""
    ra = {r["pmid"]: r for r in _ok(a)}
    rb = {r["pmid"]: r for r in _ok(b)}
    common = sorted(set(ra) & set(rb))
    lines = [
        f"# Comparison on {len(common)} questions answered in both runs",
        "",
        "| Run | Accuracy (95% CI) | Commits to yes/no | Accuracy when committed (95% CI) | Mean tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, records in ((a_name, [ra[p] for p in common]), (b_name, [rb[p] for p in common])):
        s = summarize_mode(records, seed=seed)
        acc_ci, com_ci = s["accuracy_ci95"], s["accuracy_when_committed_ci95"]
        lines.append(
            f"| {name} | {_pct(s['accuracy'])} ({_pct(acc_ci[0])} to {_pct(acc_ci[1])}) | "
            f"{_pct(s['coverage'])} | {_pct(s['accuracy_when_committed'])} "
            f"({_pct(com_ci[0])} to {_pct(com_ci[1])}) | {s['mean_tokens'] or '-'} |"
        )
    diff = paired_difference(
        [ra[p]["label"] for p in common],
        [ra[p]["pred"] for p in common],
        [rb[p]["pred"] for p in common],
        accuracy,
        seed=seed,
    )
    if common:
        verdict = (
            "the data cannot tell them apart"
            if diff["ci95_low"] <= 0 <= diff["ci95_high"]
            else "a clear difference"
        )
        lines += [
            "",
            f"Accuracy, {a_name} minus {b_name}: {diff['difference'] * 100:+.1f} points "
            f"(95% CI {diff['ci95_low'] * 100:+.1f} to {diff['ci95_high'] * 100:+.1f}): {verdict}.",
        ]
    return "\n".join(lines) + "\n"
