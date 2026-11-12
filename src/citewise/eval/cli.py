"""citewise-eval: run PubMedQA in one or more settings and write a report.

    citewise-eval --download --split dev --limit 50 --modes agent --policy best_supported
    citewise-eval --split test --limit 100

Tune on the dev split (the 500 labelled questions outside the test set); run the test split once,
with the chosen configuration, for the number you report.

Results go to results/<model>/<split>/: closed_book.jsonl and with_context.jsonl for the baselines,
agent-<policy>-t<tool budget>.jsonl for each agent configuration (one line per question,
resumable), and report-<configuration>.md and .json. Changing the policy or tool budget writes to a
new file, so answers from different configurations never mix.

    citewise-eval --compare results/m/dev/agent-strict-t6.jsonl results/m/dev/agent-best_supported-t6.jsonl
    citewise-eval --check-retrieval --split test --limit 100   # free: PubMed searches only

compares two runs on the questions both answered. --gate agent:0.60 exits with code 2 if the
agent's accuracy falls below 0.60, so CI can block a change that makes answers worse.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from citewise.agent import AGENT_VERSION, CitewiseAgent
from citewise.config import Settings, load_env_file
from citewise.eval.pubmedqa import download, load_items, sample
from citewise.eval.report import build_report, compare_runs, render_markdown
from citewise.eval.runner import AGENT_MODES, MODES, read_records, run_mode
from citewise.llm import ANSWER_PROMPTS, ClaudeLLM


def parse_gates(values: list[str]) -> dict[str, float]:
    gates = {}
    for value in values:
        mode, _, threshold = value.partition(":")
        if mode not in MODES or not threshold:
            raise SystemExit(f"bad --gate {value!r}; use MODE:THRESHOLD, e.g. agent:0.6")
        gates[mode] = float(threshold)
    return gates


def agent_tag(settings: Settings, same_year_ok: bool) -> str:
    tag = f"{settings.verdict_policy}-t{settings.max_tool_calls}-{AGENT_VERSION}"
    return f"{tag}-sameyear" if same_year_ok else tag


def results_file(out_dir: Path, mode: str, tag: str) -> Path:
    return out_dir / (f"{mode}-{tag}.jsonl" if mode in AGENT_MODES else f"{mode}.jsonl")


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    overrides = {}
    if args.policy:
        overrides["verdict_policy"] = args.policy
    if args.max_tool_calls:
        overrides["max_tool_calls"] = args.max_tool_calls
    settings = dataclasses.replace(settings, **overrides)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; choose from {MODES}")
    gates = parse_gates(args.gate or [])

    if args.download:
        download(args.data_dir)
    items = sample(load_items(args.data_dir, args.split), args.limit, args.seed)
    out_dir = Path(args.out) / settings.model.replace("/", "_") / args.split
    tag = agent_tag(settings, args.same_year_ok)
    llm = ClaudeLLM(settings)
    print(
        f"{len(items)} {args.split} questions; modes {', '.join(modes)}; "
        f"agent configuration {tag}; results in {out_dir}",
        flush=True,
    )

    records = {}
    if set(AGENT_MODES) & set(modes):
        from citewise.toolbox import MCPToolbox

        async with MCPToolbox() as toolbox:
            agent = CitewiseAgent(llm, toolbox, settings)
            for mode in modes:
                records[mode] = await run_mode(
                    mode,
                    items,
                    llm,
                    agent,
                    results_file(out_dir, mode, tag),
                    concurrency=args.concurrency,
                    same_year_ok=args.same_year_ok,
                )
    else:
        for mode in modes:
            records[mode] = await run_mode(
                mode, items, llm, None, results_file(out_dir, mode, tag), concurrency=args.concurrency
            )

    report = build_report(
        records,
        {
            "split": args.split,
            "items": len(items),
            "seed": args.seed,
            "model": settings.model,
            "fast_model": settings.fast_model,
            "verdict_policy": settings.verdict_policy,
            "max_tool_calls": settings.max_tool_calls,
            "same_year_ok": args.same_year_ok,
        },
        seed=args.seed,
    )
    agent_modes = [m for m in modes if m in AGENT_MODES]
    name = f"report-{'+'.join(agent_modes)}-{tag}" if agent_modes else "report-baselines"
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"Saved {out_dir / f'{name}.md'}")

    failed = [
        f"{mode}: accuracy {report['modes'][mode]['accuracy']:.3f} < {threshold}"
        for mode, threshold in gates.items()
        if mode in report["modes"] and not report["modes"][mode]["accuracy"] >= threshold
    ]
    if failed:
        print("Quality gate failed: " + "; ".join(failed), file=sys.stderr)
        return 2
    return 0


def _check_retrieval(args: argparse.Namespace) -> int:
    from citewise.eval.retrieval_check import check_retrieval, render_retrieval
    from citewise.pubmed import OfflinePubMed, PubMedClient

    settings = Settings()
    if args.download:
        download(args.data_dir)
    items = sample(load_items(args.data_dir, args.split), args.limit, args.seed)
    print(f"Searching PubMed for {len(items)} {args.split} questions (two searches each)...", flush=True)
    source = (
        OfflinePubMed(settings.offline_fixture)
        if settings.offline_fixture
        else PubMedClient(api_key=settings.ncbi_api_key, email=settings.ncbi_email)
    )
    result = check_retrieval(items, source)
    text = render_retrieval(result)
    out = Path(args.out) / "retrieval-check" / f"{args.split}-{len(items)}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {out}")
    return 0


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Evaluate Citewise on PubMedQA.")
    parser.add_argument("--modes", default="closed_book,with_context,agent")
    parser.add_argument("--split", default="test", choices=["test", "dev", "all"])
    parser.add_argument("--limit", type=int, default=None, help="random sample of N questions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=sorted(ANSWER_PROMPTS), help="agent verdict policy")
    parser.add_argument("--max-tool-calls", type=int, help="agent tool budget per question")
    parser.add_argument("--data-dir", default="data/pubmedqa")
    parser.add_argument("--out", default="results")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--download", action="store_true", help="fetch PubMedQA if missing")
    parser.add_argument(
        "--same-year-ok",
        action="store_true",
        help="allow papers from the source paper's own year (default: only earlier years)",
    )
    parser.add_argument("--gate", action="append", help="MODE:MIN_ACCURACY, may repeat")
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="compare two result files and exit"
    )
    parser.add_argument(
        "--check-retrieval",
        action="store_true",
        help="free check: how often PubMed search finds each question's source paper (no Claude calls)",
    )
    args = parser.parse_args()
    if args.check_retrieval:
        sys.exit(_check_retrieval(args))
    if args.compare:
        a, b = (Path(p) for p in args.compare)
        print(compare_runs(list(read_records(a).values()), list(read_records(b).values()), a.stem, b.stem))
        sys.exit(0)
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
