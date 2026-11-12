"""citewise-eval: run PubMedQA in one or more settings and write a report.

    citewise-eval --download --limit 100 --modes closed_book,with_context,agent

Results go to results/<model>/<mode>.jsonl (one line per question, resumable) and
results/<model>/report.md and report.json. --gate agent:0.60 exits with code 2 if the agent's
accuracy falls below 0.60, so CI can block a change that makes answers worse.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from citewise.agent import CitewiseAgent
from citewise.config import Settings, load_env_file
from citewise.eval.pubmedqa import download, load_items, sample
from citewise.eval.report import build_report, render_markdown
from citewise.eval.runner import MODES, run_mode
from citewise.llm import ClaudeLLM


def parse_gates(values: list[str]) -> dict[str, float]:
    gates = {}
    for value in values:
        mode, _, threshold = value.partition(":")
        if mode not in MODES or not threshold:
            raise SystemExit(f"bad --gate {value!r}; use MODE:THRESHOLD, e.g. agent:0.6")
        gates[mode] = float(threshold)
    return gates


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; choose from {MODES}")
    gates = parse_gates(args.gate or [])

    if args.download:
        download(args.data_dir)
    items = sample(load_items(args.data_dir, args.split), args.limit, args.seed)
    out_dir = Path(args.out) / settings.model.replace("/", "_")
    llm = ClaudeLLM(settings)

    records = {}
    if "agent" in modes:
        from citewise.toolbox import MCPToolbox

        async with MCPToolbox() as toolbox:
            agent = CitewiseAgent(llm, toolbox, settings)
            for mode in modes:
                records[mode] = await run_mode(
                    mode,
                    items,
                    llm,
                    agent,
                    out_dir / f"{mode}.jsonl",
                    concurrency=args.concurrency,
                    same_year_ok=args.same_year_ok,
                )
    else:
        for mode in modes:
            records[mode] = await run_mode(
                mode, items, llm, None, out_dir / f"{mode}.jsonl", concurrency=args.concurrency
            )

    report = build_report(
        records,
        {
            "split": args.split,
            "items": len(items),
            "seed": args.seed,
            "model": settings.model,
            "fast_model": settings.fast_model,
            "same_year_ok": args.same_year_ok,
        },
        seed=args.seed,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))

    failed = [
        f"{mode}: accuracy {report['modes'][mode]['accuracy']:.3f} < {threshold}"
        for mode, threshold in gates.items()
        if mode in report["modes"] and not report["modes"][mode]["accuracy"] >= threshold
    ]
    if failed:
        print("Quality gate failed: " + "; ".join(failed), file=sys.stderr)
        return 2
    return 0


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Evaluate Citewise on PubMedQA.")
    parser.add_argument("--modes", default="closed_book,with_context,agent")
    parser.add_argument("--split", default="test", choices=["test", "all"])
    parser.add_argument("--limit", type=int, default=None, help="random sample of N questions")
    parser.add_argument("--seed", type=int, default=0)
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
    sys.exit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
