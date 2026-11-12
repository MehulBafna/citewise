"""Run the evaluation item by item, saving each result so an interrupted run resumes where it stopped."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from citewise.agent import CitewiseAgent
from citewise.eval.pubmedqa import Item, leakage_policy, to_label
from citewise.llm import LLM
from citewise.tracing import Trace, current_trace

MODES = ("closed_book", "with_context", "agent")


async def run_item(
    mode: str,
    item: Item,
    llm: LLM,
    agent: CitewiseAgent | None,
    *,
    same_year_ok: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {"pmid": item.pmid, "mode": mode, "label": item.label}
    start = time.perf_counter()
    trace = Trace()
    token = current_trace.set(trace)
    try:
        if mode == "closed_book":
            reply = await llm.closed_book(item.question)
            record.update(pred=reply.verdict, verdict=reply.verdict)
        elif mode == "with_context":
            reply = await llm.with_context(item.question, list(item.contexts))
            record.update(pred=reply.verdict, verdict=reply.verdict)
        elif mode == "agent":
            if agent is None:
                raise ValueError("agent mode needs an agent")
            policy = leakage_policy(item, same_year_ok=same_year_ok)
            result = await agent.run(item.question, policy)
            answer = result.answer
            verdict = answer.verdict if answer else (result.refusal_reason or "refused")
            cited = sorted({p for c in answer.claims for p in c.pmids}) if answer else []
            record.update(
                pred=to_label(verdict),
                verdict=verdict,
                status=result.status,
                claims=len(answer.claims) if answer else 0,
                removed_claims=len(result.removed_claims),
                cited_pmids=cited,
                evidence_pmids=sorted(result.evidence),
                leaked=item.pmid in cited or item.pmid in result.evidence,
                tool_calls=len(result.tool_calls),
            )
            # The agent keeps its own trace for the run; take the numbers from it.
            record["tokens"] = result.trace.get("total_tokens", 0)
            record["usage"] = result.trace.get("usage", {})
        else:
            raise ValueError(f"unknown mode {mode!r}")
        record["error"] = None
    except Exception as exc:  # one failed item must not stop the run
        record.update(pred=None, error=f"{type(exc).__name__}: {exc}")
    finally:
        current_trace.reset(token)
    if "tokens" not in record:
        record["tokens"] = trace.total_tokens
        record["usage"] = trace.usage
    record["ms"] = round((time.perf_counter() - start) * 1000)
    return record


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["pmid"]] = record
    return records


async def run_mode(
    mode: str,
    items: list[Item],
    llm: LLM,
    agent: CitewiseAgent | None,
    out_path: Path,
    *,
    concurrency: int = 2,
    same_year_ok: bool = False,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run one mode over the items. Items already answered without error are not run again."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {pmid: r for pmid, r in read_records(out_path).items() if r.get("error") is None}
    todo = [item for item in items if item.pmid not in done]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    finished = 0

    async def one(item: Item) -> None:
        nonlocal finished
        async with semaphore:
            record = await run_item(mode, item, llm, agent, same_year_ok=same_year_ok)
        async with lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            done[item.pmid] = record
            finished += 1
            if progress:
                status = "error" if record["error"] else record["pred"]
                print(f"[{mode}] {finished}/{len(todo)} {item.pmid}: {status}", flush=True)

    await asyncio.gather(*(one(item) for item in todo))
    latest = read_records(out_path)
    return [latest[item.pmid] for item in items if item.pmid in latest]
