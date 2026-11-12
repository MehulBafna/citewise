"""citewise-ask: ask one question from the terminal.

citewise-ask "Does vitamin D supplementation reduce fracture risk in older adults?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable

from citewise.agent import CitewiseAgent
from citewise.config import Settings, load_env_file
from citewise.guardrails import DISCLAIMER
from citewise.llm import ClaudeLLM
from citewise.toolbox import MCPToolbox

QUIET_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "mcp", "langchain")


def _progress_printer() -> Callable[[str], None]:
    start = time.perf_counter()

    def show(message: str) -> None:
        print(f"  {time.perf_counter() - start:5.1f}s  {message}", file=sys.stderr, flush=True)

    return show


async def _ask(question: str, as_json: bool) -> None:
    settings = Settings()
    progress = _progress_printer()
    progress(f"Starting (main model {settings.model}); a question usually takes a minute or two")
    async with MCPToolbox() as toolbox:
        agent = CitewiseAgent(ClaudeLLM(settings), toolbox, settings, progress=progress)
        result = await agent.run(question)
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    if result.status == "refused":
        print(result.refusal)
        return
    answer = result.answer
    print(f"\nVerdict: {answer.verdict}\n\n{answer.summary}\n")
    for claim in answer.claims:
        print(f"- {claim.statement} [PMID {', '.join(claim.pmids)}]")
    if result.sources():
        print("\nSources:")
        for source in result.sources():
            print(f"  {source['pmid']}  {source['title']} ({source['year']})  {source['url']}")
    if result.removed_claims:
        print(f"\n{len(result.removed_claims)} claim(s) removed by the verifier.")
    trace = result.trace
    cost = f", ~${trace['cost_usd']}" if trace.get("cost_usd") is not None else ""
    print(
        f"\n{trace['total_ms'] / 1000:.1f} s, {trace['total_tokens']} tokens, "
        f"{trace['tool_calls']} tool calls{cost}\n{DISCLAIMER}"
    )


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Ask Citewise a biomedical research question.")
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--verbose", action="store_true", help="show step logs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    if not args.verbose:
        for name in QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    asyncio.run(_ask(args.question, args.json))


if __name__ == "__main__":
    main()
