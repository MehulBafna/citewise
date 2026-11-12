"""The knowledge service, exposed as an MCP server.

Three tools: search PubMed, read abstracts, and search the evidence index semantically. Any MCP
client can use them: the Citewise agent connects over stdio, and Claude Desktop or other MCP clients
can be pointed at the same server.

`exclude_pmids` and `max_year` are policy parameters. The agent sets them from its evidence policy
and hides them from the model, so the model cannot widen what it is allowed to read.
"""

from __future__ import annotations

import argparse
import json
import threading
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from citewise.config import Settings, load_env_file
from citewise.index import EvidenceIndex, FastEmbedEmbedder, HashEmbedder
from citewise.papers import Paper, Section
from citewise.pubmed import OfflinePubMed, PaperSource, PubMedClient

POLICY_PARAMS = ("exclude_pmids", "max_year")
SNIPPET_CHARS = 280

mcp = FastMCP(
    "citewise",
    instructions=(
        "Biomedical literature tools backed by PubMed. Search first, then read abstracts or search "
        "the evidence index for passages that answer the question."
    ),
)
_lock = threading.Lock()


class Knowledge:
    def __init__(self, source: PaperSource, index: EvidenceIndex) -> None:
        self.source = source
        self.index = index

    def papers(self, pmids: list[str]) -> list[Paper]:
        """Fetch and index papers. Ones already indexed are read back from the index."""
        missing = [pmid for pmid in pmids if not self.index.has(pmid)]
        fetched = {p.pmid: p for p in self.source.fetch(missing)} if missing else {}
        self.index.add(list(fetched.values()))
        papers = []
        for pmid in pmids:
            if pmid in fetched:
                papers.append(fetched[pmid])
                continue
            passages = self.index.paper_passages(pmid)
            if passages:
                first = passages[0]
                papers.append(
                    Paper(
                        pmid=pmid,
                        title=first.title,
                        sections=tuple(Section(p.section, p.text) for p in passages),
                        journal=first.journal,
                        year=first.year,
                    )
                )
        return papers


@lru_cache(maxsize=1)
def knowledge() -> Knowledge:
    settings = Settings()
    source: PaperSource
    if settings.offline_fixture:
        source = OfflinePubMed(settings.offline_fixture)
    else:
        source = PubMedClient(api_key=settings.ncbi_api_key, email=settings.ncbi_email)
    embedder = HashEmbedder() if settings.fake_embeddings else FastEmbedEmbedder(settings.embedding_model)
    return Knowledge(source, EvidenceIndex(settings.index_path, embedder))


def _allowed(paper: Paper, exclude: set[str], max_year: int | None) -> bool:
    if paper.pmid in exclude:
        return False
    return max_year is None or (paper.year is not None and paper.year <= max_year)


@mcp.tool()
def search_pubmed(
    query: str,
    max_results: int = 8,
    exclude_pmids: list[str] | None = None,
    max_year: int | None = None,
) -> str:
    """Search PubMed. Returns matching papers (PMID, title, year, journal, abstract snippet).

    Use short keyword queries, for example "statins primary prevention mortality". Results are added
    to the evidence index, so search_evidence can find passages in them afterwards.
    """
    exclude = set(exclude_pmids or [])
    max_results = max(1, min(int(max_results), 20))
    with _lock:
        k = knowledge()
        pmids = k.source.search(query, max_results=max_results + len(exclude), max_year=max_year)
        papers = [p for p in k.papers(pmids) if _allowed(p, exclude, max_year)][:max_results]
    return json.dumps(
        [
            {
                "pmid": p.pmid,
                "title": p.title,
                "year": p.year,
                "journal": p.journal,
                "snippet": p.abstract[:SNIPPET_CHARS],
            }
            for p in papers
        ],
        ensure_ascii=False,
    )


@mcp.tool()
def get_abstracts(
    pmids: list[str],
    exclude_pmids: list[str] | None = None,
    max_year: int | None = None,
) -> str:
    """Read the full abstracts of up to 5 papers by PMID."""
    exclude = set(exclude_pmids or [])
    wanted = [str(p) for p in pmids][:5]
    with _lock:
        papers = [p for p in knowledge().papers(wanted) if _allowed(p, exclude, max_year)]
    return json.dumps(
        [
            {
                "pmid": p.pmid,
                "title": p.title,
                "year": p.year,
                "journal": p.journal,
                "abstract": p.abstract,
            }
            for p in papers
        ],
        ensure_ascii=False,
    )


@mcp.tool()
def search_evidence(
    question: str,
    k: int = 8,
    exclude_pmids: list[str] | None = None,
    max_year: int | None = None,
) -> str:
    """Semantic search over every abstract read so far. Returns the passages most relevant to the
    question, each with its PMID and section (BACKGROUND, METHODS, RESULTS, CONCLUSIONS, ...)."""
    k = max(1, min(int(k), 20))
    with _lock:
        passages = knowledge().index.search(question, k=k, exclude_pmids=exclude_pmids, max_year=max_year)
    return json.dumps([p.to_dict() for p in passages], ensure_ascii=False)


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Run the Citewise MCP server.")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP, not stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.http:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
