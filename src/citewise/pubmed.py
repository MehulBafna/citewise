"""PubMed through the NCBI E-utilities API, plus an offline stand-in for tests and demos.

E-utilities asks for at most 3 requests per second without an API key and 10 with one, and for a
`tool` and `email` parameter so NCBI can get in touch about heavy use.
"""

from __future__ import annotations

import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Protocol

import httpx

from citewise.papers import Paper, Section

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
YEAR = re.compile(r"(1[89]\d\d|20\d\d)")


class PaperSource(Protocol):
    def search(self, query: str, *, max_results: int = 10, max_year: int | None = None) -> list[str]:
        """PMIDs matching the query, most relevant first."""

    def fetch(self, pmids: list[str]) -> list[Paper]:
        """Full records (title, abstract sections, journal, year) for the given PMIDs."""


def _text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _year(article: ET.Element) -> int | None:
    for path in (
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//Article/Journal/JournalIssue/PubDate/MedlineDate",
        ".//Article/ArticleDate/Year",
    ):
        match = YEAR.search(_text(article.find(path)))
        if match:
            return int(match.group(1))
    return None


def parse_efetch_xml(xml: str | bytes) -> list[Paper]:
    """Turn an efetch response (rettype=abstract, retmode=xml) into Paper records."""
    root = ET.fromstring(xml)
    papers = []
    for article in root.iter("PubmedArticle"):
        pmid = _text(article.find("./MedlineCitation/PMID"))
        if not pmid:
            continue
        sections = []
        for part in article.findall(".//Article/Abstract/AbstractText"):
            text = _text(part)
            if text:
                label = (part.get("Label") or part.get("NlmCategory") or "ABSTRACT").upper()
                sections.append(Section(label, text))
        papers.append(
            Paper(
                pmid=pmid,
                title=_text(article.find(".//Article/ArticleTitle")),
                sections=tuple(sections),
                journal=_text(article.find(".//Article/Journal/Title")),
                year=_year(article),
            )
        )
    return papers


class PubMedClient:
    """A small, polite E-utilities client with retries and rate limiting."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        http: httpx.Client | None = None,
        timeout: float = 20.0,
        retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.email = email
        self.http = http or httpx.Client(timeout=timeout)
        self.retries = retries
        self.min_interval = 0.11 if api_key else 0.34
        self._lock = threading.Lock()
        self._last = 0.0

    def _params(self, **params: str | int) -> dict[str, str | int]:
        base: dict[str, str | int] = {"tool": "citewise"}
        if self.email:
            base["email"] = self.email
        if self.api_key:
            base["api_key"] = self.api_key
        return {**base, **params}

    def _get(self, endpoint: str, params: dict[str, str | int]) -> httpx.Response:
        for attempt in range(self.retries + 1):
            with self._lock:
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            response = self.http.get(f"{EUTILS}/{endpoint}", params=params)
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                response.raise_for_status()
                return response
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def search(self, query: str, *, max_results: int = 10, max_year: int | None = None) -> list[str]:
        params = self._params(db="pubmed", term=query, retmode="json", retmax=max_results, sort="relevance")
        if max_year is not None:
            params.update(datetype="pdat", mindate="1800/01/01", maxdate=f"{max_year}/12/31")
        data = self._get("esearch.fcgi", params).json()
        return [str(pmid) for pmid in data.get("esearchresult", {}).get("idlist", [])]

    def fetch(self, pmids: list[str]) -> list[Paper]:
        if not pmids:
            return []
        params = self._params(db="pubmed", id=",".join(pmids), retmode="xml", rettype="abstract")
        papers = {p.pmid: p for p in parse_efetch_xml(self._get("efetch.fcgi", params).content)}
        return [papers[pmid] for pmid in pmids if pmid in papers]


class OfflinePubMed:
    """Searches a local JSON list of papers by word overlap. For tests and offline demos only."""

    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.papers = {p.pmid: p for p in (Paper.from_dict(item) for item in data)}

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    def search(self, query: str, *, max_results: int = 10, max_year: int | None = None) -> list[str]:
        wanted = self._words(query)
        scored = []
        for paper in self.papers.values():
            if max_year is not None and (paper.year is None or paper.year > max_year):
                continue
            overlap = len(wanted & self._words(f"{paper.title} {paper.abstract}"))
            if overlap:
                scored.append((overlap, paper.pmid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [pmid for _, pmid in scored[:max_results]]

    def fetch(self, pmids: list[str]) -> list[Paper]:
        return [self.papers[pmid] for pmid in pmids if pmid in self.papers]
