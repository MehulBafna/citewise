"""The PubMed client: XML parsing, request parameters, retries, and the offline stand-in."""

from __future__ import annotations

import httpx
import pytest

from citewise import pubmed
from citewise.pubmed import OfflinePubMed, PubMedClient, parse_efetch_xml
from tests.conftest import FIXTURES, PAPERS


def test_parses_structured_unstructured_and_medline_dates() -> None:
    papers = parse_efetch_xml((FIXTURES / "efetch_sample.xml").read_bytes())
    assert [p.pmid for p in papers] == ["11111111", "22222222", "33333333"]
    structured, unstructured, medline = papers
    assert [s.label for s in structured.sections] == [
        "BACKGROUND",
        "METHODS",
        "RESULTS",
        "CONCLUSIONS",
    ]
    assert structured.title == "A structured abstract with italic text inside."
    assert "italic" in structured.title and structured.year == 2015
    assert structured.journal == "Fixture Journal"
    assert [s.label for s in unstructured.sections] == ["ABSTRACT"]
    assert medline.year == 1998  # from <MedlineDate>1998 Mar-Apr</MedlineDate>
    assert medline.sections == ()  # no abstract


def _client(handler) -> PubMedClient:
    return PubMedClient(http=httpx.Client(transport=httpx.MockTransport(handler)), email="me@x.org")


def test_search_sends_polite_parameters_and_a_date_limit() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={"esearchresult": {"idlist": ["3", "1", "2"]}})

    assert _client(handler).search("vitamin D", max_results=3, max_year=2015) == ["3", "1", "2"]
    assert seen["term"] == "vitamin D"
    assert seen["tool"] == "citewise" and seen["email"] == "me@x.org"
    assert seen["datetype"] == "pdat" and seen["maxdate"] == "2015/12/31"


def test_fetch_keeps_the_requested_order() -> None:
    xml = (FIXTURES / "efetch_sample.xml").read_bytes()
    client = _client(lambda request: httpx.Response(200, content=xml))
    assert [p.pmid for p in client.fetch(["33333333", "11111111"])] == ["33333333", "11111111"]
    assert client.fetch([]) == []


def test_retries_rate_limits_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubmed.time, "sleep", lambda seconds: None)
    answers = iter(
        [
            httpx.Response(429),
            httpx.Response(503),
            httpx.Response(200, json={"esearchresult": {"idlist": ["9"]}}),
        ]
    )
    assert _client(lambda request: next(answers)).search("x") == ["9"]


def test_gives_up_after_the_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubmed.time, "sleep", lambda seconds: None)
    with pytest.raises(httpx.HTTPStatusError):
        _client(lambda request: httpx.Response(500)).search("x")


def test_offline_source_respects_the_year_limit() -> None:
    source = OfflinePubMed(PAPERS)
    assert "90000004" in source.search("vitamin D hip fracture")
    assert "90000004" not in source.search("vitamin D hip fracture", max_year=2016)
    assert [p.pmid for p in source.fetch(["90000002", "missing"])] == ["90000002"]
