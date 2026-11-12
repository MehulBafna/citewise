"""The evidence index: abstracts split into passages, embedded and stored in a local Qdrant collection.

Qdrant runs embedded in the MCP server process (no separate database), either on disk or in memory.
Every abstract fetched from PubMed is added, so the index grows into a cache of the literature the
agent has already read. Searches can exclude given PMIDs and anything published after a given year;
the evaluation uses that to keep a question's own paper, and later papers, out of reach.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient, models

from citewise.papers import Paper

COLLECTION = "abstracts"
MAX_PASSAGE_CHARS = 1200
NAMESPACE = uuid.UUID("6f1d3a52-9a0e-4c55-8f7a-2b1c4d5e6f70")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Local ONNX sentence embeddings via fastembed. The model downloads once, on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self.model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]


class HashEmbedder:
    """A deterministic bag-of-words embedding. Fast and dependency-free, but weak: tests only."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) > 2:
                    bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                    vector[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@dataclass(frozen=True)
class Passage:
    pmid: str
    title: str
    year: int | None
    journal: str
    section: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "section": self.section,
            "text": self.text,
            "score": round(self.score, 4),
        }


def split_passages(paper: Paper) -> list[tuple[str, str]]:
    """(section label, text) pairs. Structured abstracts split by section; long ones by sentence."""
    passages = []
    for section in paper.sections:
        if len(section.text) <= MAX_PASSAGE_CHARS:
            passages.append((section.label, section.text))
            continue
        chunk = ""
        for sentence in re.split(r"(?<=[.!?])\s+", section.text):
            if chunk and len(chunk) + len(sentence) > MAX_PASSAGE_CHARS:
                passages.append((section.label, chunk.strip()))
                chunk = ""
            chunk += sentence + " "
        if chunk.strip():
            passages.append((section.label, chunk.strip()))
    if not passages and paper.title:
        passages.append(("TITLE", paper.title))
    return passages


class EvidenceIndex:
    def __init__(self, path: str, embedder: Embedder) -> None:
        self.embedder = embedder
        self.client = QdrantClient(location=":memory:") if path == ":memory:" else QdrantClient(path=path)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
            )
        self._known: set[str] = set()

    def has(self, pmid: str) -> bool:
        if pmid in self._known:
            return True
        point = self.client.retrieve(COLLECTION, ids=[str(uuid.uuid5(NAMESPACE, f"{pmid}#0"))])
        if point:
            self._known.add(pmid)
        return bool(point)

    def add(self, papers: list[Paper]) -> int:
        """Index papers not seen before. Returns the number of passages written."""
        points = []
        texts = []
        for paper in papers:
            if self.has(paper.pmid):
                continue
            for position, (label, text) in enumerate(split_passages(paper)):
                texts.append(f"{paper.title}\n{label}: {text}")
                points.append(
                    (
                        str(uuid.uuid5(NAMESPACE, f"{paper.pmid}#{position}")),
                        {
                            "pmid": paper.pmid,
                            "title": paper.title,
                            "year": paper.year,
                            "journal": paper.journal,
                            "section": label,
                            "text": text,
                            "position": position,
                        },
                    )
                )
            self._known.add(paper.pmid)
        if not points:
            return 0
        vectors = self.embedder.embed(texts)
        self.client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)
                for (point_id, payload), vector in zip(points, vectors, strict=True)
            ],
        )
        return len(points)

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        exclude_pmids: list[str] | None = None,
        max_year: int | None = None,
    ) -> list[Passage]:
        conditions_must = []
        conditions_not = []
        if exclude_pmids:
            conditions_not.append(
                models.FieldCondition(key="pmid", match=models.MatchAny(any=list(exclude_pmids)))
            )
        if max_year is not None:
            conditions_must.append(models.FieldCondition(key="year", range=models.Range(lte=max_year)))
        query_filter = (
            models.Filter(must=conditions_must or None, must_not=conditions_not or None)
            if conditions_must or conditions_not
            else None
        )
        [vector] = self.embedder.embed([query])
        hits = self.client.query_points(
            COLLECTION, query=vector, limit=k, query_filter=query_filter, with_payload=True
        ).points
        return [
            Passage(
                pmid=hit.payload["pmid"],
                title=hit.payload["title"],
                year=hit.payload.get("year"),
                journal=hit.payload.get("journal", ""),
                section=hit.payload["section"],
                text=hit.payload["text"],
                score=float(hit.score),
            )
            for hit in hits
        ]

    def paper_passages(self, pmid: str) -> list[Passage]:
        records, _ = self.client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="pmid", match=models.MatchValue(value=pmid))]
            ),
            limit=50,
            with_payload=True,
        )
        return [
            Passage(
                pmid=r.payload["pmid"],
                title=r.payload["title"],
                year=r.payload.get("year"),
                journal=r.payload.get("journal", ""),
                section=r.payload["section"],
                text=r.payload["text"],
            )
            for r in sorted(records, key=lambda r: r.payload.get("position", 0))
        ]
