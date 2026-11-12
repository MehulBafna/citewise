"""Keyword queries: the question without stopwords, the way people search PubMed."""

from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "do", "does", "did",
    "for", "from", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "should",
    "than", "that", "the", "their", "there", "this", "to", "was", "were", "what", "when", "which",
    "who", "whether", "will", "with", "would", "after", "before", "during", "any", "all", "more",
    "less", "other", "same", "such", "between", "among", "versus", "vs", "really", "truly",
}  # fmt: skip


def keyword_query(question: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", question)
    kept = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    return " ".join(kept[:max_words])
