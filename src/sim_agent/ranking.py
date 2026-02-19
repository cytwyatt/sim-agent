from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from sim_agent.config import RankingWeights
from sim_agent.types import PaperMetadata, RankedPaper


def rank_papers(
    papers: list[PaperMetadata],
    query: str,
    weights: RankingWeights,
    years_window: int = 10,
) -> list[RankedPaper]:
    ranked: list[RankedPaper] = []
    for paper in papers:
        relevance = _relevance_score(query, f"{paper.title} {paper.abstract}")
        recency = _recency_score(paper.year, years_window)
        citation = _citation_score(paper.citation_count)
        score = (
            weights.relevance_weight * relevance
            + weights.recency_weight * recency
            + weights.citation_weight * citation
        )
        ranked.append(RankedPaper(metadata=paper, score=score))

    return sorted(ranked, key=lambda p: p.score, reverse=True)


def _tokenize(text: str) -> set[str]:
    lower = text.lower()
    normalized = lower.replace("-", " ")
    return set(re.findall(r"[a-zA-Z0-9\+]+", normalized))


def _relevance_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q:
        return 0.0
    overlap = len(q & t) / len(q)
    return min(max(overlap, 0.0), 1.0)


def _recency_score(year: int | None, years_window: int) -> float:
    if not year:
        return 0.2
    current_year = datetime.now(timezone.utc).year
    age = max(current_year - year, 0)
    if age <= years_window:
        return 1.0 - (age / max(years_window, 1)) * 0.8
    return max(0.0, 0.2 - (age - years_window) * 0.02)


def _citation_score(citation_count: int) -> float:
    if citation_count <= 0:
        return 0.0
    return min(math.log1p(citation_count) / 10.0, 1.0)
