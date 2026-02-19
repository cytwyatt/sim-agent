from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SimulationType(str, Enum):
    MD = "MD"
    QM = "QM"
    QMMM = "QMMM"
    MC = "MC"
    CG = "CG"
    OTHER = "Other/Unknown"


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    abstract: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    venue: str | None = None
    citation_count: int = 0
    open_access_pdf_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "open_access_pdf_url": self.open_access_pdf_url,
        }


@dataclass
class PaperRecord:
    paper_metadata: PaperMetadata
    simulation_type: str
    type_confidence: float
    core_simulation_details: dict[str, Any]
    domain_details: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    validation_flags: list[dict[str, Any]]
    summary: str
    source_mode: str
    raw_excerpt: str = ""
    ranking_score: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_metadata": self.paper_metadata.to_dict(),
            "simulation_type": self.simulation_type,
            "type_confidence": self.type_confidence,
            "core_simulation_details": self.core_simulation_details,
            "domain_details": self.domain_details,
            "evidence": self.evidence,
            "validation_flags": self.validation_flags,
            "summary": self.summary,
            "source_mode": self.source_mode,
            "raw_excerpt": self.raw_excerpt,
            "ranking_score": self.ranking_score,
            "errors": self.errors,
        }


@dataclass
class RankedPaper:
    metadata: PaperMetadata
    score: float
