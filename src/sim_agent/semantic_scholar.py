from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sim_agent.types import PaperMetadata

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
DEFAULT_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "year",
        "externalIds",
        "url",
        "venue",
        "citationCount",
        "openAccessPdf",
    ]
)


@dataclass
class SemanticScholarClient:
    api_key: str | None = None
    timeout_seconds: int = 30

    def search_papers(self, query: str, limit: int = 30, year_from: int | None = None) -> list[PaperMetadata]:
        params = {
            "query": query,
            "limit": min(max(limit, 1), 100),
            "offset": 0,
            "fields": DEFAULT_FIELDS,
        }
        req = Request(f"{SEARCH_URL}?{urlencode(params)}", method="GET")
        if self.api_key:
            req.add_header("x-api-key", self.api_key)

        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {400, 401, 403, 404, 429, 500, 502, 503}:
                return self._search_openalex(query=query, limit=limit, year_from=year_from)
            raise RuntimeError(f"Semantic Scholar request failed: {exc}") from exc
        except URLError:
            return self._search_openalex(query=query, limit=limit, year_from=year_from)

        papers: list[PaperMetadata] = []
        for item in payload.get("data", []):
            paper = self._to_metadata(item)
            if year_from and paper.year and paper.year < year_from:
                continue
            papers.append(paper)

        deduped: dict[str, PaperMetadata] = {}
        for paper in papers:
            key = paper.doi or paper.paper_id
            if key not in deduped:
                deduped[key] = paper
        return list(deduped.values())

    def _search_openalex(self, query: str, limit: int, year_from: int | None) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        limit = max(limit, 1)
        fetched = 0
        page = 1
        while fetched < limit:
            page_size = min(50, limit - fetched)
            params: dict[str, Any] = {
                "search": query,
                "per-page": page_size,
                "page": page,
                "sort": "relevance_score:desc",
            }
            if year_from:
                params["filter"] = f"from_publication_date:{year_from}-01-01"

            req = Request(f"{OPENALEX_URL}?{urlencode(params)}", method="GET")
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except (URLError, HTTPError) as exc:
                raise RuntimeError(f"Search failed on Semantic Scholar and OpenAlex: {exc}") from exc

            results = payload.get("results", [])
            if not results:
                break

            for item in results:
                paper = self._openalex_to_metadata(item)
                if year_from and paper.year and paper.year < year_from:
                    continue
                papers.append(paper)
                fetched += 1
                if fetched >= limit:
                    break
            if len(results) < page_size:
                break
            page += 1

        deduped: dict[str, PaperMetadata] = {}
        for paper in papers:
            key = paper.doi or paper.paper_id
            if key not in deduped:
                deduped[key] = paper
        return list(deduped.values())

    @staticmethod
    def _to_metadata(item: dict[str, Any]) -> PaperMetadata:
        oa_pdf = item.get("openAccessPdf") or {}
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI") or item.get("doi")
        return PaperMetadata(
            paper_id=str(item.get("paperId") or ""),
            title=str(item.get("title") or "").strip(),
            abstract=str(item.get("abstract") or "").strip(),
            year=int(item["year"]) if isinstance(item.get("year"), int) else None,
            doi=str(doi) if doi else None,
            url=str(item.get("url")) if item.get("url") else None,
            venue=str(item.get("venue")) if item.get("venue") else None,
            citation_count=int(item.get("citationCount") or 0),
            open_access_pdf_url=str(oa_pdf.get("url")) if oa_pdf.get("url") else None,
        )

    @staticmethod
    def _openalex_to_metadata(item: dict[str, Any]) -> PaperMetadata:
        paper_id_raw = str(item.get("id") or "")
        paper_id = paper_id_raw.rsplit("/", 1)[-1] if "/" in paper_id_raw else paper_id_raw
        doi_raw = str(item.get("doi") or "")
        doi = doi_raw.replace("https://doi.org/", "").strip() if doi_raw else None
        primary = item.get("primary_location") or {}
        best_oa = item.get("best_oa_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name")
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
        pdf_url = (
            best_oa.get("pdf_url")
            or primary.get("pdf_url")
            or best_oa.get("landing_page_url")
            or primary.get("landing_page_url")
        )
        return PaperMetadata(
            paper_id=paper_id,
            title=str(item.get("display_name") or "").strip(),
            abstract=abstract,
            year=int(item["publication_year"]) if isinstance(item.get("publication_year"), int) else None,
            doi=doi,
            url=str(primary.get("landing_page_url")) if primary.get("landing_page_url") else None,
            venue=str(venue) if venue else None,
            citation_count=int(item.get("cited_by_count") or 0),
            open_access_pdf_url=str(pdf_url) if pdf_url else None,
        )


def _reconstruct_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        if isinstance(positions, list) and positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos <= max_pos:
                words[pos] = token
    return " ".join(word for word in words if word).strip()
