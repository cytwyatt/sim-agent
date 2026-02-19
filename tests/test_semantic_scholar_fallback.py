import io
import json
from urllib.error import HTTPError

from sim_agent.semantic_scholar import SemanticScholarClient


class _Resp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_semantic_scholar_falls_back_to_openalex(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=30):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(req.full_url, 429, "Too Many Requests", hdrs=None, fp=io.BytesIO(b"{}"))
        return _Resp(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "Semicrystalline polymer simulation",
                        "publication_year": 2022,
                        "doi": "https://doi.org/10.1/abc",
                        "cited_by_count": 12,
                        "primary_location": {
                            "landing_page_url": "https://example.org/paper",
                            "source": {"display_name": "Journal X"},
                        },
                        "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
                        "abstract_inverted_index": {"Semicrystalline": [0], "polymer": [1]},
                    }
                ]
            }
        )

    monkeypatch.setattr("sim_agent.semantic_scholar.urlopen", fake_urlopen)
    client = SemanticScholarClient(api_key=None)
    papers = client.search_papers("semicrystalline polymer", limit=5, year_from=2020)
    assert len(papers) == 1
    assert papers[0].paper_id == "W123"
    assert papers[0].doi == "10.1/abc"
    assert papers[0].open_access_pdf_url.endswith(".pdf")


def test_openalex_pagination_respects_limit(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=30):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(req.full_url, 429, "Too Many Requests", hdrs=None, fp=io.BytesIO(b"{}"))
        if "page=1" in req.full_url:
            results = [
                {
                    "id": f"https://openalex.org/W{i}",
                    "display_name": f"Paper {i}",
                    "publication_year": 2023,
                    "doi": f"https://doi.org/10.1/{i}",
                    "cited_by_count": i,
                    "primary_location": {"landing_page_url": f"https://example.org/{i}", "source": {"display_name": "J"}},
                    "best_oa_location": {"pdf_url": f"https://example.org/{i}.pdf"},
                    "abstract_inverted_index": {"Paper": [0], str(i): [1]},
                }
                for i in range(1, 51)
            ]
            return _Resp({"results": results})
        if "page=2" in req.full_url:
            results = [
                {
                    "id": f"https://openalex.org/W{i}",
                    "display_name": f"Paper {i}",
                    "publication_year": 2023,
                    "doi": f"https://doi.org/10.1/{i}",
                    "cited_by_count": i,
                    "primary_location": {"landing_page_url": f"https://example.org/{i}", "source": {"display_name": "J"}},
                    "best_oa_location": {"pdf_url": f"https://example.org/{i}.pdf"},
                    "abstract_inverted_index": {"Paper": [0], str(i): [1]},
                }
                for i in range(51, 71)
            ]
            return _Resp({"results": results})
        return _Resp({"results": []})

    monkeypatch.setattr("sim_agent.semantic_scholar.urlopen", fake_urlopen)
    client = SemanticScholarClient(api_key=None)
    papers = client.search_papers("plastic binding peptide", limit=70, year_from=2020)
    assert len(papers) == 70
    assert papers[0].paper_id == "W1"
    assert papers[-1].paper_id == "W70"
