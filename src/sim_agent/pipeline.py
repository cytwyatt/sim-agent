from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sim_agent.classify.sim_type import classify_simulation_type
from sim_agent.config import AppConfig
from sim_agent.downloader import download_open_access_pdf
from sim_agent.extract.core import extract_core_details
from sim_agent.extract.domains.md import extract_md_details
from sim_agent.extract.domains.qmmm import extract_qmmm_details_stub
from sim_agent.llm import OpenAIClient
from sim_agent.pdf_reader import extract_pdf_text
from sim_agent.ranking import rank_papers
from sim_agent.report.html import generate_html_report
from sim_agent.report.markdown import generate_markdown_report
from sim_agent.semantic_scholar import SemanticScholarClient
from sim_agent.store.json_store import JsonStore
from sim_agent.store.sqlite_store import SQLiteStore
from sim_agent.types import PaperRecord
from sim_agent.validate.sanity import run_sanity_checks


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    records: list[PaperRecord]
    manifest_path: Path
    markdown_path: Path
    html_path: Path
    summary_json_path: Path


def run_topic(
    topic: str,
    config: AppConfig,
    top_n: int,
    years: int,
    output_dir: str | Path,
    deep_profiles: Iterable[str],
    custom_fields: list[str],
    use_sqlite: bool,
) -> RunResult:
    run_id = _make_run_id(topic)
    created_at = datetime.now(timezone.utc).isoformat()
    deep_profiles_set = {profile.upper() for profile in deep_profiles}

    store = JsonStore(output_dir)
    run_dir = store.run_dir(run_id)
    (run_dir / "pdfs").mkdir(parents=True, exist_ok=True)

    scholar_client = SemanticScholarClient(api_key=config.semantic_scholar_api_key)
    llm_client = OpenAIClient(
        api_key=config.openai_api_key,
        model=config.models.openai_model,
        base_url=config.models.openai_base_url,
    )

    year_from = datetime.now(timezone.utc).year - years
    candidate_map = {}
    retrieval_queries = _build_retrieval_queries(topic)
    per_query_limit = max(top_n * 3, top_n)
    for retrieval_query in retrieval_queries:
        retrieved = scholar_client.search_papers(
            query=retrieval_query,
            limit=per_query_limit,
            year_from=year_from,
        )
        for paper in retrieved:
            key = paper.doi or paper.paper_id
            existing = candidate_map.get(key)
            if not existing:
                candidate_map[key] = paper
            elif len(paper.abstract or "") > len(existing.abstract or ""):
                candidate_map[key] = paper

    candidates = list(candidate_map.values())
    ranking_query = " ".join(retrieval_queries)
    ranked = rank_papers(candidates, query=ranking_query, weights=config.ranking, years_window=years)
    selected = ranked[:top_n]

    records: list[PaperRecord] = []
    for ranked_paper in selected:
        paper = ranked_paper.metadata
        errors: list[str] = []
        source_mode = "abstract"
        source_text = paper.abstract or ""

        if paper.open_access_pdf_url:
            pdf_path = run_dir / "pdfs" / f"{_safe_file_fragment(paper.paper_id)}.pdf"
            try:
                download_open_access_pdf(paper.open_access_pdf_url, pdf_path)
                text = extract_pdf_text(pdf_path)
                if text and len(text) > 200:
                    source_text = text
                    source_mode = "pdf"
                else:
                    errors.append("PDF parsed with insufficient text; fell back to abstract.")
            except Exception as exc:
                errors.append(f"PDF download/parse failed: {exc}")
        else:
            errors.append("No open-access PDF URL available; using abstract.")

        classify_text = f"{paper.title}\n{paper.abstract}\n{source_text[:5000]}"
        simulation_type, type_conf = classify_simulation_type(classify_text, llm_client=llm_client)

        core_details, summary, evidence = extract_core_details(
            text=source_text,
            paper=paper,
            simulation_type=simulation_type,
            custom_fields=custom_fields,
            llm_client=llm_client,
        )

        domain_details = None
        if simulation_type == "MD" and "MD" in deep_profiles_set:
            domain_details = extract_md_details(source_text, paper, llm_client=llm_client)
        elif simulation_type == "QMMM" and "QMMM" in deep_profiles_set:
            domain_details = extract_qmmm_details_stub()

        flags = run_sanity_checks(
            simulation_type=simulation_type,
            core_details=core_details,
            domain_details=domain_details,
            settings=config.validation,
        )

        record = PaperRecord(
            paper_metadata=paper,
            simulation_type=simulation_type,
            type_confidence=type_conf,
            core_simulation_details=core_details,
            domain_details=domain_details,
            evidence=evidence,
            validation_flags=flags,
            summary=summary,
            source_mode=source_mode,
            raw_excerpt=source_text[:1000],
            ranking_score=ranked_paper.score,
            errors=errors,
        )
        store.save_paper(run_id, record)
        records.append(record)

    markdown = generate_markdown_report(topic=topic, run_id=run_id, records=records)
    html = generate_html_report(topic=topic, run_id=run_id, records=records)
    markdown_path = store.save_markdown(run_id, markdown)
    html_path = store.save_html(run_id, html)
    summary_json_path = store.save_aggregate_json(run_id, records)

    manifest = {
        "run_id": run_id,
        "topic": topic,
        "created_at": created_at,
        "top_n": top_n,
        "years": years,
        "deep_profiles": sorted(deep_profiles_set),
        "custom_fields": custom_fields,
        "paper_count": len(records),
        "classification_breakdown": _count_by(records),
        "pdf_mode_count": sum(1 for r in records if r.source_mode == "pdf"),
        "abstract_mode_count": sum(1 for r in records if r.source_mode != "pdf"),
        "total_validation_flags": sum(len(r.validation_flags) for r in records),
        "model_used": config.models.openai_model if llm_client.enabled else "heuristic-fallback",
    }
    manifest_path = store.save_manifest(run_id, manifest)

    if use_sqlite:
        sqlite = SQLiteStore(Path(output_dir) / "sim_agent.db")
        sqlite.store_run(
            run_id=run_id,
            topic=topic,
            created_at=created_at,
            top_n=top_n,
            years=years,
            records=records,
        )

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        records=records,
        manifest_path=manifest_path,
        markdown_path=markdown_path,
        html_path=html_path,
        summary_json_path=summary_json_path,
    )


def _make_run_id(topic: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _safe_file_fragment("_".join(topic.lower().split())[:48]).strip("_")
    return f"run_{stamp}_{slug}" if slug else f"run_{stamp}"


def _safe_file_fragment(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _count_by(records: list[PaperRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in records:
        out[record.simulation_type] = out.get(record.simulation_type, 0) + 1
    return out


def _build_retrieval_queries(topic: str) -> list[str]:
    topic_norm = " ".join(topic.split())
    lower = topic_norm.lower()
    queries = [topic_norm]
    if "plastic" in lower and ("binding" in lower or "adsorption" in lower or "sorption" in lower):
        queries.extend(
            [
                "plastic-binding peptides molecular dynamics simulation",
                "microplastic binding peptide adsorption simulation",
                "polymer-binding peptide molecular simulation",
            ]
        )
    if "microplastic" in lower and "pollutant" in lower:
        queries.append("microplastic pollutant binding free energy molecular dynamics")
    deduped: list[str] = []
    seen = set()
    for query in queries:
        key = query.lower()
        if key not in seen:
            deduped.append(query)
            seen.add(key)
    return deduped
