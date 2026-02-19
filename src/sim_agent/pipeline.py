from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable

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
    candidate_titles_path: Path


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
    topic_keywords = _expand_topic_keywords(topic, llm_client)
    anchor_groups, exclude_title_terms, topic_constraint_source = _infer_topic_constraints(topic, llm_client)
    candidate_map = {}
    retrieval_queries = _build_retrieval_queries(topic, topic_keywords)
    heuristic_pool_target = max(100, top_n * 5)
    per_query_limit = min(
        100,
        max(top_n, (heuristic_pool_target // max(len(retrieval_queries), 1)) * 2),
    )
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
    ranking_query = " ".join(topic_keywords if topic_keywords else retrieval_queries)
    ranked = rank_papers(candidates, query=ranking_query, weights=config.ranking, years_window=years)
    filtered_ranked = [
        paper
        for paper in ranked
        if _paper_matches_topic_constraints(paper, anchor_groups, exclude_title_terms)
    ]
    anchor_filter_mode = "applied"
    ranked_for_pool = filtered_ranked
    if len(ranked_for_pool) < top_n:
        anchor_filter_mode = "relaxed_fallback"
        ranked_for_pool = ranked

    heuristic_pool = ranked_for_pool[:heuristic_pool_target]
    candidate_titles_payload = _build_candidate_titles_payload(heuristic_pool)

    title_selection_mode = "heuristic-ranking"
    llm_selected = _select_top_by_title_with_llm(
        topic,
        heuristic_pool,
        top_n,
        llm_client,
        anchor_groups,
        exclude_title_terms,
    )
    if llm_selected:
        selected = llm_selected
        title_selection_mode = "llm-title-rerank"
    else:
        selected = heuristic_pool[:top_n]
    selected_ids = [paper.metadata.paper_id for paper in selected]
    candidate_titles_path = store.save_candidate_titles(
        run_id,
        {
            "topic": topic,
            "retrieval_keywords": topic_keywords,
            "retrieval_queries": retrieval_queries,
            "candidate_count": len(candidates),
            "anchor_groups": anchor_groups,
            "exclude_title_terms": exclude_title_terms,
            "topic_constraint_source": topic_constraint_source,
            "anchor_filter_mode": anchor_filter_mode,
            "heuristic_pool_size": len(heuristic_pool),
            "title_selection_mode": title_selection_mode,
            "selected_paper_ids": selected_ids,
            "titles": candidate_titles_payload,
        },
    )

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
        "retrieval_keywords": topic_keywords,
        "retrieval_queries": retrieval_queries,
        "candidate_count": len(candidates),
        "anchor_groups": anchor_groups,
        "exclude_title_terms": exclude_title_terms,
        "topic_constraint_source": topic_constraint_source,
        "anchor_filter_mode": anchor_filter_mode,
        "heuristic_pool_size": len(heuristic_pool),
        "title_selection_mode": title_selection_mode,
        "candidate_titles_path": str(candidate_titles_path),
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
        candidate_titles_path=candidate_titles_path,
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


def _expand_topic_keywords(topic: str, llm_client: OpenAIClient) -> list[str]:
    topic_norm = " ".join(topic.split())
    base = _heuristic_topic_keywords(topic_norm)

    if not llm_client.enabled:
        return _sanitize_keywords(_dedupe_texts([topic_norm, *base], max_items=10))

    response = llm_client.chat_json(
        system_prompt=(
            "You expand user research topics into concise paper-search keywords. "
            "Return JSON with key 'keywords' as a list of short keyword phrases."
        ),
        user_prompt=(
            f"Topic: {topic_norm}\n"
            "Return 5-10 concise search keywords/phrases focused on simulation literature retrieval."
        ),
        temperature=0.0,
        max_tokens=300,
    )
    if not response:
        return _sanitize_keywords(_dedupe_texts([topic_norm, *base], max_items=10))

    llm_keywords = _coerce_text_list(response.get("keywords"))
    if not llm_keywords:
        llm_keywords = _coerce_text_list(response.get("key_phrases"))
    return _sanitize_keywords(_dedupe_texts([topic_norm, *llm_keywords, *base], max_items=10))


def _heuristic_topic_keywords(topic: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9\+]+", topic.lower())
    stop_words = {
        "a",
        "an",
        "and",
        "or",
        "the",
        "on",
        "of",
        "to",
        "for",
        "in",
        "with",
        "by",
        "using",
        "considering",
        "study",
        "analysis",
    }
    filtered = [tok for tok in tokens if len(tok) > 2 and tok not in stop_words]
    phrases: list[str] = []
    phrases.extend(filtered[:6])
    phrases.extend(" ".join(filtered[i : i + 2]) for i in range(max(0, min(len(filtered) - 1, 4))))
    return _dedupe_texts(phrases, max_items=10)


def _infer_topic_constraints(topic: str, llm_client: OpenAIClient) -> tuple[list[list[str]], list[str], str]:
    heuristic_groups = _build_topic_anchor_groups(topic)
    heuristic_excludes = _default_exclude_title_terms(topic)

    if not llm_client.enabled:
        return heuristic_groups, heuristic_excludes, "heuristic"

    response = llm_client.chat_json(
        system_prompt=(
            "You derive paper-selection constraints from a research topic. "
            "Return JSON with keys: anchor_groups (list of synonym lists for must-have concepts), "
            "exclude_title_terms (list of title terms to exclude), rationale."
        ),
        user_prompt=(
            f"Topic: {topic}\n"
            "Rules:\n"
            "- anchor_groups should include 1-3 concept groups that best define topical relevance.\n"
            "- Use concise keywords, not long phrases.\n"
            "- exclude_title_terms should include generic off-target patterns only when appropriate.\n"
            "- If topic asks for review/survey, do not exclude review terms.\n"
        ),
        temperature=0.0,
        max_tokens=500,
    )
    if not response:
        return heuristic_groups, heuristic_excludes, "heuristic"

    parsed_groups = _coerce_nested_text_list(response.get("anchor_groups"))
    cleaned_groups = _sanitize_anchor_groups(parsed_groups)
    if not cleaned_groups:
        cleaned_groups = heuristic_groups

    parsed_excludes = _coerce_text_list(response.get("exclude_title_terms"))
    cleaned_excludes = _sanitize_exclude_terms(parsed_excludes)
    if not cleaned_excludes:
        cleaned_excludes = heuristic_excludes

    return cleaned_groups, cleaned_excludes, "llm"


def _build_retrieval_queries(topic: str, keywords: list[str]) -> list[str]:
    topic_norm = " ".join(topic.split())
    lower = topic_norm.lower()
    queries = [topic_norm, *keywords]
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
    return _dedupe_texts(queries, max_items=12)


def _build_candidate_titles_payload(ranked: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for idx, paper in enumerate(ranked, start=1):
        payload.append(
            {
                "ref": f"R{idx:03d}",
                "paper_id": paper.metadata.paper_id,
                "year": paper.metadata.year,
                "title": paper.metadata.title,
                "ranking_score": round(paper.score, 6),
            }
        )
    return payload


def _select_top_by_title_with_llm(
    topic: str,
    ranked_pool: list[Any],
    top_n: int,
    llm_client: OpenAIClient,
    anchor_groups: list[list[str]] | None = None,
    exclude_title_terms: list[str] | None = None,
) -> list[Any]:
    if not llm_client.enabled or not ranked_pool:
        return []

    anchor_groups = anchor_groups or []
    exclude_title_terms = exclude_title_terms or []
    anchor_notes = ", ".join("/".join(group) for group in anchor_groups) if anchor_groups else "N/A"
    exclude_notes = ", ".join(exclude_title_terms) if exclude_title_terms else "N/A"
    ref_to_paper: dict[str, Any] = {}
    lines: list[str] = []
    for idx, paper in enumerate(ranked_pool, start=1):
        ref = f"R{idx:03d}"
        ref_to_paper[ref] = paper
        year = paper.metadata.year if paper.metadata.year else "Unknown"
        lines.append(f"{ref} | {year} | {paper.metadata.title}")

    response = llm_client.chat_json(
        system_prompt=(
            "You select paper titles that are most relevant to a research topic. "
            "Return JSON with key 'selected_refs' as a ranked list of refs."
        ),
        user_prompt=(
            f"Topic: {topic}\n"
            f"Select top {top_n} most relevant refs.\n"
            "Prioritize direct topic match and simulation relevance.\n"
            f"Strong topic anchor concepts: {anchor_notes}\n"
            f"Prefer excluding titles containing: {exclude_notes}\n"
            "Exclude generic reviews or off-topic domains when anchor concepts are missing.\n\n"
            "Candidates:\n"
            + "\n".join(lines[:120])
        ),
        temperature=0.0,
        max_tokens=1000,
    )
    if not response:
        return []

    selected_refs = _coerce_text_list(response.get("selected_refs"))
    if not selected_refs:
        selected_refs = _coerce_text_list(response.get("refs"))
    picked: list[Any] = []
    seen = set()
    for raw_ref in selected_refs:
        ref = raw_ref.strip().upper()
        if ref in ref_to_paper and ref not in seen:
            paper = ref_to_paper[ref]
            if not _paper_matches_topic_constraints(paper, anchor_groups, exclude_title_terms):
                continue
            picked.append(paper)
            seen.add(ref)
        if len(picked) >= top_n:
            break

    if len(picked) < top_n:
        for idx in range(1, len(ranked_pool) + 1):
            ref = f"R{idx:03d}"
            if ref not in seen:
                paper = ref_to_paper[ref]
                if not _paper_matches_topic_constraints(paper, anchor_groups, exclude_title_terms):
                    continue
                picked.append(paper)
                seen.add(ref)
            if len(picked) >= top_n:
                break
    if len(picked) < top_n:
        for idx in range(1, len(ranked_pool) + 1):
            ref = f"R{idx:03d}"
            if ref not in seen:
                picked.append(ref_to_paper[ref])
                seen.add(ref)
            if len(picked) >= top_n:
                break
    return picked[:top_n]


def _coerce_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    return []


def _dedupe_texts(values: list[str], max_items: int = 20) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for value in values:
        item = " ".join(str(value).split()).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
        if len(deduped) >= max_items:
            break
    return deduped


def _build_topic_anchor_groups(topic: str) -> list[list[str]]:
    generic_terms = {
        "molecular",
        "simulation",
        "simulations",
        "dynamics",
        "dynamic",
        "study",
        "analysis",
        "model",
        "modeling",
        "modelling",
        "considering",
        "effect",
        "effects",
        "properties",
        "topic",
        "research",
    }
    tokens = re.findall(r"[a-zA-Z0-9\+]+", topic.lower())
    groups: list[list[str]] = []
    seen = set()

    def add_group(values: list[str]) -> None:
        key = tuple(values)
        if key in seen:
            return
        seen.add(key)
        groups.append(values)

    for token in tokens:
        if token in generic_terms or len(token) < 4:
            continue
        add_group([token])

    return groups[:4]


def _paper_matches_anchor_groups(paper: Any, anchor_groups: list[list[str]]) -> bool:
    if not anchor_groups:
        return True
    text = f"{paper.metadata.title} {paper.metadata.abstract}".lower()
    matched = 0
    for group in anchor_groups:
        if any(token.lower() in text for token in group):
            matched += 1
    required = len(anchor_groups) if len(anchor_groups) <= 2 else 2
    return matched >= required


def _paper_matches_topic_constraints(
    paper: Any,
    anchor_groups: list[list[str]],
    exclude_title_terms: list[str],
) -> bool:
    if exclude_title_terms and _paper_has_excluded_title_term(paper, exclude_title_terms):
        return False
    return _paper_matches_anchor_groups(paper, anchor_groups)


def _paper_has_excluded_title_term(paper: Any, exclude_title_terms: list[str]) -> bool:
    title = str(paper.metadata.title or "").lower()
    return any(term in title for term in exclude_title_terms)


def _default_exclude_title_terms(topic: str) -> list[str]:
    lower = topic.lower()
    if any(token in lower for token in {"review", "survey", "overview", "mini review"}):
        return []
    return ["mini review", "tutorial", "editorial"]


def _sanitize_keywords(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        item = " ".join(value.split()).strip()
        if not item:
            continue
        tokens = re.findall(r"[a-zA-Z0-9\+]+", item.lower())
        if len(tokens) == 1 and tokens[0] in _generic_keyword_tokens():
            continue
        out.append(item)
    return _dedupe_texts(out, max_items=10)


def _sanitize_anchor_groups(groups: list[list[str]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for group in groups:
        tokens: list[str] = []
        for item in group:
            token = " ".join(str(item).strip().lower().split())
            if not token:
                continue
            if token in _generic_keyword_tokens():
                continue
            tokens.append(token)
        deduped = _dedupe_texts(tokens, max_items=8)
        if deduped:
            cleaned.append(deduped)
    return cleaned[:4]


def _sanitize_exclude_terms(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = " ".join(str(value).lower().split())
        if not item or len(item) < 4:
            continue
        cleaned.append(item)
    return _dedupe_texts(cleaned, max_items=10)


def _coerce_nested_text_list(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    out: list[list[str]] = []
    for item in value:
        if isinstance(item, list):
            arr = [str(part).strip() for part in item if str(part).strip()]
            if arr:
                out.append(arr)
        elif isinstance(item, str):
            parts = [part.strip() for part in re.split(r"[/,;|]+", item) if part.strip()]
            if parts:
                out.append(parts)
    return out


def _generic_keyword_tokens() -> set[str]:
    return {
        "molecular",
        "simulation",
        "simulations",
        "dynamics",
        "dynamic",
        "model",
        "modeling",
        "modelling",
        "study",
        "analysis",
        "topic",
        "research",
    }
