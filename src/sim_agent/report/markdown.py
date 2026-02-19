from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from sim_agent.types import PaperRecord


def generate_markdown_report(topic: str, run_id: str, records: list[PaperRecord]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    grouped: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in records:
        grouped[record.simulation_type].append(record)

    lines: list[str] = []
    lines.append(f"# Simulation Literature Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Generated at (UTC): `{generated_at}`")
    lines.append(f"- Topic: {topic}")
    lines.append(f"- Papers analyzed: {len(records)}")
    lines.append("")
    lines.append("## Simulation Type Breakdown")
    for sim_type, papers in sorted(grouped.items(), key=lambda kv: kv[0]):
        lines.append(f"- {sim_type}: {len(papers)}")
    lines.append("")

    lines.append("## Cross-Paper Synthesis")
    for sim_type, papers in sorted(grouped.items(), key=lambda kv: kv[0]):
        lines.append(f"### {sim_type}")
        engines = _collect_field(papers, "software_or_engine")
        properties = _collect_field(papers, "computed_properties")
        lines.append(f"- Frequent engines/tools: {', '.join(engines[:8]) if engines else 'N/A'}")
        lines.append(f"- Frequent computed properties: {', '.join(properties[:8]) if properties else 'N/A'}")
        lines.append("")

    lines.append("## Per-Paper Results")
    for record in records:
        meta = record.paper_metadata
        lines.append(f"### {meta.title}")
        lines.append(f"- Paper ID: `{meta.paper_id}`")
        lines.append(f"- Year: {meta.year if meta.year else 'Unknown'}")
        lines.append(f"- Type: `{record.simulation_type}` (confidence: {record.type_confidence:.2f})")
        lines.append(f"- Source mode: `{record.source_mode}`")
        lines.append(f"- Summary: {record.summary or 'N/A'}")
        lines.append(
            f"- System build protocol: {record.core_simulation_details.get('system_build_protocol') or 'N/A'}"
        )
        if record.domain_details and record.simulation_type == "MD":
            lines.append("- MD details:")
            lines.append(f"  - Engine: {record.domain_details.get('engine') or 'N/A'}")
            lines.append(f"  - Force field: {record.domain_details.get('force_field') or 'N/A'}")
            lines.append(f"  - Ensemble: {record.domain_details.get('ensemble') or 'N/A'}")
            lines.append(f"  - Timestep: {record.domain_details.get('timestep') or 'N/A'}")
            lines.append(f"  - Production time: {record.domain_details.get('production_time') or 'N/A'}")
        if record.validation_flags:
            lines.append(f"- Validation flags: {len(record.validation_flags)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _collect_field(records: list[PaperRecord], key: str) -> list[str]:
    counter: dict[str, int] = {}
    for record in records:
        value = record.core_simulation_details.get(key)
        if isinstance(value, list):
            for item in value:
                _inc(counter, str(item))
        elif value:
            _inc(counter, str(value))
    ordered = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    return [item[0] for item in ordered]


def _inc(counter: dict[str, int], key: str) -> None:
    normalized = key.strip()
    if not normalized:
        return
    counter[normalized] = counter.get(normalized, 0) + 1
