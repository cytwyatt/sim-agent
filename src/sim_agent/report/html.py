from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape

from sim_agent.types import PaperRecord


def generate_html_report(topic: str, run_id: str, records: list[PaperRecord]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    grouped: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in records:
        grouped[record.simulation_type].append(record)

    type_rows = "\n".join(
        f"<li><strong>{escape(sim_type)}:</strong> {len(items)}</li>"
        for sim_type, items in sorted(grouped.items(), key=lambda kv: kv[0])
    )

    paper_sections = []
    for record in records:
        meta = record.paper_metadata
        title = escape(meta.title or "Untitled")
        year = escape(str(meta.year)) if meta.year else "Unknown"
        sim_type = escape(record.simulation_type)
        conf = f"{record.type_confidence:.2f}"
        source_mode = escape(record.source_mode)
        summary = escape(record.summary or "N/A")
        build_protocol = escape(record.core_simulation_details.get("system_build_protocol") or "N/A")
        raw_steps = record.core_simulation_details.get("system_build_steps") or []
        step_items = "".join(f"<li>{escape(str(step))}</li>" for step in raw_steps) or "<li>N/A</li>"

        ref_link = meta.url or (f"https://doi.org/{meta.doi}" if meta.doi else "")
        link_html = (
            f'<a href="{escape(ref_link)}" target="_blank" rel="noopener">{escape(ref_link)}</a>'
            if ref_link
            else "N/A"
        )

        md_details_html = ""
        if record.domain_details and record.simulation_type == "MD":
            d = record.domain_details
            md_details_html = (
                "<h4>MD Details</h4>"
                "<ul>"
                f"<li>Engine: {escape(str(d.get('engine') or 'N/A'))}</li>"
                f"<li>Force field: {escape(str(d.get('force_field') or 'N/A'))}</li>"
                f"<li>Ensemble: {escape(str(d.get('ensemble') or 'N/A'))}</li>"
                f"<li>Timestep: {escape(str(d.get('timestep') or 'N/A'))}</li>"
                f"<li>Production time: {escape(str(d.get('production_time') or 'N/A'))}</li>"
                "</ul>"
            )

        paper_sections.append(
            f"""
            <section class="paper card">
              <h3>{title}</h3>
              <p class="meta"><strong>Paper ID:</strong> {escape(meta.paper_id)} |
              <strong>Year:</strong> {year} |
              <strong>Type:</strong> {sim_type} (confidence {conf}) |
              <strong>Source:</strong> {source_mode}</p>
              <p><strong>Reference:</strong> {link_html}</p>
              <p><strong>Summary:</strong> {summary}</p>
              <h4>System Build Protocol</h4>
              <p>{build_protocol}</p>
              <h4>System Build Steps</h4>
              <ol>{step_items}</ol>
              {md_details_html}
            </section>
            """
        )

    body = "\n".join(paper_sections)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simulation Literature Report - {escape(run_id)}</title>
  <style>
    :root {{
      --bg: #f6f8f5;
      --ink: #1f2a21;
      --muted: #5c695f;
      --card: #ffffff;
      --line: #d4ddd5;
      --accent: #0d6b46;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1080px;
      margin: 30px auto;
      padding: 0 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 14px;
      box-shadow: 0 5px 16px rgba(0, 0, 0, 0.04);
    }}
    h1, h2, h3, h4 {{ margin-top: 0; }}
    h2, h3, h4 {{ color: var(--accent); }}
    p, li {{ color: var(--muted); }}
    .meta {{ font-size: 0.92rem; }}
    a {{ color: #0b57a5; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Simulation Literature Report</h1>
      <p><strong>Run ID:</strong> {escape(run_id)}</p>
      <p><strong>Generated (UTC):</strong> {escape(generated_at)}</p>
      <p><strong>Topic:</strong> {escape(topic)}</p>
      <p><strong>Papers analyzed:</strong> {len(records)}</p>
      <h2>Simulation Type Breakdown</h2>
      <ul>
        {type_rows}
      </ul>
    </section>
    {body}
  </div>
</body>
</html>
"""
