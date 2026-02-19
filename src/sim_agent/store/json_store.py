from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sim_agent.types import PaperRecord


class JsonStore:
    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)
        self.runs_root = self.output_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "papers").mkdir(parents=True, exist_ok=True)
        return path

    def save_paper(self, run_id: str, record: PaperRecord) -> Path:
        run_dir = self.run_dir(run_id)
        path = run_dir / "papers" / f"{_safe_name(record.paper_metadata.paper_id)}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_manifest(self, run_id: str, manifest: dict[str, Any]) -> Path:
        run_dir = self.run_dir(run_id)
        path = run_dir / "run_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_markdown(self, run_id: str, markdown: str) -> Path:
        run_dir = self.run_dir(run_id)
        path = run_dir / "summary.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def save_aggregate_json(self, run_id: str, records: list[PaperRecord]) -> Path:
        run_dir = self.run_dir(run_id)
        path = run_dir / "summary.json"
        payload = [record.to_dict() for record in records]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_html(self, run_id: str, html: str) -> Path:
        run_dir = self.run_dir(run_id)
        path = run_dir / "summary.html"
        path.write_text(html, encoding="utf-8")
        return path

    def load_paper(self, run_id: str, paper_id: str) -> dict[str, Any]:
        path = self.runs_root / run_id / "papers" / f"{_safe_name(paper_id)}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        path = self.runs_root / run_id / "run_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_markdown(self, run_id: str) -> str:
        path = self.runs_root / run_id / "summary.md"
        return path.read_text(encoding="utf-8")

    def load_aggregate_json(self, run_id: str) -> list[dict[str, Any]]:
        path = self.runs_root / run_id / "summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_html(self, run_id: str) -> str:
        path = self.runs_root / run_id / "summary.html"
        return path.read_text(encoding="utf-8")


def _safe_name(raw: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in raw)
