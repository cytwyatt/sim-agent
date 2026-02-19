from pathlib import Path

from sim_agent.config import AppConfig
from sim_agent.pipeline import run_topic
from sim_agent.types import PaperMetadata


def test_pipeline_mixed_domain(monkeypatch, tmp_path: Path):
    def fake_search(self, query: str, limit: int = 30, year_from: int | None = None):
        return [
            PaperMetadata(
                paper_id="md_paper",
                title="Protein molecular dynamics with GROMACS",
                abstract="Molecular dynamics simulation in NPT with 2 fs timestep and CHARMM36.",
                year=2024,
                citation_count=20,
                open_access_pdf_url="https://example.com/md.pdf",
            ),
            PaperMetadata(
                paper_id="qm_paper",
                title="DFT study of catalyst surface",
                abstract="Density functional theory calculations with PBE functional.",
                year=2023,
                citation_count=10,
                open_access_pdf_url="https://example.com/qm.pdf",
            ),
            PaperMetadata(
                paper_id="qmmm_paper",
                title="QM/MM investigation of enzyme reaction",
                abstract="QM/MM approach with ONIOM for catalytic mechanism.",
                year=2022,
                citation_count=8,
                open_access_pdf_url=None,
            ),
        ]

    def fake_download(url: str, target_path: Path, timeout_seconds: int = 45):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"%PDF-1.4 test")
        return target_path

    def fake_pdf_text(path: Path, max_pages: int = 30, max_chars: int = 80000):
        return (
            "Molecular dynamics with GROMACS CHARMM36 in NPT ensemble using 2 fs timestep "
            "and 100 ns production in TIP3P solvent. The system was built with CHARMM-GUI, "
            "solvated in a water box, and neutralized with Na+ ions."
            if "md_paper" in path.name
            else "Density functional theory with PBE functional. A surface supercell was constructed before calculations."
        )

    monkeypatch.setattr("sim_agent.pipeline.SemanticScholarClient.search_papers", fake_search)
    monkeypatch.setattr("sim_agent.pipeline.download_open_access_pdf", fake_download)
    monkeypatch.setattr("sim_agent.pipeline.extract_pdf_text", fake_pdf_text)

    config = AppConfig()
    config.openai_api_key = None
    result = run_topic(
        topic="mixed simulation topic",
        config=config,
        top_n=3,
        years=10,
        output_dir=tmp_path,
        deep_profiles=["MD"],
        custom_fields=["ionic_strength"],
        use_sqlite=False,
    )

    assert len(result.records) == 3
    assert result.manifest_path.exists()
    assert result.markdown_path.exists()
    assert result.html_path.exists()
    assert result.summary_json_path.exists()
    assert result.candidate_titles_path.exists()

    types = {record.simulation_type for record in result.records}
    assert "MD" in types
    assert "QM" in types or "QMMM" in types

    md_records = [record for record in result.records if record.simulation_type == "MD"]
    assert md_records
    assert md_records[0].domain_details is not None
    assert md_records[0].core_simulation_details["system_build_protocol"]
