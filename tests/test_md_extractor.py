from sim_agent.extract.domains.md import extract_md_details
from sim_agent.types import PaperMetadata


def test_md_extractor_heuristics():
    text = (
        "Molecular dynamics simulations were performed with GROMACS 2022 and CHARMM36 force field. "
        "Protein was solvated in TIP3P water. Simulations were run in NPT ensemble with "
        "a 2 fs time step, using Nose-Hoover thermostat and Parrinello-Rahman barostat. "
        "Production simulation time was 100 ns with PME electrostatics."
    )
    paper = PaperMetadata(paper_id="p1", title="MD test", abstract=text)
    details = extract_md_details(text, paper, llm_client=None)
    assert details["engine"] == "GROMACS"
    assert "charmm" in details["force_field"].lower()
    assert details["ensemble"] == "npt"
    assert details["timestep"] == "2 fs"
    assert details["production_time"] == "100 ns"
    assert len(details["evidence"]) > 0
