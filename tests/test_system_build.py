from sim_agent.extract.system_build import extract_system_build_protocol


def test_extract_system_build_protocol_md():
    text = (
        "The starting structure was obtained from the PDB database. "
        "System preparation was performed using CHARMM-GUI and parameterized with CHARMM36. "
        "The protein was solvated in a TIP3P water box and neutralized with Na+ and Cl- ions. "
        "Energy minimization was followed by NVT and NPT equilibration before production runs."
    )
    result = extract_system_build_protocol(text=text, simulation_type="MD")
    assert result["system_build_protocol"]
    assert len(result["system_build_steps"]) >= 3
    assert len(result["evidence"]) >= 2


def test_extract_system_build_protocol_polymer_abstract_style():
    text = (
        "A coarse-grained model of semicrystalline polymer was constructed with 200 chains and periodic boundaries. "
        "The melt was equilibrated at high temperature and then quenched to induce crystallization and lamella growth."
    )
    result = extract_system_build_protocol(text=text, simulation_type="MD")
    assert result["system_build_protocol"]
    assert any("constructed" in step.lower() or "quench" in step.lower() for step in result["system_build_steps"])
