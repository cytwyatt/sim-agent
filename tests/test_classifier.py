from sim_agent.classify.sim_type import classify_simulation_type


def test_classify_md():
    text = "We performed molecular dynamics simulations in GROMACS with CHARMM36 force field."
    sim_type, confidence = classify_simulation_type(text)
    assert sim_type == "MD"
    assert confidence > 0.5


def test_classify_qmmm():
    text = "QM/MM calculations were carried out using ONIOM with a QM region and MM environment."
    sim_type, confidence = classify_simulation_type(text)
    assert sim_type == "QMMM"
    assert confidence > 0.5


def test_classify_other():
    text = "This work presents an experimental synthesis and microscopy characterization."
    sim_type, confidence = classify_simulation_type(text)
    assert sim_type == "Other/Unknown"
    assert 0.0 <= confidence <= 1.0
