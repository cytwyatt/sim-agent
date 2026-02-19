from sim_agent.extract.schema import normalize_core_details


def test_core_schema_normalization_adds_custom_fields():
    details = normalize_core_details(
        {
            "objective": "Test objective",
            "software_or_engine": "GROMACS",
            "computed_properties": "free energy",
            "system_build_steps": "solvated and neutralized",
            "custom_fields": [{"name": "initial_field", "value": "x", "unit": "", "evidence": ""}],
        },
        simulation_type="MD",
        requested_custom_fields=["temperature_schedule", "initial_field"],
    )
    assert details["simulation_type"] == "MD"
    assert isinstance(details["software_or_engine"], list)
    assert isinstance(details["computed_properties"], list)
    assert isinstance(details["system_build_steps"], list)
    names = {item["name"] for item in details["custom_fields"]}
    assert "temperature_schedule" in names
    assert "initial_field" in names
