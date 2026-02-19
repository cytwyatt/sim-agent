from sim_agent.config import ValidationSettings
from sim_agent.validate.sanity import run_sanity_checks


def test_sanity_flags_outlier_timestep_and_temperature():
    flags = run_sanity_checks(
        simulation_type="MD",
        core_details={
            "environment_conditions": "Simulation temperature set to 5000 K",
            "sampling_or_propagation_setup": "",
        },
        domain_details={"timestep": "20 fs"},
        settings=ValidationSettings(),
    )
    fields = {flag["field"] for flag in flags}
    assert "environment_conditions.temperature" in fields
    assert "domain_details.timestep" in fields
