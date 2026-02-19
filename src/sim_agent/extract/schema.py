from __future__ import annotations

from typing import Any

CORE_FIELDS = [
    "objective",
    "simulation_type",
    "software_or_engine",
    "theory_or_force_model",
    "system_description",
    "system_build_protocol",
    "system_build_steps",
    "environment_conditions",
    "sampling_or_propagation_setup",
    "runtime_or_compute_budget",
    "computed_properties",
    "key_limitations",
    "custom_fields",
]

MD_FIELDS = [
    "engine",
    "engine_version",
    "force_field",
    "solvent_model",
    "ion_parameters",
    "ensemble",
    "thermostat",
    "barostat",
    "timestep",
    "equilibration_time",
    "production_time",
    "long_range_method",
    "constraints",
    "cutoffs",
    "system_size",
    "composition",
    "replicates",
    "enhanced_sampling",
    "hardware_notes",
    "evidence",
]


def default_core_details(simulation_type: str) -> dict[str, Any]:
    return {
        "objective": "",
        "simulation_type": simulation_type,
        "software_or_engine": [],
        "theory_or_force_model": "",
        "system_description": "",
        "system_build_protocol": "",
        "system_build_steps": [],
        "environment_conditions": "",
        "sampling_or_propagation_setup": "",
        "runtime_or_compute_budget": "",
        "computed_properties": [],
        "key_limitations": "",
        "custom_fields": [],
    }


def normalize_core_details(
    data: dict[str, Any] | None,
    simulation_type: str,
    requested_custom_fields: list[str] | None = None,
) -> dict[str, Any]:
    details = default_core_details(simulation_type)
    requested_custom_fields = requested_custom_fields or []
    if data:
        for k in CORE_FIELDS:
            if k in data and data[k] is not None:
                details[k] = data[k]

    if not isinstance(details["software_or_engine"], list):
        details["software_or_engine"] = [str(details["software_or_engine"])]
    if not isinstance(details["computed_properties"], list):
        details["computed_properties"] = [str(details["computed_properties"])]
    if not isinstance(details["system_build_steps"], list):
        details["system_build_steps"] = [str(details["system_build_steps"])]
    if not isinstance(details["custom_fields"], list):
        details["custom_fields"] = []

    existing_names = {str(item.get("name")).strip().lower() for item in details["custom_fields"] if isinstance(item, dict)}
    for name in requested_custom_fields:
        if name.strip().lower() not in existing_names:
            details["custom_fields"].append({"name": name, "value": "", "unit": "", "evidence": ""})
    details["simulation_type"] = simulation_type
    return details


def normalize_md_details(data: dict[str, Any] | None) -> dict[str, Any]:
    details = {key: "" for key in MD_FIELDS}
    details["evidence"] = []
    if data:
        for key in MD_FIELDS:
            if key in data and data[key] is not None:
                details[key] = data[key]
    if not isinstance(details["evidence"], list):
        details["evidence"] = []
    return details
