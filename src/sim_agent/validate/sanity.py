from __future__ import annotations

import re
from typing import Any

from sim_agent.config import ValidationSettings


def run_sanity_checks(
    simulation_type: str,
    core_details: dict[str, Any],
    domain_details: dict[str, Any] | None,
    settings: ValidationSettings,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    text_sources = [
        str(core_details.get("environment_conditions", "")),
        str(core_details.get("sampling_or_propagation_setup", "")),
    ]

    temp = _extract_number(" ".join(text_sources), r"(\d+(\.\d+)?)\s*k")
    if temp is not None and not (settings.temperature_k_min <= temp <= settings.temperature_k_max):
        flags.append(
            {
                "severity": "warning",
                "field": "environment_conditions.temperature",
                "value": temp,
                "message": f"Temperature appears outside expected range [{settings.temperature_k_min}, {settings.temperature_k_max}] K.",
            }
        )

    pressure = _extract_number(" ".join(text_sources), r"(\d+(\.\d+)?)\s*(bar|atm|pa)")
    if pressure is not None and pressure > settings.pressure_bar_max:
        flags.append(
            {
                "severity": "warning",
                "field": "environment_conditions.pressure",
                "value": pressure,
                "message": f"Pressure appears very high (> {settings.pressure_bar_max} bar equivalent).",
            }
        )

    if simulation_type == "MD" and domain_details:
        timestep = str(domain_details.get("timestep") or "")
        timestep_fs = _convert_timestep_to_fs(timestep)
        if timestep_fs is not None and not (settings.md_timestep_fs_min <= timestep_fs <= settings.md_timestep_fs_max):
            flags.append(
                {
                    "severity": "warning",
                    "field": "domain_details.timestep",
                    "value": timestep,
                    "message": f"MD timestep outside expected range [{settings.md_timestep_fs_min}, {settings.md_timestep_fs_max}] fs.",
                }
            )
    return flags


def _extract_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text.lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _convert_timestep_to_fs(value: str) -> float | None:
    text = value.strip().lower()
    match = re.search(r"(\d+(\.\d+)?)\s*(fs|femtoseconds|ps)", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(3)
    if unit in {"fs", "femtoseconds"}:
        return number
    if unit == "ps":
        return number * 1000.0
    return None
