from __future__ import annotations

import re
from typing import Any

from sim_agent.extract.schema import normalize_md_details
from sim_agent.llm import OpenAIClient
from sim_agent.types import PaperMetadata


def extract_md_details(
    text: str,
    paper: PaperMetadata,
    llm_client: OpenAIClient | None = None,
) -> dict[str, Any]:
    if llm_client and llm_client.enabled:
        llm = _extract_with_llm(text, paper, llm_client)
        if llm:
            return normalize_md_details(llm.get("md_details"))
    return _heuristic_md_extraction(text)


def _extract_with_llm(text: str, paper: PaperMetadata, llm_client: OpenAIClient) -> dict[str, Any] | None:
    system_prompt = (
        "You extract molecular dynamics protocol details from paper text and return JSON. "
        "Return key 'md_details' with fields: engine, engine_version, force_field, solvent_model, "
        "ion_parameters, ensemble, thermostat, barostat, timestep, equilibration_time, production_time, "
        "long_range_method, constraints, cutoffs, system_size, composition, replicates, "
        "enhanced_sampling, hardware_notes, evidence(list of {field,snippet,confidence})."
    )
    user_prompt = f"Title: {paper.title}\nAbstract: {paper.abstract}\nPaper text:\n{text[:14000]}"
    return llm_client.chat_json(system_prompt, user_prompt, temperature=0.0, max_tokens=1500)


def _heuristic_md_extraction(text: str) -> dict[str, Any]:
    lower = text.lower()
    md = normalize_md_details(None)
    evidence: list[dict[str, Any]] = []

    engine = _find_first(lower, ["gromacs", "lammps", "amber", "namd", "charmm", "desmond", "openmm"])
    if engine:
        md["engine"] = engine.upper()
        evidence.append({"field": "engine", "snippet": _snippet(lower, engine), "confidence": 0.70})

    force_field = _find_first(lower, ["charmm36", "charmm", "amber99", "amber14", "opls", "martini"])
    if force_field:
        md["force_field"] = force_field
        evidence.append({"field": "force_field", "snippet": _snippet(lower, force_field), "confidence": 0.66})

    solvent = _find_first(lower, ["tip3p", "tip4p", "spce", "spc/e", "implicit solvent"])
    if solvent:
        md["solvent_model"] = solvent
        evidence.append({"field": "solvent_model", "snippet": _snippet(lower, solvent), "confidence": 0.65})

    md["ensemble"] = _find_first(lower, ["nvt", "npt", "nve"]) or ""
    if md["ensemble"]:
        evidence.append({"field": "ensemble", "snippet": _snippet(lower, md["ensemble"]), "confidence": 0.62})

    thermostat = _find_first(lower, ["nose-hoover", "langevin", "berendsen", "velocity rescale"])
    if thermostat:
        md["thermostat"] = thermostat

    barostat = _find_first(lower, ["parrinello-rahman", "berendsen barostat", "monte carlo barostat"])
    if barostat:
        md["barostat"] = barostat

    timestep = _extract_value(lower, r"(\d+(\.\d+)?)\s*(fs|femtoseconds|ps)\s*(time\s*step|timestep)")
    if timestep:
        md["timestep"] = timestep
        evidence.append({"field": "timestep", "snippet": _snippet(lower, timestep.split()[0]), "confidence": 0.60})

    prod = _extract_value(
        lower,
        r"(production\s*(simulation|run|time)\s*(of|was)?\s*)?(\d+(\.\d+)?)\s*(ns|ps|us|µs)",
        value_group=4,
        unit_group=6,
    )
    if prod:
        md["production_time"] = prod
        evidence.append(
            {"field": "production_time", "snippet": _snippet(lower, prod.split()[0]), "confidence": 0.58}
        )

    equil = _extract_value(lower, r"(\d+(\.\d+)?)\s*(ns|ps)\s*(equilibration|equilibrated)")
    if equil:
        md["equilibration_time"] = equil

    long_range = _find_first(lower, ["particle mesh ewald", "pme", "ewald"])
    if long_range:
        md["long_range_method"] = long_range

    constraints = _find_first(lower, ["shake", "lincs", "settle"])
    if constraints:
        md["constraints"] = constraints

    cutoffs = _extract_value(lower, r"(\d+(\.\d+)?)\s*(nm|angstrom|å)\s*cutoff")
    if cutoffs:
        md["cutoffs"] = cutoffs

    hardware = _find_first(lower, ["gpu", "nvidia", "cuda", "cpu core", "supercomputer"])
    if hardware:
        md["hardware_notes"] = hardware

    md["evidence"] = evidence
    return md


def _find_first(text: str, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in text:
            return candidate
    return None


def _extract_value(text: str, pattern: str, value_group: int = 1, unit_group: int = 3) -> str:
    match = re.search(pattern, text)
    if not match:
        return ""
    num = match.group(value_group)
    unit = match.group(unit_group)
    return f"{num} {unit}"


def _snippet(text: str, token: str, width: int = 120) -> str:
    idx = text.find(token)
    if idx < 0:
        return ""
    start = max(0, idx - width // 2)
    end = min(len(text), idx + width // 2)
    return " ".join(text[start:end].split())
