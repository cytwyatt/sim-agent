from __future__ import annotations

import re
from typing import Any

from sim_agent.extract.schema import normalize_core_details
from sim_agent.extract.system_build import extract_system_build_protocol
from sim_agent.llm import OpenAIClient
from sim_agent.types import PaperMetadata

KNOWN_ENGINES = [
    "gromacs",
    "lammps",
    "amber",
    "namd",
    "charmm",
    "cp2k",
    "vasp",
    "quantum espresso",
    "gaussian",
    "orca",
]

KNOWN_MODELS = [
    "charmm",
    "amber",
    "opls",
    "martini",
    "b3lyp",
    "pbe",
    "revpbe",
    "hartree-fock",
    "reaxff",
    "eam",
]


def extract_core_details(
    text: str,
    paper: PaperMetadata,
    simulation_type: str,
    custom_fields: list[str],
    llm_client: OpenAIClient | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    if llm_client and llm_client.enabled:
        llm = _extract_with_llm(text, paper, simulation_type, custom_fields, llm_client)
        if llm:
            details = normalize_core_details(llm.get("core_simulation_details"), simulation_type, custom_fields)
            summary = str(llm.get("summary") or "").strip()
            evidence = llm.get("evidence") if isinstance(llm.get("evidence"), list) else []
            details, evidence = _inject_system_build_details(
                text=text,
                simulation_type=simulation_type,
                details=details,
                evidence=evidence,
                llm_client=llm_client,
            )
            if summary:
                return details, summary, evidence

    details, summary, evidence = _heuristic_core_extraction(text, paper, simulation_type, custom_fields)
    details, evidence = _inject_system_build_details(
        text=text,
        simulation_type=simulation_type,
        details=details,
        evidence=evidence,
        llm_client=llm_client,
    )
    return details, summary, evidence


def _extract_with_llm(
    text: str,
    paper: PaperMetadata,
    simulation_type: str,
    custom_fields: list[str],
    llm_client: OpenAIClient,
) -> dict[str, Any] | None:
    source_text = text[:14000]
    custom_hint = ", ".join(custom_fields) if custom_fields else "(none)"
    system_prompt = (
        "You extract simulation-study details into structured JSON for scientific literature. "
        "Be concise and avoid hallucinations. Leave empty strings/lists if unknown. "
        "Return keys: core_simulation_details, summary, evidence."
    )
    user_prompt = (
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract}\n"
        f"Detected simulation_type: {simulation_type}\n"
        f"Custom fields requested: {custom_hint}\n\n"
        "Core schema keys: objective, simulation_type, software_or_engine(list), "
        "theory_or_force_model, system_description, system_build_protocol, system_build_steps(list), environment_conditions, "
        "sampling_or_propagation_setup, runtime_or_compute_budget, computed_properties(list), "
        "key_limitations, custom_fields(list of {name,value,unit,evidence}).\n"
        "Evidence should be a list of {field, snippet, confidence} for key fields only.\n"
        f"Paper text:\n{source_text}"
    )
    return llm_client.chat_json(system_prompt, user_prompt, temperature=0.0, max_tokens=1400)


def _heuristic_core_extraction(
    text: str,
    paper: PaperMetadata,
    simulation_type: str,
    custom_fields: list[str],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    lower = text.lower()
    engines = [engine for engine in KNOWN_ENGINES if engine in lower]
    models = [model for model in KNOWN_MODELS if model in lower]

    first_sentence = _first_sentence(text) or _first_sentence(paper.abstract)
    objective = first_sentence if first_sentence else f"Study related to {simulation_type} simulation."
    computed_props = _extract_computed_properties(lower)
    limitations = "Not explicitly stated in parsed text."

    details = normalize_core_details(
        {
            "objective": objective,
            "simulation_type": simulation_type,
            "software_or_engine": [e.upper() for e in engines] if engines else [],
            "theory_or_force_model": ", ".join(sorted(set(models))),
            "system_description": _find_section_like(text, ["system", "molecule", "material", "protein", "surface"]),
            "environment_conditions": _find_section_like(text, ["temperature", "pressure", "solvent", "vacuum"]),
            "sampling_or_propagation_setup": _find_section_like(text, ["ensemble", "sampling", "time step", "timestep"]),
            "runtime_or_compute_budget": _find_section_like(text, ["ns", "ps", "hours", "gpu", "cpu"]),
            "computed_properties": computed_props,
            "key_limitations": limitations,
        },
        simulation_type,
        custom_fields,
    )

    evidence = []
    if engines:
        snippet = _snippet_around(lower, engines[0])
        evidence.append({"field": "software_or_engine", "snippet": snippet, "confidence": 0.62})
    if models:
        snippet = _snippet_around(lower, models[0])
        evidence.append({"field": "theory_or_force_model", "snippet": snippet, "confidence": 0.58})
    if computed_props:
        evidence.append({"field": "computed_properties", "snippet": ", ".join(computed_props), "confidence": 0.52})

    summary = f"{paper.title}: {objective}"
    return details, summary, evidence


def _inject_system_build_details(
    text: str,
    simulation_type: str,
    details: dict[str, Any],
    evidence: list[dict[str, Any]],
    llm_client: OpenAIClient | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    build = extract_system_build_protocol(
        text=text,
        simulation_type=simulation_type,
        llm_client=llm_client,
    )
    if not details.get("system_build_protocol") and build.get("system_build_protocol"):
        details["system_build_protocol"] = build["system_build_protocol"]
    if (not details.get("system_build_steps")) and build.get("system_build_steps"):
        details["system_build_steps"] = build["system_build_steps"]

    for item in build.get("evidence", []):
        evidence.append(
            {
                "field": "system_build_protocol",
                "snippet": str(item.get("snippet") or ""),
                "confidence": float(item.get("confidence") or 0.5),
            }
        )
    return details, evidence


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return parts[0].strip() if parts and parts[0] else ""


def _find_section_like(text: str, keywords: list[str], width: int = 180) -> str:
    lower = text.lower()
    for kw in keywords:
        idx = lower.find(kw)
        if idx >= 0:
            start = max(0, idx - width // 2)
            end = min(len(text), idx + width)
            return " ".join(text[start:end].split())
    return ""


def _snippet_around(text: str, token: str, width: int = 140) -> str:
    idx = text.find(token)
    if idx < 0:
        return ""
    start = max(0, idx - width // 2)
    end = min(len(text), idx + width // 2)
    return " ".join(text[start:end].split())


def _extract_computed_properties(lower: str) -> list[str]:
    props = []
    mapping = {
        "free energy": "free energy",
        "binding affinity": "binding affinity",
        "diffusion": "diffusion coefficient",
        "rdf": "radial distribution function",
        "density": "density",
        "band gap": "band gap",
        "energy barrier": "energy barrier",
    }
    for token, value in mapping.items():
        if token in lower:
            props.append(value)
    return sorted(set(props))
