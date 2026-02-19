from __future__ import annotations

import re
from typing import Any

from sim_agent.llm import OpenAIClient

SYSTEM_BUILD_HINTS: list[tuple[str, list[str]]] = [
    ("initial_structure", ["pdb", "initial structure", "starting structure", "homology model", "crystal structure"]),
    ("builder_tool", ["charmm-gui", "tleap", "packmol", "vmd", "builder", "moltemplate", "avogadro"]),
    ("system_construction", ["constructed", "built", "generated", "created", "initialized", "initial configuration"]),
    ("polymer_architecture", ["chain length", "degree of polymerization", "polymer chains", "chain packing", "lamella"]),
    ("topology_parameterization", ["topology", "parameterized", "parametrized", "gaff", "cgenff", "fftk"]),
    ("solvation_environment", ["solvated", "solvation", "water box", "solvent box", "explicit solvent", "implicit solvent"]),
    ("ionization_charge", ["neutralized", "neutralised", "counterion", "na+", "cl-", "ionic strength", "salt concentration"]),
    ("cell_or_boundary", ["periodic boundary", "simulation cell", "supercell", "unit cell", "box size"]),
    ("minimization", ["energy minimization", "minimized", "steepest descent", "conjugate gradient"]),
    ("equilibration", ["equilibration", "equilibrated", "nvt", "npt", "annealing"]),
    ("thermal_protocol", ["melt", "melting", "cooling", "quench", "quenched", "crystallization", "nucleation"]),
    ("qmmm_partition", ["qm region", "mm region", "partition", "link atom", "embedding"]),
]


def extract_system_build_protocol(
    text: str,
    simulation_type: str,
    llm_client: OpenAIClient | None = None,
) -> dict[str, Any]:
    if llm_client and llm_client.enabled:
        llm_result = _extract_with_llm(text, simulation_type, llm_client)
        if llm_result:
            summary = str(llm_result.get("system_build_protocol") or "").strip()
            steps = llm_result.get("system_build_steps") if isinstance(llm_result.get("system_build_steps"), list) else []
            evidence = llm_result.get("evidence") if isinstance(llm_result.get("evidence"), list) else []
            if summary or steps:
                return {
                    "system_build_protocol": summary,
                    "system_build_steps": [str(item).strip() for item in steps if str(item).strip()],
                    "evidence": evidence,
                    "confidence": _coerce_confidence(llm_result.get("confidence", 0.7)),
                }
    return _heuristic_extract(text, simulation_type)


def _extract_with_llm(
    text: str,
    simulation_type: str,
    llm_client: OpenAIClient,
) -> dict[str, Any] | None:
    system_prompt = (
        "You extract how a simulation system is built from a scientific paper. "
        "Return JSON with keys: system_build_protocol (string), system_build_steps (list), "
        "evidence (list of {step,snippet,confidence}), confidence (0-1)."
    )
    user_prompt = (
        f"Simulation type: {simulation_type}\n"
        "Focus on system preparation details: initial structure/material setup, parameterization, "
        "solvation/environment, ion placement, cell setup, minimization/equilibration, and QM/MM partition if present.\n"
        f"Paper text:\n{text[:9000]}"
    )
    return llm_client.chat_json(system_prompt, user_prompt, temperature=0.0, max_tokens=900)


def _heuristic_extract(text: str, simulation_type: str) -> dict[str, Any]:
    sentences = _split_sentences(text)
    found_steps: list[str] = []
    evidence: list[dict[str, Any]] = []
    lower_sentences = [sentence.lower() for sentence in sentences]

    for step_name, patterns in SYSTEM_BUILD_HINTS:
        for idx, lower_sentence in enumerate(lower_sentences):
            if any(pattern in lower_sentence for pattern in patterns):
                sentence = _compact(sentences[idx])
                if sentence and sentence not in found_steps:
                    found_steps.append(sentence)
                    evidence.append(
                        {
                            "step": step_name,
                            "snippet": sentence[:260],
                            "confidence": 0.58,
                        }
                    )
                break

    if not found_steps:
        return {
            "system_build_protocol": "",
            "system_build_steps": [],
            "evidence": [],
            "confidence": 0.0,
        }

    prefix = f"{simulation_type} system setup"
    summary = f"{prefix}: " + "; ".join(found_steps[:4])
    return {
        "system_build_protocol": summary[:900],
        "system_build_steps": found_steps[:8],
        "evidence": evidence[:8],
        "confidence": min(0.9, 0.45 + 0.06 * len(found_steps)),
    }


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _compact(text: str) -> str:
    return " ".join(text.split())


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, min(1.0, parsed))
