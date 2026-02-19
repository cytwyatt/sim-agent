from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from sim_agent.llm import OpenAIClient
from sim_agent.types import SimulationType

KEYWORDS: dict[str, list[str]] = {
    SimulationType.QMMM.value: ["qmmm", "qm/mm", "quantum mechanics/molecular mechanics", "oniom"],
    SimulationType.MD.value: [
        "molecular dynamics",
        "md simulation",
        "gromacs",
        "lammps",
        "amber",
        "namd",
        "charmm",
        "trajectory",
        "force field",
    ],
    SimulationType.QM.value: [
        "density functional theory",
        "dft",
        "ab initio",
        "hartree-fock",
        "gaussian",
        "orca",
        "quantum chemistry",
        "basis set",
    ],
    SimulationType.MC.value: ["monte carlo", "metropolis", "kinetic monte carlo", "markov chain monte carlo"],
    SimulationType.CG.value: ["coarse-grained", "coarse grained", "martini", "bead-spring", "united atom"],
}


def classify_simulation_type(text: str, llm_client: OpenAIClient | None = None) -> tuple[str, float]:
    type_guess, confidence = _rule_based_classify(text)
    if llm_client and llm_client.enabled:
        llm_result = _llm_refine(text, llm_client)
        if llm_result:
            llm_type = llm_result.get("simulation_type")
            llm_conf = _coerce_confidence(llm_result.get("confidence"))
            if llm_type in {t.value for t in SimulationType}:
                # Keep the stronger of rule vs LLM confidence.
                if llm_conf >= confidence:
                    return llm_type, llm_conf
    return type_guess, confidence


def _rule_based_classify(text: str) -> tuple[str, float]:
    lower = text.lower()
    scores: dict[str, int] = defaultdict(int)
    for sim_type, patterns in KEYWORDS.items():
        for pattern in patterns:
            scores[sim_type] += len(re.findall(re.escape(pattern), lower))

    if not scores or max(scores.values()) == 0:
        return SimulationType.OTHER.value, 0.35

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    total = sum(scores.values()) or 1
    confidence = min(0.95, 0.4 + best_score / total * 0.55)
    return best_type, round(confidence, 3)


def _llm_refine(text: str, llm_client: OpenAIClient) -> dict[str, Any] | None:
    prompt_text = text[:6000]
    system = (
        "You classify scientific papers by simulation type. "
        "Allowed simulation_type values: MD, QM, QMMM, MC, CG, Other/Unknown. "
        "Return JSON with keys: simulation_type, confidence, reason."
    )
    user = f"Paper text:\n{prompt_text}"
    return llm_client.chat_json(system, user, temperature=0.0, max_tokens=250)


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))
