from __future__ import annotations

from typing import Any


def extract_qmmm_details_stub() -> dict[str, Any]:
    return {
        "status": "stub",
        "message": "QMMM deep profile is planned for v1.1; using core extraction only in v1.",
        "schema_preview": [
            "qm_method_basis_functional",
            "mm_force_field",
            "qm_mm_coupling",
            "partition_regions",
            "embedding_scheme",
            "charge_treatment",
            "time_step_and_coupling",
        ],
    }
