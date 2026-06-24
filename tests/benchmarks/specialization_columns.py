"""Lightweight specialization column extraction for benchmark persistence.

This stays benchmark-local so SQLite/report tooling does not need to import the
full `mojogp` package just to read specialization metadata.
"""

from __future__ import annotations

from typing import Any


def extract_specialization_columns(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_case_id": config.get("base_case_id"),
        "specialization_key": config.get("specialization_key"),
        "specialization_family": config.get("specialization_family"),
        "specialization_mode": config.get("specialization_mode"),
        "specialization_source": config.get("specialization_source"),
        "specialization_descriptor_json": config.get("specialization_descriptor", {}),
        "specialization_config_json": config.get("specialization_config", {}),
    }
