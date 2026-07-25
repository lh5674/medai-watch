from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from agent.monitoring_agent import MedicalAIWatchAgent


def analysis_json_bytes(result: Mapping[str, Any]) -> bytes:
    payload = MedicalAIWatchAgent.export_payload(result)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def analysis_csv_bytes(result: Mapping[str, Any]) -> bytes:
    original = result["original_prediction"]
    risk = result["risk"]
    quality = result["quality"]
    rows: list[dict[str, Any]] = []
    for item in result["perturbation_results"]:
        prediction = item["prediction"]
        comparison = item["comparison"]
        rows.append(
            {
                "record_id": result.get("record_id"),
                "original_class": original["class_name"],
                "original_confidence": original["confidence"],
                "quality_score": quality["quality_score"],
                "risk_score": risk["risk_score"],
                "decision": risk["decision"],
                "perturbation_name": item["test_name"],
                "changed_class": prediction["class_name"],
                "changed_confidence": prediction["confidence"],
                "confidence_change": comparison["confidence_change"],
                "class_unchanged": comparison["class_unchanged"],
                "gradcam_cosine_similarity": comparison[
                    "gradcam_cosine_similarity"
                ],
                "gradcam_ssim": comparison["gradcam_ssim"],
            }
        )
    dataframe = pd.DataFrame(rows)
    return dataframe.to_csv(index=False).encode("utf-8-sig")
