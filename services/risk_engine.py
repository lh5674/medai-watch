from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from config.settings import RISK_DISCLAIMER, RiskSettings


_DECISIONS = {
    "pass": ("通过", "✅"),
    "caution": ("谨慎使用", "⚠️"),
    "review": ("建议人工复核", "🔴"),
}


def _bounded_number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def _class_unchanged(value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("class_unchanged must be a boolean")
    return bool(value)


def _similarity_value(
    comparison: Mapping[str, Any],
    primary_key: str,
    alias_key: str,
) -> float:
    if primary_key in comparison:
        value = comparison[primary_key]
    elif alias_key in comparison:
        value = comparison[alias_key]
    else:
        raise ValueError(
            f"perturbation result must include {primary_key} or {alias_key}"
        )
    return _bounded_number(
        value,
        name=primary_key,
        minimum=0.0,
        maximum=1.0,
    )


def _validated_perturbations(
    perturbations: Iterable[Mapping[str, Any]],
) -> list[tuple[bool, float, float]]:
    if isinstance(perturbations, (Mapping, str, bytes)):
        raise ValueError("perturbations must be a collection of result mappings")
    try:
        items = list(perturbations)
    except TypeError as exc:
        raise ValueError("perturbations must be iterable") from exc
    if not items:
        raise ValueError("at least one perturbation result is required")

    validated: list[tuple[bool, float, float]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("each perturbation result must be a mapping")
        if "class_unchanged" not in item:
            raise ValueError(
                "perturbation result must include class_unchanged"
            )
        unchanged = _class_unchanged(item["class_unchanged"])
        cosine = _similarity_value(
            item,
            "cosine",
            "gradcam_cosine_similarity",
        )
        ssim = _similarity_value(item, "ssim", "gradcam_ssim")
        validated.append((unchanged, cosine, ssim))
    return validated


def calculate_risk(
    original_confidence: float,
    perturbations: Iterable[Mapping[str, Any]],
    quality_score: float,
    settings: RiskSettings | None = None,
) -> dict[str, Any]:
    """Calculate the configured transparent risk score on a 0-100 scale."""

    risk_settings = settings or RiskSettings()
    confidence = _bounded_number(
        original_confidence,
        name="original_confidence",
        minimum=0.0,
        maximum=1.0,
    )
    quality = _bounded_number(
        quality_score,
        name="quality_score",
        minimum=0.0,
        maximum=100.0,
    )
    comparisons = _validated_perturbations(perturbations)

    count = len(comparisons)
    changed_count = sum(not unchanged for unchanged, _, _ in comparisons)
    explanation_similarities = [
        (cosine + ssim) / 2.0 for _, cosine, ssim in comparisons
    ]
    mean_explanation_similarity = (
        sum(explanation_similarities) / count
    )

    components = {
        "confidence_risk": float(1.0 - confidence),
        "prediction_instability": float(changed_count / count),
        "explanation_risk": float(1.0 - mean_explanation_similarity),
        "quality_risk": float(1.0 - quality / 100.0),
    }
    weights = {
        name: float(weight)
        for name, weight in risk_settings.weights.items()
    }
    if any(not np.isfinite(weight) for weight in weights.values()):
        raise ValueError("risk settings weights must be finite")
    if any(weight < 0.0 for weight in weights.values()) or not np.isclose(
        sum(weights.values()),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("risk settings weights must be non-negative and sum to 1")
    contributions = {
        name: float(components[name] * weights[name] * 100.0)
        for name in components
    }
    score = float(
        np.clip(round(sum(contributions.values()), 12), 0.0, 100.0)
    )

    if score < risk_settings.pass_threshold:
        decision, icon = _DECISIONS["pass"]
    elif score < risk_settings.caution_threshold:
        decision, icon = _DECISIONS["caution"]
    else:
        decision, icon = _DECISIONS["review"]

    return {
        "score": score,
        "risk_score": score,
        "decision": decision,
        "decision_text": decision,
        "icon": icon,
        "decision_icon": icon,
        "decision_display": f"{icon} {decision}",
        "components": components,
        "contributions": contributions,
        "weights": weights,
        "thresholds": {
            "pass": float(risk_settings.pass_threshold),
            "caution": float(risk_settings.caution_threshold),
        },
        "disclaimer": RISK_DISCLAIMER,
    }


class RiskEngine:
    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings()

    def evaluate(
        self,
        original_confidence: float,
        perturbations: Iterable[Mapping[str, Any]],
        quality_score: float,
    ) -> dict[str, Any]:
        return calculate_risk(
            original_confidence=original_confidence,
            perturbations=perturbations,
            quality_score=quality_score,
            settings=self.settings,
        )

    calculate = evaluate
