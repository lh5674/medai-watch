from __future__ import annotations

import json

import numpy as np
import pytest

from config.settings import RISK_DISCLAIMER, RiskSettings
from services.risk_engine import RiskEngine, calculate_risk


def comparison(
    *,
    unchanged: bool,
    cosine: float,
    ssim: float,
) -> dict[str, bool | float]:
    return {
        "class_unchanged": unchanged,
        "cosine": cosine,
        "ssim": ssim,
    }


def test_risk_score_uses_exact_configured_formula_and_exposes_contributions() -> None:
    result = calculate_risk(
        original_confidence=0.8,
        perturbations=[
            comparison(unchanged=True, cosine=0.8, ssim=0.6),
            comparison(unchanged=False, cosine=0.4, ssim=0.2),
        ],
        quality_score=80.0,
    )

    assert result["components"] == pytest.approx(
        {
            "confidence_risk": 0.2,
            "prediction_instability": 0.5,
            "explanation_risk": 0.5,
            "quality_risk": 0.2,
        }
    )
    assert result["contributions"] == pytest.approx(
        {
            "confidence_risk": 5.0,
            "prediction_instability": 17.5,
            "explanation_risk": 12.5,
            "quality_risk": 3.0,
        }
    )
    assert result["score"] == pytest.approx(38.0)
    assert sum(result["contributions"].values()) == pytest.approx(result["score"])


def test_zero_risk_inputs_produce_zero_score_and_pass_decision() -> None:
    result = calculate_risk(
        original_confidence=1.0,
        perturbations=[comparison(unchanged=True, cosine=1.0, ssim=1.0)],
        quality_score=100.0,
    )

    assert result["score"] == pytest.approx(0.0)
    assert result["decision"] == "通过"
    assert result["icon"] == "✅"


def test_maximum_risk_inputs_produce_bounded_score_and_review_decision() -> None:
    result = calculate_risk(
        original_confidence=0.0,
        perturbations=[comparison(unchanged=False, cosine=0.0, ssim=0.0)],
        quality_score=0.0,
    )

    assert result["score"] == pytest.approx(100.0)
    assert result["decision"] == "建议人工复核"
    assert result["icon"] == "🔴"


@pytest.mark.parametrize(
    ("confidence", "perturbation", "quality", "score", "decision", "icon"),
    [
        (
            0.8,
            comparison(unchanged=True, cosine=0.0, ssim=0.0),
            100.0,
            30.0,
            "谨慎使用",
            "⚠️",
        ),
        (
            0.0,
            comparison(unchanged=False, cosine=1.0, ssim=1.0),
            100.0,
            60.0,
            "建议人工复核",
            "🔴",
        ),
    ],
)
def test_exact_thresholds_enter_the_higher_risk_decision(
    confidence: float,
    perturbation: dict[str, bool | float],
    quality: float,
    score: float,
    decision: str,
    icon: str,
) -> None:
    result = calculate_risk(
        original_confidence=confidence,
        perturbations=[perturbation],
        quality_score=quality,
    )

    assert result["score"] == pytest.approx(score)
    assert result["decision"] == decision
    assert result["icon"] == icon


def test_custom_config_weights_and_thresholds_are_honored() -> None:
    settings = RiskSettings(
        confidence_weight=1.0,
        instability_weight=0.0,
        explanation_weight=0.0,
        quality_weight=0.0,
        pass_threshold=10.0,
        caution_threshold=20.0,
    )

    result = calculate_risk(
        original_confidence=0.85,
        perturbations=[comparison(unchanged=True, cosine=1.0, ssim=1.0)],
        quality_score=100.0,
        settings=settings,
    )

    assert result["score"] == pytest.approx(15.0)
    assert result["decision"] == "谨慎使用"


def test_non_finite_config_weight_is_rejected() -> None:
    settings = RiskSettings(confidence_weight=np.nan)

    with pytest.raises(ValueError, match="weight"):
        calculate_risk(
            original_confidence=0.8,
            perturbations=[comparison(unchanged=True, cosine=1.0, ssim=1.0)],
            quality_score=100.0,
            settings=settings,
        )


def test_at_least_one_perturbation_is_required() -> None:
    with pytest.raises(ValueError, match="perturbation"):
        calculate_risk(
            original_confidence=0.8,
            perturbations=[],
            quality_score=80.0,
        )


@pytest.mark.parametrize(
    ("confidence", "quality"),
    [
        (-0.1, 50.0),
        (1.1, 50.0),
        (np.nan, 50.0),
        (0.5, -0.1),
        (0.5, 100.1),
        (0.5, np.inf),
    ],
)
def test_invalid_confidence_and_quality_inputs_are_rejected(
    confidence: float,
    quality: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_risk(
            original_confidence=confidence,
            perturbations=[comparison(unchanged=True, cosine=1.0, ssim=1.0)],
            quality_score=quality,
        )


@pytest.mark.parametrize(
    "invalid_comparison",
    [
        {"class_unchanged": True, "cosine": 1.0},
        {"class_unchanged": "yes", "cosine": 1.0, "ssim": 1.0},
        {"class_unchanged": True, "cosine": -0.1, "ssim": 1.0},
        {"class_unchanged": True, "cosine": 1.0, "ssim": np.nan},
    ],
)
def test_invalid_perturbation_results_are_rejected(
    invalid_comparison: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        calculate_risk(
            original_confidence=0.8,
            perturbations=[invalid_comparison],
            quality_score=80.0,
        )


def test_gradcam_alias_fields_are_accepted_by_risk_engine() -> None:
    result = calculate_risk(
        original_confidence=0.8,
        perturbations=[
            {
                "class_unchanged": True,
                "gradcam_cosine_similarity": 0.8,
                "gradcam_ssim": 0.6,
            }
        ],
        quality_score=80.0,
    )

    assert result["components"]["explanation_risk"] == pytest.approx(0.3)


def test_result_is_json_safe_and_always_contains_required_disclaimer() -> None:
    result = calculate_risk(
        original_confidence=np.float32(0.8),
        perturbations=[
            {
                "class_unchanged": np.bool_(True),
                "cosine": np.float32(0.8),
                "ssim": np.float32(0.6),
            }
        ],
        quality_score=np.float32(80.0),
    )

    assert result["disclaimer"] == RISK_DISCLAIMER
    assert result["risk_score"] == result["score"]
    assert result["decision_text"] == result["decision"]
    assert result["decision_icon"] == result["icon"]
    json.dumps(result, allow_nan=False)


def test_risk_engine_class_uses_supplied_settings() -> None:
    settings = RiskSettings(pass_threshold=5.0, caution_threshold=10.0)
    engine = RiskEngine(settings)

    result = engine.evaluate(
        original_confidence=0.8,
        perturbations=[comparison(unchanged=True, cosine=1.0, ssim=1.0)],
        quality_score=100.0,
    )

    assert result["score"] == pytest.approx(5.0)
    assert result["decision"] == "谨慎使用"
