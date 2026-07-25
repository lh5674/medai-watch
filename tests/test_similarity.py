from __future__ import annotations

import json

import numpy as np
import pytest

from services.similarity import (
    SimilarityService,
    compare_predictions,
    gradcam_cosine_similarity,
    gradcam_ssim,
    normalize_heatmap,
)


def test_normalize_heatmap_scales_finite_values_to_unit_interval() -> None:
    heatmap = np.array([[2.0, 4.0], [6.0, 10.0]])

    normalized = normalize_heatmap(heatmap)

    assert normalized.dtype == np.float64
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    assert np.isfinite(normalized).all()


def test_extreme_finite_heatmap_values_remain_finite_and_bounded() -> None:
    first = np.array([[-1e308, 1e308], [1e308, -1e308]])
    second = np.array([[-1e308, 9e307], [1e308, -9e307]])

    normalized = normalize_heatmap(first)
    cosine = gradcam_cosine_similarity(first, second)
    ssim = gradcam_ssim(first, second)

    assert np.isfinite(normalized).all()
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    assert np.isfinite(cosine) and 0.0 <= cosine <= 1.0
    assert np.isfinite(ssim) and 0.0 <= ssim <= 1.0


@pytest.mark.parametrize(
    "heatmap",
    [
        np.array([]),
        np.empty((0, 4)),
        np.array([[0.0, np.nan]]),
        np.array([[0.0, np.inf]]),
    ],
)
def test_normalize_heatmap_rejects_empty_or_non_finite_input(
    heatmap: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        normalize_heatmap(heatmap)


def test_identical_nonconstant_heatmaps_have_maximum_similarity() -> None:
    heatmap = np.arange(64, dtype=float).reshape(8, 8)

    assert gradcam_cosine_similarity(heatmap, heatmap) == pytest.approx(1.0)
    assert gradcam_ssim(heatmap, heatmap) == pytest.approx(1.0)


def test_spatially_disjoint_heatmaps_have_bounded_similarity() -> None:
    left = np.zeros((8, 8), dtype=float)
    left[:, :4] = 1.0
    right = np.zeros((8, 8), dtype=float)
    right[:, 4:] = 1.0

    cosine = gradcam_cosine_similarity(left, right)
    ssim = gradcam_ssim(left, right)

    assert cosine == pytest.approx(0.0)
    assert 0.0 <= ssim <= 1.0


@pytest.mark.parametrize("metric", [gradcam_cosine_similarity, gradcam_ssim])
def test_equal_constant_heatmaps_are_identical(metric) -> None:
    first = np.full((8, 8), 3.0)
    second = np.full((8, 8), 3.0)

    assert metric(first, second) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [gradcam_cosine_similarity, gradcam_ssim])
def test_unequal_constant_heatmaps_are_not_identical(metric) -> None:
    first = np.full((8, 8), 3.0)
    second = np.full((8, 8), 4.0)

    assert metric(first, second) == pytest.approx(0.0)


@pytest.mark.parametrize("metric", [gradcam_cosine_similarity, gradcam_ssim])
def test_constant_and_nonconstant_heatmaps_have_zero_similarity(metric) -> None:
    constant = np.ones((8, 8), dtype=float)
    varying = np.arange(64, dtype=float).reshape(8, 8)

    assert metric(constant, varying) == pytest.approx(0.0)


@pytest.mark.parametrize("metric", [gradcam_cosine_similarity, gradcam_ssim])
def test_similarity_rejects_empty_and_mismatched_heatmaps(metric) -> None:
    with pytest.raises(ValueError, match="empty"):
        metric(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="shape"):
        metric(np.zeros((8, 8)), np.zeros((7, 8)))


def test_ssim_handles_small_heatmaps_with_a_finite_bounded_result() -> None:
    first = np.array([[0.0, 1.0], [1.0, 0.0]])
    second = np.array([[0.0, 0.8], [0.9, 0.1]])

    result = gradcam_ssim(first, second)

    assert isinstance(result, float)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("metric", [gradcam_cosine_similarity, gradcam_ssim])
def test_single_pixel_constant_heatmaps_are_handled_safely(metric) -> None:
    heatmap = np.array([[0.5]])

    assert metric(heatmap, heatmap) == pytest.approx(1.0)


def test_compare_predictions_returns_json_safe_prediction_and_cam_deltas() -> None:
    original = {"class_index": np.int64(1), "confidence": np.float32(0.9)}
    perturbed = {"class_index": np.int64(1), "confidence": np.float32(0.7)}
    original_heatmap = np.arange(64, dtype=float).reshape(8, 8)
    perturbed_heatmap = original_heatmap.copy()

    result = compare_predictions(
        original,
        perturbed,
        original_heatmap,
        perturbed_heatmap,
    )

    assert result["class_unchanged"] is True
    assert result["confidence_change"] == pytest.approx(0.2)
    assert result["cosine"] == pytest.approx(1.0)
    assert result["ssim"] == pytest.approx(1.0)
    assert result["gradcam_cosine_similarity"] == result["cosine"]
    assert result["gradcam_ssim"] == result["ssim"]
    json.dumps(result, allow_nan=False)


def test_compare_predictions_requires_class_and_bounded_confidence() -> None:
    heatmap = np.arange(64, dtype=float).reshape(8, 8)

    with pytest.raises(ValueError, match="class"):
        compare_predictions(
            {"confidence": 0.9},
            {"class_index": 1, "confidence": 0.8},
            heatmap,
            heatmap,
        )

    with pytest.raises(ValueError, match="confidence"):
        compare_predictions(
            {"class_index": 1, "confidence": 1.1},
            {"class_index": 1, "confidence": 0.8},
            heatmap,
            heatmap,
        )


def test_similarity_service_uses_the_same_comparison_contract() -> None:
    service = SimilarityService()
    heatmap = np.arange(64, dtype=float).reshape(8, 8)

    result = service.compare_predictions(
        {"class_index": 0, "confidence": 0.75},
        {"class_index": 1, "confidence": 0.25},
        heatmap,
        heatmap,
    )

    assert result["class_unchanged"] is False
    assert result["confidence_change"] == pytest.approx(0.5)
