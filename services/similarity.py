from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
except ModuleNotFoundError:  # Python 3.14 test environments may lack a wheel.
    _skimage_ssim = None


_CLASS_KEYS = ("class_index", "predicted_class", "class_name", "label", "class_id")
_CONFIDENCE_KEYS = ("confidence", "predicted_confidence", "score")


def _as_heatmap(heatmap: Any) -> np.ndarray:
    try:
        array = np.asarray(heatmap, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("heatmap must contain numeric values") from exc

    if array.size == 0:
        raise ValueError("heatmap must not be empty")

    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError("heatmap must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("heatmap must contain only finite values")
    return array


def _heatmap_pair(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
    first_array = _as_heatmap(first)
    second_array = _as_heatmap(second)
    if first_array.shape != second_array.shape:
        raise ValueError("heatmaps must have the same shape")
    return first_array, second_array


def _is_constant(heatmap: np.ndarray) -> bool:
    return bool(heatmap.max() == heatmap.min())


def _constant_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    first_is_constant = _is_constant(first)
    second_is_constant = _is_constant(second)
    if not first_is_constant and not second_is_constant:
        return None
    if first_is_constant and second_is_constant and np.array_equal(first, second):
        return 1.0
    return 0.0


def normalize_heatmap(heatmap: Any) -> np.ndarray:
    """Return a finite float64 heatmap scaled to [0, 1]."""

    array = _as_heatmap(heatmap)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.zeros_like(array, dtype=np.float64)

    scale = max(abs(minimum), abs(maximum))
    scaled = array / scale
    scaled_minimum = float(scaled.min())
    scaled_range = float(scaled.max()) - scaled_minimum
    return np.asarray(
        (scaled - scaled_minimum) / scaled_range,
        dtype=np.float64,
    )


def gradcam_cosine_similarity(first: Any, second: Any) -> float:
    """Compare equally shaped Grad-CAM heatmaps using normalized cosine similarity."""

    first_array, second_array = _heatmap_pair(first, second)
    constant_result = _constant_similarity(first_array, second_array)
    if constant_result is not None:
        return constant_result
    if np.array_equal(first_array, second_array):
        return 1.0

    first_vector = normalize_heatmap(first_array).ravel()
    second_vector = normalize_heatmap(second_array).ravel()
    denominator = float(
        np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    )
    if denominator == 0.0:
        return 0.0
    similarity = float(np.dot(first_vector, second_vector) / denominator)
    return float(np.clip(similarity, 0.0, 1.0))


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Small-array/dependency fallback using the standard SSIM formula."""

    first_mean = float(first.mean())
    second_mean = float(second.mean())
    first_centered = first - first_mean
    second_centered = second - second_mean
    first_variance = float(np.mean(first_centered * first_centered))
    second_variance = float(np.mean(second_centered * second_centered))
    covariance = float(np.mean(first_centered * second_centered))
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (
        (first_mean**2 + second_mean**2 + c1)
        * (first_variance + second_variance + c2)
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return (
        (2.0 * first_mean * second_mean + c1)
        * (2.0 * covariance + c2)
        / denominator
    )


def gradcam_ssim(first: Any, second: Any) -> float:
    """Compare Grad-CAM heatmaps with SSIM mapped from [-1, 1] to [0, 1]."""

    first_array, second_array = _heatmap_pair(first, second)
    constant_result = _constant_similarity(first_array, second_array)
    if constant_result is not None:
        return constant_result
    if np.array_equal(first_array, second_array):
        return 1.0

    first_normalized = normalize_heatmap(first_array)
    second_normalized = normalize_heatmap(second_array)
    shortest_side = min(first_normalized.shape)

    if _skimage_ssim is not None and shortest_side >= 3:
        win_size = min(7, shortest_side)
        if win_size % 2 == 0:
            win_size -= 1
        raw_similarity = float(
            _skimage_ssim(
                first_normalized,
                second_normalized,
                data_range=1.0,
                win_size=win_size,
            )
        )
    else:
        raw_similarity = float(
            _global_ssim(first_normalized, second_normalized)
        )

    similarity = (raw_similarity + 1.0) / 2.0
    return float(np.clip(similarity, 0.0, 1.0))


def _prediction_value(
    prediction: Mapping[str, Any],
    keys: tuple[str, ...],
    value_name: str,
) -> Any:
    for key in keys:
        if key in prediction:
            value = prediction[key]
            if isinstance(value, np.generic):
                value = value.item()
            return value
    raise ValueError(f"prediction must include a {value_name} value")


def _prediction_confidence(prediction: Mapping[str, Any]) -> float:
    value = _prediction_value(prediction, _CONFIDENCE_KEYS, "confidence")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("prediction confidence must be numeric")
    confidence = float(value)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("prediction confidence must be between 0 and 1")
    return confidence


def compare_predictions(
    original_prediction: Mapping[str, Any],
    perturbed_prediction: Mapping[str, Any],
    original_heatmap: Any,
    perturbed_heatmap: Any,
) -> dict[str, bool | float]:
    """Compare predictions and heatmaps using JSON-safe scalar results."""

    if not isinstance(original_prediction, Mapping) or not isinstance(
        perturbed_prediction, Mapping
    ):
        raise ValueError("predictions must be mappings")

    original_class = _prediction_value(
        original_prediction,
        _CLASS_KEYS,
        "class",
    )
    perturbed_class = _prediction_value(
        perturbed_prediction,
        _CLASS_KEYS,
        "class",
    )
    original_confidence = _prediction_confidence(original_prediction)
    perturbed_confidence = _prediction_confidence(perturbed_prediction)
    cosine = gradcam_cosine_similarity(original_heatmap, perturbed_heatmap)
    ssim = gradcam_ssim(original_heatmap, perturbed_heatmap)

    return {
        "class_unchanged": bool(original_class == perturbed_class),
        "confidence_change": float(
            abs(perturbed_confidence - original_confidence)
        ),
        "cosine": float(cosine),
        "ssim": float(ssim),
        "gradcam_cosine_similarity": float(cosine),
        "gradcam_ssim": float(ssim),
    }


def compare_results(
    original_prediction: Mapping[str, Any],
    perturbed_prediction: Mapping[str, Any],
    original_heatmap: Any,
    perturbed_heatmap: Any,
) -> dict[str, bool | float]:
    """Backward-compatible name used by the implementation plan."""

    return compare_predictions(
        original_prediction,
        perturbed_prediction,
        original_heatmap,
        perturbed_heatmap,
    )


cosine_similarity = gradcam_cosine_similarity
ssim_similarity = gradcam_ssim


class SimilarityService:
    normalize_heatmap = staticmethod(normalize_heatmap)
    gradcam_cosine_similarity = staticmethod(gradcam_cosine_similarity)
    gradcam_ssim = staticmethod(gradcam_ssim)
    cosine_similarity = staticmethod(gradcam_cosine_similarity)
    ssim_similarity = staticmethod(gradcam_ssim)
    compare_predictions = staticmethod(compare_predictions)
    compare_results = staticmethod(compare_results)
