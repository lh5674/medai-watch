from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from config.settings import QualitySettings, get_settings


ImageSource = (
    Image.Image
    | str
    | os.PathLike[str]
    | bytes
    | bytearray
    | memoryview
    | BinaryIO
)


def _loaded_copy(image: Image.Image) -> Image.Image:
    image.load()
    return image.copy()


def load_image(source: ImageSource) -> Image.Image:
    """Load and detach a PIL image from common local input forms."""
    if isinstance(source, Image.Image):
        try:
            return _loaded_copy(source)
        except (OSError, ValueError) as exc:
            raise ValueError("Unable to decode image input") from exc

    try:
        if isinstance(source, (str, os.PathLike)):
            with Image.open(Path(source)) as image:
                return _loaded_copy(image)

        if isinstance(source, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(source))) as image:
                return _loaded_copy(image)

        if hasattr(source, "read"):
            position: int | None = None
            try:
                position = source.tell()
            except (AttributeError, OSError):
                pass
            try:
                with Image.open(source) as image:
                    return _loaded_copy(image)
            finally:
                if position is not None:
                    try:
                        source.seek(position)
                    except (AttributeError, OSError):
                        pass
    except (
        AttributeError,
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        TypeError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise ValueError("Unable to decode image input") from exc

    raise TypeError(
        "source must be a PIL image, path, bytes, or binary file-like object"
    )


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _linear_score(value: float, low: float, good: float) -> float:
    return _bounded((value - low) / (good - low))


def _brightness_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return _bounded(value / low) if low > 0.0 else 1.0
    return _bounded((1.0 - value) / (1.0 - high)) if high < 1.0 else 1.0


def _laplacian_variance(grayscale: np.ndarray) -> float:
    padded = np.pad(grayscale, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(np.var(laplacian, dtype=np.float64))


class ImageQualityService:
    """Compute deterministic technical image-quality metrics and scores."""

    def __init__(self, settings: QualitySettings | None = None) -> None:
        self.settings = settings or get_settings().quality
        self._validate_settings()

    def _validate_settings(self) -> None:
        settings = self.settings
        dimensions = (
            settings.min_width,
            settings.min_height,
            settings.max_dimension,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in dimensions
        ):
            raise ValueError("quality dimension thresholds must be positive integers")
        if settings.max_dimension < max(settings.min_width, settings.min_height):
            raise ValueError("max_dimension must cover the minimum dimensions")

        finite_thresholds = (
            settings.blur_variance_low,
            settings.blur_variance_good,
            settings.brightness_low,
            settings.brightness_high,
            settings.contrast_low,
            settings.contrast_good,
            settings.extreme_fraction_max,
        )
        if not all(np.isfinite(float(value)) for value in finite_thresholds):
            raise ValueError("quality thresholds must be finite")
        if not 0.0 <= settings.blur_variance_low < settings.blur_variance_good:
            raise ValueError("blur thresholds must satisfy 0 <= low < good")
        if not 0.0 <= settings.brightness_low < settings.brightness_high <= 1.0:
            raise ValueError("brightness thresholds must be within [0, 1]")
        if not 0.0 <= settings.contrast_low < settings.contrast_good <= 1.0:
            raise ValueError("contrast thresholds must be within [0, 1]")
        if not 0.0 < settings.extreme_fraction_max <= 1.0:
            raise ValueError("extreme_fraction_max must be in (0, 1]")

        weights = tuple(settings.weights.values())
        if (
            not weights
            or not all(np.isfinite(float(weight)) and weight >= 0 for weight in weights)
            or sum(weights) <= 0
        ):
            raise ValueError("quality weights must be finite, non-negative, and non-zero")

    @staticmethod
    def load_image(source: ImageSource) -> Image.Image:
        return load_image(source)

    def assess(self, source: ImageSource) -> dict[str, Any]:
        image = load_image(source)
        raw_pixels = np.asarray(image)
        if np.issubdtype(raw_pixels.dtype, np.number) and not np.isfinite(
            raw_pixels
        ).all():
            raise ValueError("image pixel values must be finite")

        grayscale = np.asarray(image.convert("L"), dtype=np.float32)
        if grayscale.size == 0:
            raise ValueError("image dimensions must be positive")

        width, height = image.size
        settings = self.settings
        laplacian_variance = _laplacian_variance(grayscale)
        mean_brightness = float(np.mean(grayscale, dtype=np.float64) / 255.0)
        contrast_std = float(np.std(grayscale, dtype=np.float64) / 255.0)
        normalized = grayscale / 255.0
        extreme_pixel_ratio = float(
            np.mean((normalized <= 0.01) | (normalized >= 0.99))
        )

        minimum_dimension_score = min(
            1.0,
            width / settings.min_width,
            height / settings.min_height,
        )
        maximum_dimension_score = min(
            1.0,
            settings.max_dimension / max(width, height),
        )
        dimension_score = _bounded(
            min(minimum_dimension_score, maximum_dimension_score)
        )
        blur_score = _linear_score(
            laplacian_variance,
            settings.blur_variance_low,
            settings.blur_variance_good,
        )
        brightness_score = _brightness_score(
            mean_brightness,
            settings.brightness_low,
            settings.brightness_high,
        )
        contrast_score = _linear_score(
            contrast_std,
            settings.contrast_low,
            settings.contrast_good,
        )
        extreme_score = _bounded(
            1.0 - extreme_pixel_ratio / settings.extreme_fraction_max
        )

        sub_scores = {
            "blur": blur_score,
            "brightness": brightness_score,
            "contrast": contrast_score,
            "dimensions": dimension_score,
            "extreme_pixels": extreme_score,
        }
        weighted_score = sum(
            float(weight) * sub_scores[name]
            for name, weight in settings.weights.items()
        ) / sum(float(weight) for weight in settings.weights.values())
        quality_score = float(np.clip(100.0 * weighted_score, 0.0, 100.0))

        dimensions_valid = (
            width >= settings.min_width
            and height >= settings.min_height
            and max(width, height) <= settings.max_dimension
        )
        issues: list[str] = []
        if width < settings.min_width or height < settings.min_height:
            issues.append("图像尺寸过小")
        if max(width, height) > settings.max_dimension:
            issues.append("图像尺寸过大")
        if laplacian_variance < settings.blur_variance_low:
            issues.append("图像可能存在模糊")
        if mean_brightness < settings.brightness_low:
            issues.append("图像亮度过低")
        elif mean_brightness > settings.brightness_high:
            issues.append("图像亮度过高")
        if contrast_std < settings.contrast_low:
            issues.append("图像对比度较低")
        if extreme_pixel_ratio > settings.extreme_fraction_max:
            issues.append("图像包含过多极端像素值")

        return {
            "valid": dimensions_valid,
            "mode": image.mode,
            "width": width,
            "height": height,
            "dimensions": {"width": width, "height": height},
            "laplacian_variance": laplacian_variance,
            "mean_brightness": mean_brightness,
            "contrast_std": contrast_std,
            "extreme_pixel_ratio": extreme_pixel_ratio,
            "blur_score": blur_score,
            "brightness_score": brightness_score,
            "contrast_score": contrast_score,
            "dimension_score": dimension_score,
            "extreme_score": extreme_score,
            "sub_scores": sub_scores,
            "quality_score": quality_score,
            "issues": issues,
        }

    def analyze(self, source: ImageSource) -> dict[str, Any]:
        return self.assess(source)


def assess_image_quality(
    source: ImageSource,
    settings: QualitySettings | None = None,
) -> dict[str, Any]:
    return ImageQualityService(settings).assess(source)
