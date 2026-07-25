from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Callable

import numpy as np
from PIL import Image


_SUPPORTED_MODES = {"L", "RGB"}
PERTURBATION_DISPLAY_NAMES = {
    "gaussian_blur": "高斯模糊",
    "gaussian_noise": "高斯噪声",
    "brightness": "亮度调整",
}


def _validate_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    if image.mode not in _SUPPORTED_MODES:
        raise ValueError("image mode must be RGB or L")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")


def _validate_unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _as_float_array(image: Image.Image) -> np.ndarray:
    _validate_image(image)
    return np.asarray(image, dtype=np.float32) / 255.0


def _to_image(array: np.ndarray, mode: str) -> Image.Image:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    result = Image.fromarray(pixels)
    if result.mode != mode:
        result = result.convert(mode)
    return result


def _gaussian_kernel(kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    sigma = 0.3 * (radius - 1) + 0.8
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _convolve_axis(
    array: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    radius = len(kernel) // 2
    padding = [(0, 0)] * array.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(array, padding, mode="edge")
    result = np.zeros_like(array, dtype=np.float32)

    for offset, weight in enumerate(kernel):
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(offset, offset + array.shape[axis])
        result += float(weight) * padded[tuple(selection)]

    return result


def add_gaussian_blur(
    image: Image.Image,
    kernel_size: int = 5,
) -> Image.Image:
    """Apply a Gaussian blur while preserving an RGB or grayscale image."""
    _validate_image(image)
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, Integral)
        or kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be an odd integer at least 3")

    array = _as_float_array(image)
    kernel = _gaussian_kernel(int(kernel_size))
    blurred = _convolve_axis(_convolve_axis(array, kernel, axis=1), kernel, axis=0)
    return _to_image(blurred, image.mode)


def add_gaussian_noise(
    image: Image.Image,
    sigma: float = 0.02,
    seed: int | None = 42,
) -> Image.Image:
    """Add deterministic normalized Gaussian noise for a supplied seed."""
    sigma_value = _validate_unit_interval(sigma, "sigma")
    array = _as_float_array(image)
    if sigma_value == 0.0:
        return _to_image(array, image.mode)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_value, size=array.shape).astype(np.float32)
    return _to_image(array + noise, image.mode)


def adjust_brightness(
    image: Image.Image,
    factor: float = 0.85,
) -> Image.Image:
    """Scale normalized pixel brightness with clipping at both endpoints."""
    if (
        isinstance(factor, bool)
        or not isinstance(factor, Real)
        or not np.isfinite(float(factor))
        or not 0.0 < float(factor) <= 3.0
    ):
        raise ValueError("factor must be finite and in (0, 3]")

    array = _as_float_array(image)
    return _to_image(array * float(factor), image.mode)


def apply_perturbation(
    image: Image.Image,
    name: str,
    **parameters: Any,
) -> Image.Image:
    """Dispatch a named perturbation using its validated keyword parameters."""
    if not isinstance(name, str):
        raise ValueError("Unsupported perturbation: name must be a string")

    functions: dict[str, Callable[..., Image.Image]] = {
        "blur": add_gaussian_blur,
        "gaussian_blur": add_gaussian_blur,
        "noise": add_gaussian_noise,
        "gaussian_noise": add_gaussian_noise,
        "brightness": adjust_brightness,
    }
    normalized_name = name.strip().lower()
    try:
        function = functions[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported perturbation: {name}") from exc
    return function(image, **parameters)
