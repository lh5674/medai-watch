from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from config.settings import DEMO_DISCLAIMER


IMAGE_SIZE = 224
DEFAULT_TESTS = ["gaussian_blur", "gaussian_noise", "brightness"]


def _default_parameters() -> dict[str, dict[str, float | int]]:
    return {
        "gaussian_blur": {"kernel_size": 5},
        "gaussian_noise": {"sigma": 0.02, "seed": 42},
        "brightness": {"factor": 0.85},
    }


def _coordinates() -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float32)
    return np.meshgrid(axis, axis)


def _to_rgb(grayscale: np.ndarray) -> Image.Image:
    clipped = np.clip(grayscale, 0.0, 1.0)
    rgb = np.stack(
        [
            clipped,
            np.clip(clipped * 0.95 + 0.025, 0.0, 1.0),
            np.clip(clipped * 0.90 + 0.05, 0.0, 1.0),
        ],
        axis=-1,
    )
    return Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")


def _stable_pattern() -> Image.Image:
    x, y = _coordinates()
    radius = np.sqrt(x**2 + y**2)
    center = np.exp(-4.0 * radius**2)
    ring = np.exp(-80.0 * (radius - 0.58) ** 2)
    return _to_rgb(0.16 + 0.66 * center + 0.15 * ring)


def _confidence_shift_pattern() -> Image.Image:
    x, y = _coordinates()
    broad_region = np.exp(-2.2 * ((x + 0.12) ** 2 + (y - 0.08) ** 2))
    diagonal = np.exp(-100.0 * (y - 0.45 * x) ** 2)
    return _to_rgb(0.31 + 0.25 * broad_region + 0.08 * diagonal)


def _prediction_change_pattern() -> Image.Image:
    x, y = _coordinates()
    checker = ((np.floor((x + 1.0) * 7) + np.floor((y + 1.0) * 7)) % 2) * 0.42
    sharp_disk = ((x - 0.18) ** 2 + (y + 0.15) ** 2 < 0.20).astype(np.float32)
    return _to_rgb(0.12 + checker + 0.34 * sharp_disk)


def get_demo_cases() -> list[dict[str, Any]]:
    """Return deterministic synthetic images without prefabricated model output."""
    return [
        {
            "id": "stable_low_risk",
            "name": "稳定低风险演示",
            "description": "用于演示相对平滑、结构集中的合成输入。",
            "image": _stable_pattern(),
            "simulated": True,
            "disclaimer": DEMO_DISCLAIMER,
            "recommended_tests": list(DEFAULT_TESTS),
            "recommended_parameters": _default_parameters(),
        },
        {
            "id": "confidence_shift_medium_risk",
            "name": "置信度变化中风险演示",
            "description": "用于演示低对比度合成输入在噪声扰动下的置信度与解释变化。",
            "image": _confidence_shift_pattern(),
            "simulated": True,
            "disclaimer": DEMO_DISCLAIMER,
            "recommended_tests": ["gaussian_noise"],
            "recommended_parameters": {
                "gaussian_noise": {"sigma": 0.25, "seed": 42}
            },
        },
        {
            "id": "prediction_change_high_risk",
            "name": "预测变化高风险演示",
            "description": "用于演示包含高频边缘结构的合成输入。",
            "image": _prediction_change_pattern(),
            "simulated": True,
            "disclaimer": DEMO_DISCLAIMER,
            "recommended_tests": ["brightness"],
            "recommended_parameters": {"brightness": {"factor": 0.20}},
        },
    ]
