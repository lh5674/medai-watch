from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def rgb_image() -> Image.Image:
    x = np.linspace(0, 255, 96, dtype=np.uint8)
    gradient = np.tile(x, (80, 1))
    array = np.stack(
        [gradient, np.flipud(gradient), np.roll(gradient, 12, axis=1)],
        axis=-1,
    )
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def grayscale_image() -> Image.Image:
    x = np.linspace(0, 255, 96, dtype=np.uint8)
    return Image.fromarray(np.tile(x, (80, 1)), mode="L")
