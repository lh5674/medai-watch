from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from services.gradcam_service import (
    GradCAMError,
    GradCAMService,
    find_last_convolution,
    resolve_layer,
)
from services.predictor import PredictorService


def test_auto_discovers_last_convolution() -> None:
    predictor = PredictorService()

    name, layer = find_last_convolution(predictor.model)

    assert isinstance(layer, nn.Conv2d)
    assert name


def test_resolves_explicit_dotted_layer() -> None:
    predictor = PredictorService()

    layer = resolve_layer(predictor.model, "features.2")

    assert isinstance(layer, nn.Conv2d)


def test_gradcam_outputs_expected_shapes_and_bounds(
    rgb_image: Image.Image,
) -> None:
    predictor = PredictorService()
    service = GradCAMService(predictor)
    target_class = predictor.predict(rgb_image)["class_index"]

    result = service.generate(rgb_image, target_class=target_class)

    assert result["raw_heatmap"].shape == (rgb_image.height, rgb_image.width)
    assert result["normalized_heatmap"].shape == (
        rgb_image.height,
        rgb_image.width,
    )
    assert result["overlay"].shape == (
        rgb_image.height,
        rgb_image.width,
        3,
    )
    assert result["overlay"].dtype == np.uint8
    assert np.isfinite(result["raw_heatmap"]).all()
    assert 0 <= result["normalized_heatmap"].min()
    assert result["normalized_heatmap"].max() <= 1
    assert result["target_class"] == target_class


def test_gradcam_is_repeatable_for_same_target(
    grayscale_image: Image.Image,
) -> None:
    predictor = PredictorService()
    service = GradCAMService(predictor)

    first = service.generate(grayscale_image, target_class=0)
    second = service.generate(grayscale_image, target_class=0)

    assert np.allclose(
        first["normalized_heatmap"],
        second["normalized_heatmap"],
    )


def test_explicit_invalid_layer_has_clear_error() -> None:
    predictor = PredictorService()

    with pytest.raises(GradCAMError, match="not.a.layer"):
        GradCAMService(predictor, target_layer="not.a.layer")


def test_model_without_convolution_has_clear_error() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))

    with pytest.raises(GradCAMError, match="卷积层"):
        find_last_convolution(model)


def test_target_class_is_validated(rgb_image: Image.Image) -> None:
    service = GradCAMService(PredictorService())

    with pytest.raises(GradCAMError, match="目标类别"):
        service.generate(rgb_image, target_class=999)


def test_hooks_are_removed_after_generation(rgb_image: Image.Image) -> None:
    predictor = PredictorService()
    service = GradCAMService(predictor)
    layer = service.target_layer
    before_forward = len(layer._forward_hooks)
    before_backward = len(layer._backward_hooks)

    service.generate(rgb_image, target_class=0)

    assert len(layer._forward_hooks) == before_forward
    assert len(layer._backward_hooks) == before_backward
