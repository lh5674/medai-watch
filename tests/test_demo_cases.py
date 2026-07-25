from __future__ import annotations

import numpy as np
from PIL import Image

from config.settings import DEMO_DISCLAIMER
from services.demo_cases import get_demo_cases


def test_demo_cases_are_clearly_labeled_and_complete() -> None:
    cases = get_demo_cases()

    assert [case["id"] for case in cases] == [
        "stable_low_risk",
        "confidence_shift_medium_risk",
        "prediction_change_high_risk",
    ]
    assert [case["name"] for case in cases] == [
        "稳定低风险演示",
        "置信度变化中风险演示",
        "预测变化高风险演示",
    ]
    assert all(case["disclaimer"] == DEMO_DISCLAIMER for case in cases)
    assert all(case["simulated"] is True for case in cases)


def test_demo_case_images_are_rgb_and_same_size() -> None:
    images = [case["image"] for case in get_demo_cases()]

    assert all(isinstance(image, Image.Image) for image in images)
    assert all(image.mode == "RGB" for image in images)
    assert all(image.size == (224, 224) for image in images)


def test_demo_cases_are_deterministic() -> None:
    first = get_demo_cases()
    second = get_demo_cases()

    for first_case, second_case in zip(first, second, strict=True):
        assert np.array_equal(
            np.asarray(first_case["image"]),
            np.asarray(second_case["image"]),
        )


def test_demo_cases_do_not_contain_prefabricated_model_outputs() -> None:
    forbidden_keys = {
        "prediction",
        "confidence",
        "probabilities",
        "similarity",
        "score",
        "risk_score",
        "decision",
    }

    for case in get_demo_cases():
        assert forbidden_keys.isdisjoint(case)


def test_demo_cases_define_computed_workflow_presets() -> None:
    stable, medium, high = get_demo_cases()

    assert stable["recommended_tests"] == [
        "gaussian_blur",
        "gaussian_noise",
        "brightness",
    ]
    assert medium["recommended_tests"] == ["gaussian_noise"]
    assert medium["recommended_parameters"] == {
        "gaussian_noise": {"sigma": 0.25, "seed": 42}
    }
    assert high["recommended_tests"] == ["brightness"]
    assert high["recommended_parameters"]["brightness"]["factor"] == 0.2
