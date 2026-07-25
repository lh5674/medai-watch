from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from services.perturbations import (
    add_gaussian_blur,
    add_gaussian_noise,
    adjust_brightness,
    apply_perturbation,
)


@pytest.mark.parametrize(
    "function, kwargs",
    [
        (add_gaussian_blur, {"kernel_size": 5}),
        (add_gaussian_noise, {"sigma": 0.05, "seed": 7}),
        (adjust_brightness, {"factor": 1.25}),
    ],
)
@pytest.mark.parametrize("fixture_name", ["rgb_image", "grayscale_image"])
def test_perturbations_preserve_rgb_or_grayscale_mode_and_size(
    request: pytest.FixtureRequest,
    fixture_name: str,
    function,
    kwargs: dict[str, float | int],
) -> None:
    image = request.getfixturevalue(fixture_name)

    result = function(image, **kwargs)

    assert result.mode == image.mode
    assert result.size == image.size


def test_gaussian_noise_is_deterministic_for_a_seed(rgb_image: Image.Image) -> None:
    first = add_gaussian_noise(rgb_image, sigma=0.08, seed=1234)
    second = add_gaussian_noise(rgb_image, sigma=0.08, seed=1234)
    different_seed = add_gaussian_noise(rgb_image, sigma=0.08, seed=1235)

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(different_seed))


def test_zero_sigma_returns_identical_pixels(rgb_image: Image.Image) -> None:
    result = add_gaussian_noise(rgb_image, sigma=0.0, seed=11)

    assert np.array_equal(np.asarray(result), np.asarray(rgb_image))
    assert result is not rgb_image


@pytest.mark.parametrize("kernel_size", [0, 1, 2, 4, 6, -3, 3.0, True])
def test_blur_rejects_kernel_sizes_that_are_not_odd_integers_at_least_three(
    rgb_image: Image.Image,
    kernel_size: object,
) -> None:
    with pytest.raises(ValueError, match="odd integer.*3"):
        add_gaussian_blur(rgb_image, kernel_size=kernel_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("sigma", [-0.01, 1.01, np.inf, np.nan, "0.1"])
def test_noise_rejects_sigma_outside_closed_unit_interval(
    rgb_image: Image.Image,
    sigma: object,
) -> None:
    with pytest.raises(ValueError, match=r"sigma.*\[0, 1\]"):
        add_gaussian_noise(rgb_image, sigma=sigma)  # type: ignore[arg-type]


@pytest.mark.parametrize("factor", [0.0, -0.1, 3.01, np.inf, np.nan, "1.2"])
def test_brightness_rejects_factor_outside_zero_exclusive_to_three_inclusive(
    rgb_image: Image.Image,
    factor: object,
) -> None:
    with pytest.raises(ValueError, match=r"factor.*\(0, 3\]"):
        adjust_brightness(rgb_image, factor=factor)  # type: ignore[arg-type]


def test_brightness_clips_instead_of_wrapping() -> None:
    image = Image.fromarray(np.full((5, 7, 3), 250, dtype=np.uint8), mode="RGB")

    result = adjust_brightness(image, factor=3.0)

    assert np.asarray(result).min() == 255
    assert np.asarray(result).max() == 255


def test_noise_output_is_safely_clipped_to_uint8_range() -> None:
    pixels = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    image = Image.fromarray(pixels, mode="L")

    result = add_gaussian_noise(image, sigma=1.0, seed=9)
    output = np.asarray(result)

    assert output.dtype == np.uint8
    assert 0 <= int(output.min()) <= int(output.max()) <= 255


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("blur", {"kernel_size": 3}),
        ("noise", {"sigma": 0.01, "seed": 2}),
        ("brightness", {"factor": 0.9}),
    ],
)
def test_apply_perturbation_dispatches_supported_names(
    rgb_image: Image.Image,
    name: str,
    parameters: dict[str, float | int],
) -> None:
    result = apply_perturbation(rgb_image, name, **parameters)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == rgb_image.size


def test_apply_perturbation_rejects_unknown_name(rgb_image: Image.Image) -> None:
    with pytest.raises(ValueError, match="Unsupported perturbation"):
        apply_perturbation(rgb_image, "rotate")


def test_perturbations_reject_non_images() -> None:
    with pytest.raises(TypeError, match="PIL"):
        add_gaussian_blur(np.zeros((4, 4), dtype=np.uint8), kernel_size=3)  # type: ignore[arg-type]


def test_perturbations_reject_unsupported_pil_modes() -> None:
    image = Image.fromarray(np.zeros((4, 4, 4), dtype=np.uint8), mode="RGBA")

    with pytest.raises(ValueError, match="RGB.*L"):
        add_gaussian_noise(image, sigma=0.1)
