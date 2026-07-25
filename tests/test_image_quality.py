from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFilter

from config.settings import QualitySettings
from services.image_quality import ImageQualityService, load_image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("fixture_name", ["rgb_image", "grayscale_image"])
def test_load_image_accepts_pil_images_and_preserves_mode_and_dimensions(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    source = request.getfixturevalue(fixture_name)

    result = load_image(source)

    assert result.mode == source.mode
    assert result.size == source.size
    assert result is not source


def test_load_image_accepts_path_bytes_and_file_like(
    tmp_path,
    rgb_image: Image.Image,
) -> None:
    image_bytes = _png_bytes(rgb_image)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(image_bytes)

    from_path = load_image(image_path)
    from_bytes = load_image(image_bytes)
    stream = BytesIO(image_bytes)
    from_stream = load_image(stream)

    for result in (from_path, from_bytes, from_stream):
        assert result.mode == "RGB"
        assert result.size == rgb_image.size
        assert np.array_equal(np.asarray(result), np.asarray(rgb_image))
    assert not stream.closed


@pytest.mark.parametrize(
    "source",
    [
        b"this is not an image",
        BytesIO(b"\x89PNG\r\ntruncated"),
    ],
)
def test_load_image_reports_corrupted_content(source: object) -> None:
    with pytest.raises(ValueError, match="decode image"):
        load_image(source)


def test_load_image_reports_missing_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="decode image"):
        load_image(tmp_path / "missing.png")


@pytest.mark.parametrize("source", [None, 123, object(), np.zeros((2, 2))])
def test_load_image_rejects_unsupported_input_types(source: object) -> None:
    with pytest.raises(TypeError, match="PIL image, path, bytes, or binary file-like"):
        load_image(source)


@pytest.mark.parametrize("fixture_name", ["rgb_image", "grayscale_image"])
def test_assessment_reports_dimensions_and_numeric_metrics(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    image = request.getfixturevalue(fixture_name)

    result = ImageQualityService().assess(image)

    assert result["mode"] == image.mode
    assert result["width"] == image.width
    assert result["height"] == image.height
    assert result["dimensions"] == {
        "width": image.width,
        "height": image.height,
    }
    assert result["valid"] is True
    for metric in (
        "laplacian_variance",
        "mean_brightness",
        "contrast_std",
        "extreme_pixel_ratio",
        "blur_score",
        "brightness_score",
        "contrast_score",
        "quality_score",
    ):
        assert np.isfinite(result[metric])
    assert 0.0 <= result["mean_brightness"] <= 1.0
    assert 0.0 <= result["contrast_std"] <= 1.0
    assert 0.0 <= result["extreme_pixel_ratio"] <= 1.0


def test_laplacian_variance_is_higher_for_sharp_edges() -> None:
    checkerboard = (
        (np.indices((96, 96)).sum(axis=0) // 4 % 2) * 255
    ).astype(np.uint8)
    sharp = Image.fromarray(checkerboard, mode="L")
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    service = ImageQualityService()

    sharp_result = service.assess(sharp)
    blurred_result = service.assess(blurred)

    assert sharp_result["laplacian_variance"] > blurred_result["laplacian_variance"]
    assert sharp_result["sub_scores"]["blur"] >= blurred_result["sub_scores"]["blur"]


@pytest.mark.parametrize(
    "pixel_value, expected_brightness",
    [(0, 0.0), (255, 1.0)],
)
def test_black_and_white_images_are_detected_as_extreme(
    pixel_value: int,
    expected_brightness: float,
) -> None:
    image = Image.fromarray(
        np.full((80, 96), pixel_value, dtype=np.uint8),
        mode="L",
    )

    result = ImageQualityService().assess(image)

    assert result["mean_brightness"] == pytest.approx(expected_brightness)
    assert result["contrast_std"] == pytest.approx(0.0)
    assert result["extreme_pixel_ratio"] == pytest.approx(1.0)
    assert result["sub_scores"]["brightness"] == pytest.approx(0.0)
    assert result["sub_scores"]["extreme_pixels"] == pytest.approx(0.0)
    assert any("极端像素" in issue for issue in result["issues"])


def test_small_dimensions_are_reported_and_reduce_dimension_score() -> None:
    image = Image.fromarray(np.full((20, 30, 3), 127, dtype=np.uint8), mode="RGB")

    result = ImageQualityService().assess(image)

    assert result["valid"] is False
    assert result["sub_scores"]["dimensions"] < 1.0
    assert any("尺寸过小" in issue for issue in result["issues"])


def test_configurable_thresholds_are_taken_from_quality_settings(
    rgb_image: Image.Image,
) -> None:
    permissive = QualitySettings(min_width=32, min_height=32)
    strict = QualitySettings(min_width=192, min_height=160)

    permissive_result = ImageQualityService(permissive).assess(rgb_image)
    strict_result = ImageQualityService(strict).assess(rgb_image)

    assert permissive_result["sub_scores"]["dimensions"] == pytest.approx(1.0)
    assert strict_result["sub_scores"]["dimensions"] < 1.0


def test_all_sub_scores_and_quality_score_are_finite_and_bounded(
    rgb_image: Image.Image,
) -> None:
    result = ImageQualityService().assess(rgb_image)

    assert result["sub_scores"]
    assert all(
        np.isfinite(score) and 0.0 <= score <= 1.0
        for score in result["sub_scores"].values()
    )
    assert result["blur_score"] == result["sub_scores"]["blur"]
    assert result["brightness_score"] == result["sub_scores"]["brightness"]
    assert result["contrast_score"] == result["sub_scores"]["contrast"]
    assert np.isfinite(result["quality_score"])
    assert 0.0 <= result["quality_score"] <= 100.0


def test_weighted_quality_score_uses_configured_weights() -> None:
    image = Image.fromarray(np.full((80, 96), 127, dtype=np.uint8), mode="L")
    settings = QualitySettings(
        blur_weight=0.0,
        brightness_weight=1.0,
        contrast_weight=0.0,
        dimensions_weight=0.0,
    )

    result = ImageQualityService(settings).assess(image)

    assert result["quality_score"] == pytest.approx(
        100.0 * result["sub_scores"]["brightness"]
    )


def test_assessment_accepts_encoded_bytes(rgb_image: Image.Image) -> None:
    result = ImageQualityService().assess(_png_bytes(rgb_image))

    assert result["width"] == rgb_image.width
    assert result["height"] == rgb_image.height


def test_assessment_rejects_non_finite_pixel_values() -> None:
    image = Image.fromarray(
        np.array([[0.0, np.nan], [1.0, np.inf]], dtype=np.float32),
        mode="F",
    )

    with pytest.raises(ValueError, match="finite"):
        ImageQualityService().assess(image)
