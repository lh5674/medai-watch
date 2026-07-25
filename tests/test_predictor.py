from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from config.settings import ModelSettings, PreprocessingSettings
from services.predictor import (
    DemoConvNet,
    ModelIntegrationError,
    PredictorService,
)


def test_demo_predictions_are_deterministic_and_json_serializable(
    rgb_image: Image.Image,
) -> None:
    service = PredictorService()

    first = service.predict(rgb_image)
    second = service.predict(rgb_image)

    assert first["class_index"] == second["class_index"]
    assert first["class_name"] == second["class_name"]
    assert first["confidence"] == pytest.approx(second["confidence"])
    assert first["probabilities"] == pytest.approx(second["probabilities"])
    assert first["is_demo"] is True
    json.dumps(first, ensure_ascii=False)


def test_prediction_contract_and_probability_sum(
    rgb_image: Image.Image,
) -> None:
    service = PredictorService()

    result = service.predict(rgb_image)

    assert set(result) >= {
        "class_index",
        "class_name",
        "confidence",
        "probabilities",
        "inference_time_ms",
        "model_version",
        "device",
        "is_demo",
    }
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert set(result["probabilities"]) == {"演示类别A", "演示类别B"}
    assert 0 <= result["confidence"] <= 1
    assert result["inference_time_ms"] >= 0


def test_preprocessing_uses_configured_shape_and_finite_values(
    grayscale_image: Image.Image,
) -> None:
    preprocessing = PreprocessingSettings(image_size=(96, 80))
    service = PredictorService(preprocessing=preprocessing)

    tensor = service.preprocess(grayscale_image)

    assert tuple(tensor.shape) == (1, 3, 96, 80)
    assert torch.isfinite(tensor).all()


def test_predictor_rejects_non_image_input() -> None:
    service = PredictorService()

    with pytest.raises(TypeError, match="PIL"):
        service.predict(np.zeros((32, 32), dtype=np.uint8))


def test_model_is_loaded_once_and_kept_in_eval_mode(
    rgb_image: Image.Image,
) -> None:
    service = PredictorService()
    model_identity = id(service.model)

    service.predict(rgb_image)
    service.predict(rgb_image)

    assert id(service.model) == model_identity
    assert service.load_count == 1
    assert service.model.training is False


def test_missing_checkpoint_has_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pth"
    settings = ModelSettings(
        mode="checkpoint",
        checkpoint_path=missing_path,
        class_names=("A", "B"),
    )

    with pytest.raises(ModelIntegrationError, match="missing.pth"):
        PredictorService(model_settings=settings)


def test_incompatible_checkpoint_has_clear_error(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bad.pth"
    torch.save({"state_dict": {"wrong.weight": torch.ones(2, 2)}}, checkpoint_path)
    settings = ModelSettings(
        mode="checkpoint",
        architecture="resnet18",
        checkpoint_path=checkpoint_path,
        class_names=("A", "B"),
    )

    with pytest.raises(ModelIntegrationError, match="不兼容"):
        PredictorService(model_settings=settings)


def test_unknown_checkpoint_architecture_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pth"
    torch.save({}, checkpoint_path)
    settings = ModelSettings(
        mode="checkpoint",
        architecture="unknown-net",
        checkpoint_path=checkpoint_path,
        class_names=("A", "B"),
    )

    with pytest.raises(ModelIntegrationError, match="unknown-net"):
        PredictorService(model_settings=settings)


def test_model_status_exposes_demo_warning() -> None:
    status = PredictorService().get_status()

    assert status["ready"] is True
    assert status["is_demo"] is True
    assert status["version"] == "demo-fixed-v1"
    assert "模拟演示" in status["message"]


def test_cuda_initialization_failure_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    rgb_image: Image.Image,
) -> None:
    class CudaFailingDemoModel(DemoConvNet):
        def to(self, device: torch.device | str) -> DemoConvNet:
            if torch.device(device).type == "cuda":
                raise RuntimeError("CUDA unavailable at initialization")
            return super().to(device)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    service = PredictorService(model=CudaFailingDemoModel())
    result = service.predict(rgb_image)

    assert service.device.type == "cpu"
    assert result["device"] == "cpu"
