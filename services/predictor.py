from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config.settings import (
    DEMO_DISCLAIMER,
    ModelSettings,
    PreprocessingSettings,
    get_settings,
)


class ModelIntegrationError(RuntimeError):
    """Raised when a configured model cannot be loaded safely."""


class DemoConvNet(nn.Module):
    """Small fixed-weight CNN used only for a clearly labeled demo path."""

    def __init__(self, number_of_classes: int = 2) -> None:
        super().__init__()
        if number_of_classes != 2:
            raise ModelIntegrationError("固定权重演示模型仅支持两个演示类别。")
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(6, number_of_classes)
        self._set_fixed_parameters()

    def _set_fixed_parameters(self) -> None:
        with torch.no_grad():
            first = self.features[0]
            second = self.features[2]
            assert isinstance(first, nn.Conv2d)
            assert isinstance(second, nn.Conv2d)

            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, :, 1, 1] = 1.0 / 3.0
            first.weight[1, :, :, :] = 1.0 / 27.0
            horizontal = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ) / 12.0
            vertical = horizontal.T
            first.weight[2, :, :, :] = horizontal / 3.0
            first.weight[3, :, :, :] = vertical / 3.0
            first.bias[2:] = 0.12

            second.weight.zero_()
            second.bias.zero_()
            second.weight[0, 0, 1, 1] = 1.0
            second.weight[1, 1, 1, 1] = 1.0
            second.weight[2, 2, 1, 1] = 1.0
            second.weight[3, 3, 1, 1] = 1.0
            second.weight[4, :, 1, 1] = 0.25
            second.weight[5, 0, 1, 1] = 0.8
            second.weight[5, 1, 1, 1] = -0.4
            second.bias[5] = 0.08

            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [1.20, 0.70, -0.40, -0.40, 0.20, 0.50],
                        [-0.20, 0.30, 1.00, 1.00, 0.40, -0.50],
                    ],
                    dtype=self.classifier.weight.dtype,
                )
            )
            self.classifier.bias.copy_(torch.tensor([0.10, -0.10]))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(tensor)
        return self.classifier(torch.flatten(features, 1))


def _build_torchvision_model(architecture: str, class_count: int) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model
    if architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model
    if architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, class_count)
        return model
    raise ModelIntegrationError(f"不支持的模型架构：{architecture}")


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ModelIntegrationError("检查点不包含可用的 state_dict。")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ModelIntegrationError("检查点 state_dict 的键或张量格式无效。")
        normalized[key.removeprefix("module.")] = value
    return normalized


class PredictorService:
    """Load one model and expose exact, serializable image predictions."""

    def __init__(
        self,
        model_settings: ModelSettings | None = None,
        preprocessing: PreprocessingSettings | None = None,
        model: nn.Module | None = None,
    ) -> None:
        defaults = get_settings()
        self.settings = model_settings or defaults.model
        self.preprocessing = preprocessing or defaults.preprocessing
        self.load_count = 0
        self.model = model or self._load_model()
        if model is not None:
            self.load_count = 1
        self.device = self._select_device()
        try:
            self.model = self.model.to(self.device)
        except RuntimeError:
            if self.device.type != "cuda":
                raise
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        self.model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize(self.preprocessing.image_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.preprocessing.mean,
                    std=self.preprocessing.std,
                ),
            ]
        )

    def _load_model(self) -> nn.Module:
        self.load_count += 1
        if self.settings.mode == "demo":
            return DemoConvNet(len(self.settings.class_names))

        checkpoint_path = Path(self.settings.checkpoint_path)
        if not checkpoint_path.is_file():
            raise ModelIntegrationError(
                f"模型检查点不存在：{checkpoint_path}。"
                "请在 config/settings.py 配置匹配的权重、标签和预处理参数。"
            )

        model = _build_torchvision_model(
            self.settings.architecture,
            len(self.settings.class_names),
        )
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(_extract_state_dict(payload), strict=True)
        except ModelIntegrationError:
            raise
        except Exception as exc:
            raise ModelIntegrationError(
                f"模型检查点与架构 {self.settings.architecture} "
                f"或 {len(self.settings.class_names)} 个类别不兼容：{exc}"
            ) from exc
        return model

    def _select_device(self) -> torch.device:
        if self.settings.prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("输入必须是 PIL Image。")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("输入图像尺寸必须为正数。")
        rgb_image = image.convert("RGB")
        tensor = self._transform(rgb_image).unsqueeze(0)
        if not torch.isfinite(tensor).all():
            raise ValueError("预处理结果包含无效数值。")
        return tensor

    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.preprocess(image).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ModelIntegrationError(
                    f"模型输出形状必须为 [1, 类别数]，实际为 {tuple(logits.shape)}。"
                )
            if logits.shape[1] != len(self.settings.class_names):
                raise ModelIntegrationError(
                    "模型输出类别数与 class_names 配置不一致。"
                )
            probabilities_tensor = torch.softmax(logits[0], dim=0).cpu()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        class_index = int(torch.argmax(probabilities_tensor).item())
        values = [float(value) for value in probabilities_tensor.tolist()]
        probabilities = {
            name: value
            for name, value in zip(
                self.settings.class_names,
                values,
                strict=True,
            )
        }
        return {
            "class_index": class_index,
            "class_name": self.settings.class_names[class_index],
            "confidence": float(values[class_index]),
            "probabilities": probabilities,
            "inference_time_ms": float(elapsed_ms),
            "model_version": self.settings.model_version,
            "device": str(self.device),
            "is_demo": self.settings.mode == "demo",
        }

    def get_status(self) -> dict[str, Any]:
        is_demo = self.settings.mode == "demo"
        message = (
            f"{DEMO_DISCLAIMER} 当前使用固定权重演示模型。"
            if is_demo
            else f"已加载模型检查点：{self.settings.checkpoint_path}"
        )
        return {
            "ready": True,
            "mode": self.settings.mode,
            "is_demo": is_demo,
            "version": self.settings.model_version,
            "architecture": (
                "DemoConvNet" if is_demo else self.settings.architecture
            ),
            "device": str(self.device),
            "class_names": list(self.settings.class_names),
            "message": message,
        }
