from __future__ import annotations

from numbers import Integral
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn

from services.predictor import PredictorService


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM cannot be generated for a configured model."""


def resolve_layer(model: nn.Module, dotted_path: str) -> nn.Module:
    current: Any = model
    try:
        for part in dotted_path.split("."):
            if not part:
                raise AttributeError(part)
            current = current[int(part)] if part.isdigit() else getattr(current, part)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise GradCAMError(f"无法解析 Grad-CAM 目标层：{dotted_path}") from exc
    if not isinstance(current, nn.Module):
        raise GradCAMError(f"Grad-CAM 目标路径不是模型层：{dotted_path}")
    return current


def find_last_convolution(model: nn.Module) -> tuple[str, nn.Conv2d]:
    selected: tuple[str, nn.Conv2d] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            selected = (name, module)
    if selected is None:
        raise GradCAMError("模型中没有可用于 Grad-CAM 的卷积层。")
    return selected


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum <= minimum:
        return np.zeros_like(heatmap, dtype=np.float32)
    normalized = (heatmap - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _overlay_image(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored_bgr = cv2.applyColorMap(
        np.rint(heatmap * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = 0.58 * base + 0.42 * colored_rgb
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


class GradCAMService:
    """Generate raw, normalized, and overlaid Grad-CAM arrays."""

    def __init__(
        self,
        predictor: PredictorService,
        target_layer: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.model = predictor.model
        configured_layer = (
            target_layer
            if target_layer is not None
            else predictor.settings.target_layer
        )
        if configured_layer:
            layer = resolve_layer(self.model, configured_layer)
            if not isinstance(layer, nn.Conv2d):
                raise GradCAMError(
                    f"配置的 Grad-CAM 层不是 Conv2d：{configured_layer}"
                )
            self.target_layer_name = configured_layer
            self.target_layer = layer
        else:
            self.target_layer_name, self.target_layer = find_last_convolution(
                self.model
            )

    def generate(
        self,
        image: Image.Image,
        target_class: int,
    ) -> dict[str, Any]:
        if (
            isinstance(target_class, bool)
            or not isinstance(target_class, Integral)
            or target_class < 0
        ):
            raise GradCAMError("目标类别必须是有效的非负整数。")

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def capture_activation(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            activations.append(output.detach())

        def capture_gradient(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach())

        forward_handle = self.target_layer.register_forward_hook(capture_activation)
        backward_handle = self.target_layer.register_full_backward_hook(
            capture_gradient
        )
        try:
            tensor = self.predictor.preprocess(image).to(self.predictor.device)
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = self.model(tensor)
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise GradCAMError(
                        f"模型输出形状不支持 Grad-CAM：{tuple(logits.shape)}"
                    )
                if target_class >= logits.shape[1]:
                    raise GradCAMError(
                        f"目标类别 {target_class} 超出模型输出范围。"
                    )
                logits[0, int(target_class)].backward()

            if not activations or not gradients:
                raise GradCAMError("未能从目标卷积层捕获激活或梯度。")
            activation = activations[-1]
            gradient = gradients[-1]
            if activation.ndim != 4 or gradient.shape != activation.shape:
                raise GradCAMError("Grad-CAM 激活与梯度形状不兼容。")

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            resized = functional.interpolate(
                cam,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            raw_heatmap = (
                resized[0, 0].detach().cpu().numpy().astype(np.float32)
            )
            normalized_heatmap = _normalize_heatmap(raw_heatmap)
            overlay = _overlay_image(image, normalized_heatmap)
            return {
                "raw_heatmap": raw_heatmap,
                "normalized_heatmap": normalized_heatmap,
                "overlay": overlay,
                "target_class": int(target_class),
                "target_layer": self.target_layer_name,
            }
        except GradCAMError:
            raise
        except Exception as exc:
            raise GradCAMError(f"Grad-CAM 生成失败：{exc}") from exc
        finally:
            forward_handle.remove()
            backward_handle.remove()

    def generate_gradcam(
        self,
        image: Image.Image,
        target_class: int,
    ) -> dict[str, Any]:
        return self.generate(image=image, target_class=target_class)
