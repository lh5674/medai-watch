from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from PIL import Image


_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.png$", re.IGNORECASE)


def _as_png_image(value: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, Image.Image):
        if value.mode == "L":
            return value.copy()
        return value.convert("RGB")
    if not isinstance(value, np.ndarray):
        raise TypeError("图像必须是 PIL.Image 或 NumPy 数组。")

    array = np.asarray(value)
    if array.ndim == 2:
        finite = np.isfinite(array)
        normalized = np.zeros(array.shape, dtype=np.float64)
        if finite.any():
            finite_values = array[finite].astype(np.float64, copy=False)
            low = float(finite_values.min())
            high = float(finite_values.max())
            if high > low:
                normalized[finite] = (finite_values - low) / (high - low)
        pixels = np.rint(normalized * 255.0).astype(np.uint8)
        return Image.fromarray(pixels, mode="L")

    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("NumPy 图像形状必须为 H×W、H×W×3 或 H×W×4。")

    numeric = np.nan_to_num(
        array.astype(np.float64, copy=False),
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    if np.issubdtype(array.dtype, np.floating):
        minimum = float(numeric.min()) if numeric.size else 0.0
        maximum = float(numeric.max()) if numeric.size else 0.0
        if minimum >= 0.0 and maximum <= 1.0:
            numeric *= 255.0
    pixels = np.rint(np.clip(numeric, 0.0, 255.0)).astype(np.uint8)
    mode = "RGBA" if pixels.shape[2] == 4 else "RGB"
    image = Image.fromarray(pixels, mode=mode)
    return image.convert("RGB")


def _first_value(item: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


class AssetStore:
    """Persist analysis images as PNG files below UUID-named case folders."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_case_directory(self, case_id: str | UUID | None = None) -> Path:
        identifier = uuid4() if case_id is None else UUID(str(case_id))
        case_dir = self.root / str(identifier)
        case_dir.mkdir(parents=False, exist_ok=False)
        return case_dir

    def save_png(
        self,
        image: Image.Image | np.ndarray,
        case_id: str | UUID,
        filename: str,
    ) -> Path:
        if not _SAFE_FILENAME.fullmatch(filename):
            raise ValueError("PNG 文件名只能包含安全的字母、数字、点、横线和下划线。")
        identifier = UUID(str(case_id))
        case_dir = self.root / str(identifier)
        resolved_case_dir = case_dir.resolve()
        if (
            resolved_case_dir != case_dir
            or case_dir.parent != self.root
            or not case_dir.is_dir()
        ):
            raise ValueError("案例目录不存在或不位于资产根目录中。")

        target = (case_dir / filename).resolve()
        if target.parent != case_dir:
            raise ValueError("PNG 文件名不得离开案例目录。")
        temporary = case_dir / f".{filename}.{uuid4().hex}.tmp"
        try:
            _as_png_image(image).save(temporary, format="PNG")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    save_image = save_png

    def save_case(
        self,
        original_image: Image.Image | np.ndarray,
        original_gradcam: Image.Image | np.ndarray,
        perturbations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        case_id: str | UUID | None = None,
    ) -> dict[str, Any]:
        case_dir = self.create_case_directory(case_id)
        identifier = case_dir.name
        original_path = self.save_png(original_image, identifier, "original.png")
        gradcam_path = self.save_png(
            original_gradcam, identifier, "original_gradcam.png"
        )

        saved_perturbations: list[dict[str, str]] = []
        for index, item in enumerate(perturbations, start=1):
            image = _first_value(item, "image", "perturbed_image")
            gradcam = _first_value(
                item,
                "gradcam",
                "perturbed_gradcam",
                "gradcam_overlay",
                "overlay",
            )
            if image is None or gradcam is None:
                raise ValueError("每个扰动结果都必须包含扰动图像和 Grad-CAM 图像。")
            image_path = self.save_png(
                image, identifier, f"perturbation_{index:02d}.png"
            )
            perturbation_gradcam_path = self.save_png(
                gradcam, identifier, f"perturbation_{index:02d}_gradcam.png"
            )
            name = str(
                item.get("name")
                or item.get("perturbation_name")
                or item.get("type")
                or f"perturbation_{index:02d}"
            )
            saved_perturbations.append(
                {
                    "name": name,
                    "perturbation_name": name,
                    "perturbed_image_path": str(image_path),
                    "perturbed_gradcam_path": str(perturbation_gradcam_path),
                }
            )

        return {
            "case_id": identifier,
            "case_dir": str(case_dir),
            "original_image_path": str(original_path),
            "gradcam_path": str(gradcam_path),
            "original_gradcam_path": str(gradcam_path),
            "perturbations": saved_perturbations,
        }

    save_case_assets = save_case


AssetManager = AssetStore
AssetRepository = AssetStore
