from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_DISCLAIMER = "该评分仅用于黑客松研究原型，权重和阈值尚未经过临床验证。"
MEDICAL_DISCLAIMER = "本结果仅用于研究和系统演示，不可用于临床诊断或治疗决策。"
DEMO_DISCLAIMER = "模拟演示案例，不代表真实模型或临床结果。"


@dataclass(frozen=True)
class PreprocessingSettings:
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("预处理图像尺寸必须包含两个正整数。")
        if len(self.mean) != 3 or len(self.std) != 3 or min(self.std) <= 0:
            raise ValueError("预处理均值和标准差必须各包含三个有效数值。")


@dataclass(frozen=True)
class ModelSettings:
    mode: str = "demo"
    architecture: str = "resnet18"
    checkpoint_path: Path = PROJECT_ROOT / "models" / "model.pth"
    class_names: tuple[str, ...] = ("演示类别A", "演示类别B")
    model_version: str = "demo-fixed-v1"
    target_layer: str | None = None
    prefer_cuda: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"demo", "checkpoint"}:
            raise ValueError("模型模式必须为 demo 或 checkpoint。")
        if len(self.class_names) < 2 or any(not name.strip() for name in self.class_names):
            raise ValueError("至少需要两个非空类别名称。")


@dataclass(frozen=True)
class QualitySettings:
    min_width: int = 64
    min_height: int = 64
    max_dimension: int = 8192
    blur_variance_low: float = 35.0
    blur_variance_good: float = 180.0
    brightness_low: float = 0.15
    brightness_high: float = 0.85
    contrast_low: float = 0.08
    contrast_good: float = 0.25
    extreme_fraction_max: float = 0.65
    blur_weight: float = 0.35
    brightness_weight: float = 0.25
    contrast_weight: float = 0.25
    dimensions_weight: float = 0.15

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            "blur": self.blur_weight,
            "brightness": self.brightness_weight,
            "contrast": self.contrast_weight,
            "dimensions": self.dimensions_weight,
        }


@dataclass(frozen=True)
class PerturbationSettings:
    blur_kernel_size: int = 5
    noise_sigma: float = 0.02
    noise_seed: int = 42
    brightness_factor: float = 0.85


@dataclass(frozen=True)
class RiskSettings:
    confidence_weight: float = 0.25
    instability_weight: float = 0.35
    explanation_weight: float = 0.25
    quality_weight: float = 0.15
    pass_threshold: float = 30.0
    caution_threshold: float = 60.0
    disclaimer: str = RISK_DISCLAIMER

    def __post_init__(self) -> None:
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("风险评分权重之和必须为 1。")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("风险评分权重不得为负数。")
        if not 0 <= self.pass_threshold < self.caution_threshold <= 100:
            raise ValueError("风险评分阈值必须满足 0 ≤ 通过阈值 < 谨慎阈值 ≤ 100。")

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            "confidence_risk": self.confidence_weight,
            "prediction_instability": self.instability_weight,
            "explanation_risk": self.explanation_weight,
            "quality_risk": self.quality_weight,
        }


@dataclass(frozen=True)
class StorageSettings:
    database_path: Path = PROJECT_ROOT / "data" / "medai_watch.db"
    asset_dir: Path = PROJECT_ROOT / "data" / "assets"


@dataclass(frozen=True)
class AppSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    perturbations: PerturbationSettings = field(default_factory=PerturbationSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
