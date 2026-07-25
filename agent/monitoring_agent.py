from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from PIL import Image

from config.settings import PerturbationSettings, get_settings
from services.assets import AssetStore
from services.database import SQLiteRepository
from services.gradcam_service import GradCAMService
from services.image_quality import ImageQualityService
from services.perturbations import (
    PERTURBATION_DISPLAY_NAMES,
    apply_perturbation,
)
from services.predictor import PredictorService
from services.report_generator import ReportGenerator
from services.risk_engine import RiskEngine
from services.similarity import SimilarityService


ProgressCallback = Callable[[float, str], None]

SUPPORTED_TESTS = ("gaussian_blur", "gaussian_noise", "brightness")


class MedicalAIWatchAgent:
    """Deterministic orchestration for prediction-stability monitoring."""

    def __init__(
        self,
        *,
        predictor: PredictorService,
        image_quality: ImageQualityService,
        gradcam: GradCAMService,
        similarity: SimilarityService,
        risk_engine: RiskEngine,
        report_generator: ReportGenerator,
        asset_store: AssetStore,
        database: SQLiteRepository,
        perturbation_settings: PerturbationSettings | None = None,
    ) -> None:
        self.predictor = predictor
        self.image_quality = image_quality
        self.gradcam = gradcam
        self.similarity = similarity
        self.risk_engine = risk_engine
        self.report_generator = report_generator
        self.asset_store = asset_store
        self.database = database
        self.perturbation_settings = (
            perturbation_settings or get_settings().perturbations
        )

    @staticmethod
    def _progress(
        callback: ProgressCallback | None,
        value: float,
        message: str,
    ) -> None:
        if callback is not None:
            callback(float(value), message)

    def choose_tests(
        self,
        _quality_result: Mapping[str, Any],
        selected_tests: Sequence[str] | None = None,
    ) -> list[str]:
        tests = list(SUPPORTED_TESTS if selected_tests is None else selected_tests)
        if not tests:
            raise ValueError("至少需要选择一项扰动测试。")
        if len(set(tests)) != len(tests):
            raise ValueError("扰动测试不得重复。")
        unsupported = [name for name in tests if name not in SUPPORTED_TESTS]
        if unsupported:
            raise ValueError(f"不支持的扰动测试：{', '.join(unsupported)}")
        return tests

    def _parameters_for(
        self,
        test_name: str,
        overrides: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        defaults: dict[str, dict[str, Any]] = {
            "gaussian_blur": {
                "kernel_size": self.perturbation_settings.blur_kernel_size
            },
            "gaussian_noise": {
                "sigma": self.perturbation_settings.noise_sigma,
                "seed": self.perturbation_settings.noise_seed,
            },
            "brightness": {
                "factor": self.perturbation_settings.brightness_factor
            },
        }
        parameters = defaults[test_name]
        parameters.update(dict(overrides.get(test_name, {})))
        return parameters

    def run(
        self,
        image: Image.Image,
        image_name: str | None = None,
        *,
        selected_tests: Sequence[str] | None = None,
        perturbation_parameters: Mapping[str, Mapping[str, Any]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("分析输入必须是 PIL Image。")
        parameters = perturbation_parameters or {}
        self._progress(progress_callback, 0.0, "正在检查图像技术质量")
        quality_result = self.image_quality.assess(image)
        tests = self.choose_tests(quality_result, selected_tests)

        self._progress(progress_callback, 0.10, "正在运行原始图像推理")
        original_prediction = self.predictor.predict(image)
        self._progress(progress_callback, 0.20, "正在生成原始 Grad-CAM")
        original_gradcam = self.gradcam.generate(
            image=image,
            target_class=original_prediction["class_index"],
        )

        perturbation_results: list[dict[str, Any]] = []
        for index, test_name in enumerate(tests, start=1):
            changed_image = apply_perturbation(
                image,
                test_name,
                **self._parameters_for(test_name, parameters),
            )
            changed_prediction = self.predictor.predict(changed_image)
            changed_gradcam = self.gradcam.generate(
                image=changed_image,
                target_class=original_prediction["class_index"],
            )
            comparison = self.similarity.compare_predictions(
                original_prediction,
                changed_prediction,
                original_gradcam["normalized_heatmap"],
                changed_gradcam["normalized_heatmap"],
            )
            perturbation_results.append(
                {
                    "test_name": test_name,
                    "perturbation_name": test_name,
                    "parameters": self._parameters_for(test_name, parameters),
                    "image": changed_image,
                    "prediction": changed_prediction,
                    "gradcam": changed_gradcam,
                    "comparison": comparison,
                }
            )
            fraction = 0.20 + 0.55 * index / len(tests)
            self._progress(
                progress_callback,
                fraction,
                f"已完成扰动测试：{test_name}",
            )

        comparisons = [item["comparison"] for item in perturbation_results]
        risk_result = self.risk_engine.evaluate(
            original_confidence=original_prediction["confidence"],
            perturbations=comparisons,
            quality_score=quality_result["quality_score"],
        )
        report_items = [
            {
                "name": PERTURBATION_DISPLAY_NAMES[item["test_name"]],
                "prediction": item["prediction"],
                **item["comparison"],
            }
            for item in perturbation_results
        ]
        report = self.report_generator.generate(
            quality=quality_result,
            prediction=original_prediction,
            comparisons=report_items,
            risk=risk_result,
        )
        self._progress(progress_callback, 0.85, "正在保存分析记录")

        record_id: int | None = None
        storage_error: str | None = None
        try:
            saved_assets = self.asset_store.save_case(
                original_image=image,
                original_gradcam=original_gradcam["overlay"],
                perturbations=[
                    {
                        "name": item["test_name"],
                        "image": item["image"],
                        "gradcam": item["gradcam"]["overlay"],
                    }
                    for item in perturbation_results
                ],
            )
            analysis_row = {
                "image_name": image_name or "未命名影像.png",
                "predicted_class": original_prediction["class_name"],
                "original_confidence": original_prediction["confidence"],
                "quality_score": quality_result["quality_score"],
                "risk_score": risk_result["risk_score"],
                "decision": risk_result["decision"],
                "model_version": original_prediction["model_version"],
                "report": report,
                "original_image_path": saved_assets["original_image_path"],
                "gradcam_path": saved_assets["gradcam_path"],
            }
            perturbation_rows = []
            for item, paths in zip(
                perturbation_results,
                saved_assets["perturbations"],
                strict=True,
            ):
                comparison = item["comparison"]
                prediction = item["prediction"]
                perturbation_rows.append(
                    {
                        "perturbation_name": item["test_name"],
                        "changed_class": prediction["class_name"],
                        "changed_confidence": prediction["confidence"],
                        "confidence_change": comparison["confidence_change"],
                        "class_unchanged": comparison["class_unchanged"],
                        "gradcam_cosine_similarity": comparison[
                            "gradcam_cosine_similarity"
                        ],
                        "gradcam_ssim": comparison["gradcam_ssim"],
                        "perturbed_image_path": paths["perturbed_image_path"],
                        "perturbed_gradcam_path": paths[
                            "perturbed_gradcam_path"
                        ],
                    }
                )
            record_id = self.database.save_analysis(
                analysis_row,
                perturbation_rows,
            )
        except Exception as exc:
            storage_error = str(exc)

        result = {
            "record_id": record_id,
            "quality": quality_result,
            "original_prediction": original_prediction,
            "original_gradcam": original_gradcam,
            "perturbation_results": perturbation_results,
            "risk": risk_result,
            "report": report,
            "storage_error": storage_error,
        }
        self._progress(progress_callback, 1.0, "分析完成")
        return result

    @staticmethod
    def export_payload(result: Mapping[str, Any]) -> dict[str, Any]:
        perturbations = []
        for item in result["perturbation_results"]:
            perturbations.append(
                {
                    "test_name": item["test_name"],
                    "parameters": dict(item["parameters"]),
                    "prediction": dict(item["prediction"]),
                    "comparison": dict(item["comparison"]),
                }
            )
        return {
            "record_id": result.get("record_id"),
            "quality": dict(result["quality"]),
            "original_prediction": dict(result["original_prediction"]),
            "perturbation_results": perturbations,
            "risk": dict(result["risk"]),
            "report": result["report"],
            "storage_error": result.get("storage_error"),
        }
