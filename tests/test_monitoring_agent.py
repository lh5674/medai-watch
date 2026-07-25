from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from agent.monitoring_agent import MedicalAIWatchAgent
from services.assets import AssetStore
from services.database import SQLiteRepository
from services.demo_cases import get_demo_cases
from services.gradcam_service import GradCAMService
from services.image_quality import ImageQualityService
from services.predictor import PredictorService
from services.report_generator import ReportGenerator
from services.risk_engine import RiskEngine
from services.similarity import SimilarityService


def _build_agent(tmp_path: Path) -> tuple[MedicalAIWatchAgent, SQLiteRepository]:
    predictor = PredictorService()
    repository = SQLiteRepository(tmp_path / "watch.db")
    agent = MedicalAIWatchAgent(
        predictor=predictor,
        image_quality=ImageQualityService(),
        gradcam=GradCAMService(predictor),
        similarity=SimilarityService(),
        risk_engine=RiskEngine(),
        report_generator=ReportGenerator(),
        asset_store=AssetStore(tmp_path / "assets"),
        database=repository,
    )
    return agent, repository


def test_complete_workflow_persists_analysis_and_perturbations(
    tmp_path: Path,
    rgb_image: Image.Image,
) -> None:
    agent, repository = _build_agent(tmp_path)

    result = agent.run(rgb_image, image_name="synthetic.png")

    assert result["record_id"] > 0
    assert result["quality"]["quality_score"] >= 0
    assert len(result["perturbation_results"]) == 3
    assert {item["test_name"] for item in result["perturbation_results"]} == {
        "gaussian_blur",
        "gaussian_noise",
        "brightness",
    }
    original_class = result["original_prediction"]["class_index"]
    assert result["original_gradcam"]["target_class"] == original_class
    assert all(
        item["gradcam"]["target_class"] == original_class
        for item in result["perturbation_results"]
    )
    assert 0 <= result["risk"]["risk_score"] <= 100
    assert "## 原型免责声明" in result["report"]

    stored = repository.get_analysis(result["record_id"])
    assert stored is not None
    assert stored["image_name"] == "synthetic.png"
    assert len(stored["perturbations"]) == 3


def test_workflow_requires_at_least_one_perturbation(
    tmp_path: Path,
    rgb_image: Image.Image,
) -> None:
    agent, _ = _build_agent(tmp_path)

    with pytest.raises(ValueError, match="扰动"):
        agent.run(rgb_image, selected_tests=[])


def test_progress_callback_reaches_completion(
    tmp_path: Path,
    rgb_image: Image.Image,
) -> None:
    agent, _ = _build_agent(tmp_path)
    updates: list[tuple[float, str]] = []

    agent.run(
        rgb_image,
        selected_tests=["gaussian_blur"],
        progress_callback=lambda progress, message: updates.append(
            (progress, message)
        ),
    )

    assert updates
    assert updates[0][0] == pytest.approx(0.0)
    assert updates[-1][0] == pytest.approx(1.0)
    assert all(0 <= progress <= 1 for progress, _ in updates)


def test_workflow_report_uses_chinese_perturbation_names(
    tmp_path: Path,
    rgb_image: Image.Image,
) -> None:
    agent, _ = _build_agent(tmp_path)

    result = agent.run(
        rgb_image,
        selected_tests=["gaussian_blur"],
    )

    assert "高斯模糊" in result["report"]
    assert "gaussian_blur" not in result["report"]


def test_export_payload_is_json_serializable_without_image_arrays(
    tmp_path: Path,
    rgb_image: Image.Image,
) -> None:
    agent, _ = _build_agent(tmp_path)
    result = agent.run(
        rgb_image,
        selected_tests=["gaussian_noise"],
    )

    payload = agent.export_payload(result)
    encoded = json.dumps(payload, ensure_ascii=False)

    assert "normalized_heatmap" not in encoded
    assert "overlay" not in encoded
    assert payload["record_id"] == result["record_id"]
    assert payload["perturbation_results"][0]["test_name"] == "gaussian_noise"


def test_demo_presets_cover_all_three_decision_bands(tmp_path: Path) -> None:
    agent, _ = _build_agent(tmp_path)

    results = []
    for case in get_demo_cases():
        result = agent.run(
            case["image"],
            image_name=f"{case['id']}.png",
            selected_tests=case["recommended_tests"],
            perturbation_parameters=case["recommended_parameters"],
        )
        results.append(result)

    assert [result["risk"]["decision"] for result in results] == [
        "通过",
        "谨慎使用",
        "建议人工复核",
    ]

    stable, medium, high = results
    assert all(
        item["comparison"]["class_unchanged"]
        for item in stable["perturbation_results"]
    )
    assert all(
        item["comparison"]["class_unchanged"]
        for item in medium["perturbation_results"]
    )
    assert max(
        item["comparison"]["confidence_change"]
        for item in medium["perturbation_results"]
    ) >= 0.05
    assert any(
        not item["comparison"]["class_unchanged"]
        for item in high["perturbation_results"]
    )
