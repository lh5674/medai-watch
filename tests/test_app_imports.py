from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from app import _analysis_signature
from config.settings import AppSettings, StorageSettings
from services.container import build_container
from services.exports import analysis_csv_bytes, analysis_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _result() -> dict:
    return {
        "record_id": 7,
        "quality": {"quality_score": 82.0, "issues": []},
        "original_prediction": {
            "class_index": 0,
            "class_name": "演示类别A",
            "confidence": 0.8,
            "probabilities": {"演示类别A": 0.8, "演示类别B": 0.2},
            "model_version": "demo-fixed-v1",
            "is_demo": True,
        },
        "perturbation_results": [
            {
                "test_name": "gaussian_blur",
                "parameters": {"kernel_size": 5},
                "prediction": {
                    "class_index": 0,
                    "class_name": "演示类别A",
                    "confidence": 0.7,
                    "probabilities": {"演示类别A": 0.7, "演示类别B": 0.3},
                },
                "comparison": {
                    "class_unchanged": True,
                    "confidence_change": 0.1,
                    "gradcam_cosine_similarity": 0.9,
                    "gradcam_ssim": 0.8,
                },
            }
        ],
        "risk": {
            "risk_score": 22.0,
            "decision": "通过",
            "decision_display": "✅ 通过",
            "components": {
                "confidence_risk": 0.2,
                "prediction_instability": 0.0,
                "explanation_risk": 0.15,
                "quality_risk": 0.18,
            },
            "disclaimer": "该评分仅用于黑客松研究原型，权重和阈值尚未经过临床验证。",
        },
        "report": "研究报告",
        "storage_error": None,
    }


def test_container_builds_with_storage_on_configured_path(tmp_path: Path) -> None:
    settings = AppSettings(
        storage=StorageSettings(
            database_path=tmp_path / "watch.db",
            asset_dir=tmp_path / "assets",
        )
    )

    container = build_container(settings)

    assert container.database.database_path == (tmp_path / "watch.db").resolve()
    assert container.predictor.get_status()["ready"] is True
    assert container.agent.database is container.database


def test_json_and_csv_exports_are_downloadable_bytes() -> None:
    result = _result()

    json_bytes = analysis_json_bytes(result)
    csv_bytes = analysis_csv_bytes(result)

    decoded = json.loads(json_bytes.decode("utf-8"))
    assert decoded["record_id"] == 7
    assert "normalized_heatmap" not in json_bytes.decode("utf-8")
    csv_text = csv_bytes.decode("utf-8-sig")
    assert "perturbation_name" in csv_text
    assert "gaussian_blur" in csv_text


def test_analysis_signature_changes_with_image_or_configuration() -> None:
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))
    parameters = {"gaussian_blur": {"kernel_size": 5}}

    baseline = _analysis_signature(
        image,
        "case.png",
        ["gaussian_blur"],
        parameters,
    )

    assert baseline == _analysis_signature(
        image.copy(),
        "case.png",
        ["gaussian_blur"],
        parameters,
    )
    assert baseline != _analysis_signature(
        Image.new("RGB", (8, 8), color=(11, 20, 30)),
        "case.png",
        ["gaussian_blur"],
        parameters,
    )
    assert baseline != _analysis_signature(
        image,
        "case.png",
        ["gaussian_blur"],
        {"gaussian_blur": {"kernel_size": 7}},
    )


def test_analysis_page_contains_required_chinese_controls() -> None:
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    for text in (
        "MedAI Watch",
        "研究原型",
        "开始安全监控分析",
        "模拟演示案例，不代表真实模型或临床结果。",
        "recommended_tests",
        "recommended_parameters",
        "JSON",
        "CSV",
    ):
        assert text in source
    assert 'risk["weights"]' in source
    assert 'risk["contributions"]' in source
    assert "max_value=0.30" in source
    assert 'value=float(noise_defaults["sigma"])' in source
    assert "@st.cache_resource" in (
        PROJECT_ROOT / "services" / "container.py"
    ).read_text(encoding="utf-8")


def test_medium_demo_preset_loads_supported_noise_slider() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)

    app.run()
    app.selectbox[0].select_index(1).run()

    sliders = {item.proto.label: item for item in list(app.slider)}
    checkboxes = {item.label: item.value for item in list(app.checkbox)}
    noise = sliders["噪声强度 σ"]
    assert not app.exception
    assert app.selectbox[0].value == "置信度变化中风险演示"
    assert noise.value == 0.25
    assert noise.proto.max == 0.30
    assert noise.proto.step == 0.01
    assert checkboxes == {
        "高斯模糊": False,
        "高斯噪声": True,
        "亮度调整": False,
    }


def test_dashboard_contains_required_filters_and_metrics() -> None:
    source = (PROJECT_ROOT / "pages" / "2_监控仪表盘.py").read_text(
        encoding="utf-8"
    )

    for text in (
        "监控仪表盘",
        "分析总数",
        "高风险案例",
        "平均风险分",
        "日期范围",
        "预测类别",
        "决策",
        "扰动失败分布",
        "最近分析记录",
        "PERTURBATION_DISPLAY_NAMES",
        "记录编号",
        "创建时间",
        "扰动名称",
        "类别是否保持",
    ):
        assert text in source
