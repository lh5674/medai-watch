from __future__ import annotations

from config.settings import DEMO_DISCLAIMER, MEDICAL_DISCLAIMER, RISK_DISCLAIMER
from services.report_generator import ReportGenerator


def _report_inputs(*, is_demo: bool = True) -> dict[str, object]:
    return {
        "quality": {
            "score": 82.5,
            "issues": ["亮度偏低"],
            "metrics": {
                "blur_variance": 126.25,
                "mean_brightness": 0.42,
                "contrast": 0.18,
                "extreme_fraction": 0.03,
            },
        },
        "prediction": {
            "class_index": 1,
            "class_name": "演示类别B",
            "confidence": 0.81234,
            "probabilities": {"演示类别A": 0.18766, "演示类别B": 0.81234},
            "model_version": "demo-fixed-v1",
            "device": "cpu",
            "is_demo": is_demo,
        },
        "comparisons": [
            {
                "name": "高斯模糊",
                "predicted_class": "演示类别B",
                "confidence": 0.781,
                "class_changed": False,
                "confidence_delta": -0.03134,
                "cosine_similarity": 0.91,
                "ssim": 0.87,
            },
            {
                "name": "高斯噪声",
                "predicted_class": "演示类别A",
                "confidence": 0.55,
                "class_changed": True,
                "confidence_delta": -0.26234,
                "cosine_similarity": 0.62,
                "ssim": 0.58,
            },
        ],
        "risk": {
            "score": 64.2,
            "decision": "建议人工复核",
            "components": {
                "confidence_risk": 0.18766,
                "prediction_instability": 0.5,
                "explanation_risk": 0.255,
                "quality_risk": 0.175,
            },
            "weights": {
                "confidence_risk": 0.25,
                "prediction_instability": 0.35,
                "explanation_risk": 0.25,
                "quality_risk": 0.15,
            },
            "contributions": {
                "confidence_risk": 4.6915,
                "prediction_instability": 17.5,
                "explanation_risk": 6.375,
                "quality_risk": 2.625,
            },
            "disclaimer": RISK_DISCLAIMER,
        },
    }


def test_report_contains_all_eight_chinese_sections_and_supplied_facts() -> None:
    report = ReportGenerator().generate(**_report_inputs())

    expected_sections = (
        "检查摘要",
        "原始模型结果",
        "图像质量",
        "扰动测试结果",
        "解释稳定性",
        "风险评分组成",
        "建议操作",
        "原型免责声明",
    )
    actual_sections = [
        line.removeprefix("## ")
        for line in report.splitlines()
        if line.startswith("## ")
    ]
    assert actual_sections == list(expected_sections)
    assert "82.50" in report
    assert "演示类别B" in report
    assert "81.23%" in report
    assert "高斯模糊" in report
    assert "高斯噪声" in report
    assert "64.20" in report


def test_report_is_deterministic_and_uses_only_supplied_non_clinical_facts() -> None:
    inputs = _report_inputs()

    first = ReportGenerator().generate(**inputs)
    second = ReportGenerator().generate(**inputs)

    assert first == second
    assert "诊断为" not in first
    assert "建议用药" not in first
    assert "病灶" not in first
    assert "治愈" not in first
    assert "未知疾病" not in first


def test_report_includes_decision_specific_action_and_both_disclaimers() -> None:
    report = ReportGenerator().generate(**_report_inputs())

    assert "建议由具备相应资质的人员复核模型输出与技术记录" in report
    assert RISK_DISCLAIMER in report
    assert MEDICAL_DISCLAIMER in report


def test_report_displays_risk_weights_and_score_contributions() -> None:
    report = ReportGenerator().generate(**_report_inputs())
    risk_section = report.split("## 风险评分组成", 1)[1].split(
        "## 建议操作",
        1,
    )[0]

    assert "置信度风险：0.1877；权重 25.00%；贡献 4.69 分" in risk_section
    assert "预测不稳定性：0.5000；权重 35.00%；贡献 17.50 分" in risk_section
    assert "解释风险：0.2550；权重 25.00%；贡献 6.38 分" in risk_section
    assert "质量风险：0.1750；权重 15.00%；贡献 2.62 分" in risk_section
    assert RISK_DISCLAIMER in risk_section


def test_canonical_risk_disclaimer_cannot_be_replaced_by_input() -> None:
    inputs = _report_inputs()
    risk = dict(inputs["risk"])
    risk["disclaimer"] = "调用方替代文本"
    inputs["risk"] = risk

    report = ReportGenerator().generate(**inputs)

    assert RISK_DISCLAIMER in report
    assert "调用方替代文本" not in report


def test_report_marks_demo_results_but_not_real_model_results() -> None:
    demo_report = ReportGenerator().generate(**_report_inputs(is_demo=True))
    real_report = ReportGenerator().generate(**_report_inputs(is_demo=False))

    assert DEMO_DISCLAIMER in demo_report
    assert DEMO_DISCLAIMER not in real_report


def test_each_risk_decision_has_a_distinct_procedural_action() -> None:
    reports: dict[str, str] = {}
    for decision, score in (
        ("通过", 12.0),
        ("谨慎使用", 44.0),
        ("建议人工复核", 76.0),
    ):
        inputs = _report_inputs()
        risk = dict(inputs["risk"])
        risk.update(decision=decision, score=score)
        inputs["risk"] = risk
        reports[decision] = ReportGenerator().generate(**inputs)

    assert "按“通过”决策记录本次结果" in reports["通过"]
    assert "未触发复核阈值" not in reports["通过"]
    assert "核对输入质量与扰动结果" in reports["谨慎使用"]
    assert "具备相应资质的人员复核" in reports["建议人工复核"]


def test_report_accepts_quality_and_similarity_service_output_shapes() -> None:
    inputs = _report_inputs()
    inputs["quality"] = {
        "quality_score": 79.25,
        "laplacian_variance": 101.5,
        "mean_brightness": 0.4,
        "contrast_std": 0.16,
        "extreme_pixel_ratio": 0.02,
        "issues": [],
    }
    inputs["comparisons"] = [
        {
            "perturbation_name": "亮度调整",
            "prediction": {
                "class_name": "演示类别B",
                "confidence": 0.7,
            },
            "class_unchanged": True,
            "confidence_change": -0.11234,
            "cosine": 0.88,
            "ssim": 0.82,
        }
    ]

    report = ReportGenerator().generate(**inputs)

    assert "79.25/100" in report
    assert "清晰度方差：101.5000" in report
    assert "对比度：0.1600" in report
    assert "亮度调整：余弦相似度 0.8800；SSIM 0.8200" in report


def test_report_does_not_replace_missing_measurements_with_zeroes() -> None:
    report = ReportGenerator().generate(
        quality={"issues": []},
        prediction={"is_demo": False},
        comparisons=[],
        risk={"decision": "通过", "disclaimer": RISK_DISCLAIMER},
    )

    assert "风险分数：未提供" in report
    assert "原始置信度：未提供" in report
    assert "质量分：未提供" in report
