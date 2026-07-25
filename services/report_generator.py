from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from config.settings import DEMO_DISCLAIMER, MEDICAL_DISCLAIMER, RISK_DISCLAIMER


_ACTIONS = {
    "通过": "按“通过”决策记录本次结果，并继续遵循既定研究流程。",
    "谨慎使用": "建议核对输入质量与扰动结果，并在研究流程中谨慎使用该输出。",
    "建议人工复核": "建议由具备相应资质的人员复核模型输出与技术记录。",
}


def _number(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _percentage(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _optional_percentage(value: Any | None) -> str:
    return "未提供" if value is None else _percentage(value)


def _score(value: Any | None) -> str:
    return "未提供" if value is None else f"{_number(value)}/100"


def _risk_score(risk: Mapping[str, Any]) -> Any | None:
    return risk.get("score", risk.get("risk_score"))


def _decision(risk: Mapping[str, Any]) -> str:
    return str(risk.get("decision") or risk.get("decision_text") or "未提供")


def _name(item: Mapping[str, Any]) -> str:
    return str(
        item.get("name")
        or item.get("perturbation_name")
        or item.get("type")
        or "未命名扰动"
    )


def _predicted_class(item: Mapping[str, Any]) -> str:
    prediction = item.get("prediction")
    if isinstance(prediction, Mapping):
        return str(
            prediction.get("class_name")
            or prediction.get("predicted_class")
            or prediction.get("label")
            or "未提供"
        )
    return str(
        item.get("predicted_class")
        or item.get("changed_class")
        or item.get("class_name")
        or "未提供"
    )


def _confidence(item: Mapping[str, Any]) -> Any | None:
    prediction = item.get("prediction")
    if isinstance(prediction, Mapping) and "confidence" in prediction:
        return prediction["confidence"]
    return item.get("confidence", item.get("changed_confidence"))


class ReportGenerator:
    """Build a deterministic, computed-facts-only Chinese research report."""

    def generate(
        self,
        quality: Mapping[str, Any],
        prediction: Mapping[str, Any],
        comparisons: Sequence[Mapping[str, Any]],
        risk: Mapping[str, Any],
    ) -> str:
        sections = [
            self._summary(prediction, comparisons, risk),
            self._prediction(prediction),
            self._quality(quality),
            self._perturbations(comparisons),
            self._explanations(comparisons),
            self._risk(risk),
            self._action(risk),
            self._disclaimers(prediction),
        ]
        return "\n\n".join(sections).rstrip() + "\n"

    def generate_report(
        self,
        quality: Mapping[str, Any],
        prediction: Mapping[str, Any],
        comparisons: Sequence[Mapping[str, Any]],
        risk: Mapping[str, Any],
    ) -> str:
        return self.generate(quality, prediction, comparisons, risk)

    @staticmethod
    def _summary(
        prediction: Mapping[str, Any],
        comparisons: Sequence[Mapping[str, Any]],
        risk: Mapping[str, Any],
    ) -> str:
        lines = [
            "## 检查摘要",
            f"- 模型版本：{prediction.get('model_version', '未提供')}",
            f"- 执行设备：{prediction.get('device', '未提供')}",
            f"- 已执行扰动数量：{len(comparisons)}",
            f"- 风险分数：{_score(_risk_score(risk))}",
            f"- 决策：{_decision(risk)}",
        ]
        if bool(prediction.get("is_demo", False)):
            lines.append(f"- 标记：{DEMO_DISCLAIMER}")
        return "\n".join(lines)

    @staticmethod
    def _prediction(prediction: Mapping[str, Any]) -> str:
        class_name = (
            prediction.get("class_name")
            or prediction.get("predicted_class")
            or prediction.get("label")
            or "未提供"
        )
        lines = [
            "## 原始模型结果",
            f"- 类别索引：{prediction.get('class_index', '未提供')}",
            f"- 预测类别：{class_name}",
            f"- 原始置信度：{_optional_percentage(prediction.get('confidence'))}",
        ]
        probabilities = prediction.get("probabilities")
        if isinstance(probabilities, Mapping):
            for name in sorted(probabilities, key=str):
                lines.append(f"- 类别概率（{name}）：{_percentage(probabilities[name])}")
        return "\n".join(lines)

    @staticmethod
    def _quality(quality: Mapping[str, Any]) -> str:
        lines = [
            "## 图像质量",
            f"- 质量分：{_score(quality.get('score', quality.get('quality_score')))}",
        ]
        metrics = quality.get("metrics")
        metric_values = metrics if isinstance(metrics, Mapping) else quality
        if isinstance(metric_values, Mapping):
            labels = {
                "blur_variance": "清晰度方差",
                "laplacian_variance": "清晰度方差",
                "mean_brightness": "平均亮度",
                "brightness": "平均亮度",
                "contrast": "对比度",
                "contrast_std": "对比度",
                "extreme_fraction": "极端像素占比",
                "extreme_pixel_ratio": "极端像素占比",
            }
            for key in (
                "blur_variance",
                "laplacian_variance",
                "mean_brightness",
                "brightness",
                "contrast",
                "contrast_std",
                "extreme_fraction",
                "extreme_pixel_ratio",
            ):
                if key in metric_values:
                    lines.append(
                        f"- {labels[key]}：{_number(metric_values[key], 4)}"
                    )
        issues = quality.get("issues")
        if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes)):
            lines.append(f"- 技术质量提示：{'；'.join(str(issue) for issue in issues) or '无'}")
        return "\n".join(lines)

    @staticmethod
    def _perturbations(comparisons: Sequence[Mapping[str, Any]]) -> str:
        lines = ["## 扰动测试结果"]
        if not comparisons:
            lines.append("- 未提供扰动测试结果。")
            return "\n".join(lines)
        for item in comparisons:
            confidence = _confidence(item)
            confidence_text = (
                _percentage(confidence) if confidence is not None else "未提供"
            )
            if "class_changed" in item:
                changed_text = "是" if bool(item["class_changed"]) else "否"
            elif "class_unchanged" in item:
                changed_text = "否" if bool(item["class_unchanged"]) else "是"
            else:
                changed_text = "未提供"
            delta = item.get("confidence_delta", item.get("confidence_change"))
            delta_text = _number(delta, 4) if delta is not None else "未提供"
            lines.append(
                f"- {_name(item)}：类别 {_predicted_class(item)}；置信度 {confidence_text}；"
                f"类别变化 {changed_text}；置信度变化 {delta_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _explanations(comparisons: Sequence[Mapping[str, Any]]) -> str:
        lines = ["## 解释稳定性"]
        if not comparisons:
            lines.append("- 未提供解释相似度结果。")
            return "\n".join(lines)
        for item in comparisons:
            cosine = item.get(
                "cosine_similarity",
                item.get("cosine", item.get("gradcam_cosine_similarity")),
            )
            ssim = item.get("ssim", item.get("gradcam_ssim"))
            values = [float(value) for value in (cosine, ssim) if value is not None]
            average = sum(values) / len(values) if values else None
            cosine_text = _number(cosine, 4) if cosine is not None else "未提供"
            ssim_text = _number(ssim, 4) if ssim is not None else "未提供"
            average_text = _number(average, 4) if average is not None else "未提供"
            lines.append(
                f"- {_name(item)}：余弦相似度 {cosine_text}；SSIM {ssim_text}；"
                f"平均解释相似度 {average_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _risk(risk: Mapping[str, Any]) -> str:
        lines = [
            "## 风险评分组成",
            f"- 总分：{_score(_risk_score(risk))}",
            f"- 决策：{_decision(risk)}",
        ]
        components = risk.get("components")
        component_values = components if isinstance(components, Mapping) else risk
        weights = risk.get("weights")
        weight_values = weights if isinstance(weights, Mapping) else {}
        contributions = risk.get("contributions")
        contribution_values = (
            contributions if isinstance(contributions, Mapping) else {}
        )
        if isinstance(component_values, Mapping):
            labels = {
                "confidence_risk": "置信度风险",
                "prediction_instability": "预测不稳定性",
                "explanation_risk": "解释风险",
                "quality_risk": "质量风险",
            }
            for key in (
                "confidence_risk",
                "prediction_instability",
                "explanation_risk",
                "quality_risk",
            ):
                if key in component_values:
                    detail = (
                        f"- {labels[key]}：{_number(component_values[key], 4)}"
                    )
                    if key in weight_values:
                        detail += f"；权重 {_percentage(weight_values[key])}"
                    if key in contribution_values:
                        detail += (
                            f"；贡献 {_number(contribution_values[key], 2)} 分"
                        )
                    lines.append(detail)
        lines.append(f"- 评分声明：{RISK_DISCLAIMER}")
        return "\n".join(lines)

    @staticmethod
    def _action(risk: Mapping[str, Any]) -> str:
        decision = _decision(risk)
        action = _ACTIONS.get(
            decision,
            "请按既定研究流程记录结果，并由项目负责人确定后续技术核查步骤。",
        )
        return f"## 建议操作\n- 决策对应操作：{action}"

    @staticmethod
    def _disclaimers(prediction: Mapping[str, Any]) -> str:
        lines = [
            "## 原型免责声明",
            f"- 风险评分说明：{RISK_DISCLAIMER}",
            f"- 医疗使用限制：{MEDICAL_DISCLAIMER}",
        ]
        if bool(prediction.get("is_demo", False)):
            lines.append(f"- 模拟标记：{DEMO_DISCLAIMER}")
        return "\n".join(lines)


def generate_report(
    quality: Mapping[str, Any],
    prediction: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    risk: Mapping[str, Any],
) -> str:
    return ReportGenerator().generate(quality, prediction, comparisons, risk)
