from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from config.settings import (
    DEMO_DISCLAIMER,
    MEDICAL_DISCLAIMER,
    RISK_DISCLAIMER,
)
from services.container import get_cached_container
from services.demo_cases import get_demo_cases
from services.exports import analysis_csv_bytes, analysis_json_bytes
from services.image_quality import load_image


st.set_page_config(
    page_title="MedAI Watch｜单病例分析",
    page_icon="🩺",
    layout="wide",
)


DECISION_STYLES = {
    "通过": ("✅", "decision-pass"),
    "谨慎使用": ("⚠️", "decision-caution"),
    "建议人工复核": ("🔴", "decision-review"),
}


def _analysis_signature(
    image: Image.Image,
    image_name: str,
    selected_tests: list[str],
    parameters: dict[str, dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{image.mode}:{image.width}x{image.height}:".encode("utf-8")
    )
    digest.update(image.tobytes())
    digest.update(
        json.dumps(
            {
                "image_name": image_name,
                "selected_tests": selected_tests,
                "parameters": parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
        .prototype-banner {
            border: 1px solid #c9a227; background: #fff9df; color: #4b3b00;
            border-radius: 12px; padding: 1rem 1.1rem; margin: .5rem 0 1.2rem;
        }
        .status-card {
            border: 1px solid #d9e2e8; border-radius: 12px;
            padding: .9rem 1rem; background: #f8fbfc;
        }
        .decision-card {
            border-radius: 14px; padding: 1rem 1.2rem; font-size: 1.35rem;
            font-weight: 700; border: 2px solid transparent;
        }
        .decision-pass {background:#e8f5e9; color:#176b2c; border-color:#66a974;}
        .decision-caution {background:#fff3e0; color:#8a4b00; border-color:#e7a64b;}
        .decision-review {background:#ffebee; color:#9f1d2d; border-color:#d86270;}
        .metric-note {color:#536771; font-size:.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_disclaimer() -> None:
    st.markdown(
        f"""
        <div class="prototype-banner">
        <strong>⚠️ 黑客松研究原型</strong><br>
        {MEDICAL_DISCLAIMER}<br>
        {RISK_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_status(container: Any) -> None:
    status = container.predictor.get_status()
    st.sidebar.subheader("模型状态")
    st.sidebar.markdown(
        f"""
        <div class="status-card">
        <strong>{'模拟演示模式' if status['is_demo'] else '真实检查点模式'}</strong><br>
        版本：{status['version']}<br>
        架构：{status['architecture']}<br>
        设备：{status['device']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(status["message"])


def _input_image() -> tuple[
    Image.Image | None,
    str,
    bool,
    dict[str, Any] | None,
]:
    uploaded = st.file_uploader(
        "上传医学影像",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="支持灰度或 RGB 图像；系统只进行技术稳定性监测。",
    )
    demo_cases = get_demo_cases()
    names = [case["name"] for case in demo_cases]
    selected_name = st.selectbox("或选择合成演示案例", names)
    selected_case = demo_cases[names.index(selected_name)]

    if uploaded is not None:
        try:
            return load_image(uploaded.getvalue()), uploaded.name, False, None
        except (TypeError, ValueError) as exc:
            st.error(f"图像无法读取：{exc}")
            return None, uploaded.name, False, None

    st.info(
        "模拟演示案例，不代表真实模型或临床结果。 "
        f"{selected_case['description']}"
    )
    return (
        selected_case["image"],
        f"{selected_case['id']}.png",
        True,
        selected_case,
    )


def _analysis_controls(
    preset_case: dict[str, Any] | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    st.sidebar.subheader("扰动测试配置")
    preset_id = preset_case["id"] if preset_case else "uploaded_image"
    recommended_tests = (
        preset_case["recommended_tests"]
        if preset_case
        else ["gaussian_blur", "gaussian_noise", "brightness"]
    )
    recommended_parameters = (
        preset_case["recommended_parameters"] if preset_case else {}
    )
    blur_defaults = recommended_parameters.get(
        "gaussian_blur",
        {"kernel_size": 5},
    )
    noise_defaults = recommended_parameters.get(
        "gaussian_noise",
        {"sigma": 0.02, "seed": 42},
    )
    brightness_defaults = recommended_parameters.get(
        "brightness",
        {"factor": 0.85},
    )
    if preset_case:
        st.sidebar.caption("已载入当前演示案例的推荐测试配置，可手动调整。")

    selected: list[str] = []
    if st.sidebar.checkbox(
        "高斯模糊",
        value="gaussian_blur" in recommended_tests,
        key=f"blur_enabled_{preset_id}",
    ):
        selected.append("gaussian_blur")
    if st.sidebar.checkbox(
        "高斯噪声",
        value="gaussian_noise" in recommended_tests,
        key=f"noise_enabled_{preset_id}",
    ):
        selected.append("gaussian_noise")
    if st.sidebar.checkbox(
        "亮度调整",
        value="brightness" in recommended_tests,
        key=f"brightness_enabled_{preset_id}",
    ):
        selected.append("brightness")

    kernel = st.sidebar.select_slider(
        "模糊核大小",
        options=[3, 5, 7, 9, 11],
        value=int(blur_defaults["kernel_size"]),
        key=f"blur_kernel_{preset_id}",
    )
    sigma = st.sidebar.slider(
        "噪声强度 σ",
        min_value=0.0,
        max_value=0.30,
        value=float(noise_defaults["sigma"]),
        step=0.01,
        key=f"noise_sigma_{preset_id}",
    )
    seed = st.sidebar.number_input(
        "噪声随机种子",
        min_value=0,
        max_value=999999,
        value=int(noise_defaults["seed"]),
        step=1,
        key=f"noise_seed_{preset_id}",
    )
    factor = st.sidebar.slider(
        "亮度系数",
        min_value=0.20,
        max_value=2.00,
        value=float(brightness_defaults["factor"]),
        step=0.05,
        key=f"brightness_factor_{preset_id}",
    )
    return selected, {
        "gaussian_blur": {"kernel_size": int(kernel)},
        "gaussian_noise": {"sigma": float(sigma), "seed": int(seed)},
        "brightness": {"factor": float(factor)},
    }


def _probability_chart(prediction: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        {
            "类别": list(prediction["probabilities"]),
            "概率": list(prediction["probabilities"].values()),
        }
    )
    figure = px.bar(
        frame,
        x="概率",
        y="类别",
        orientation="h",
        range_x=[0, 1],
        color="概率",
        color_continuous_scale="Blues",
        text=frame["概率"].map(lambda value: f"{value:.1%}"),
    )
    figure.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(figure, use_container_width=True)


def _risk_chart(risk: dict[str, Any]) -> None:
    labels = {
        "confidence_risk": "置信度风险",
        "prediction_instability": "预测不稳定性",
        "explanation_risk": "解释风险",
        "quality_risk": "图像质量风险",
    }
    component_keys = list(risk["components"])
    frame = pd.DataFrame(
        {
            "风险组成": [labels[key] for key in component_keys],
            "标准化风险": [
                risk["components"][key] for key in component_keys
            ],
            "权重": [risk["weights"][key] for key in component_keys],
            "分数贡献": [
                risk["contributions"][key] for key in component_keys
            ],
        }
    )
    figure = px.bar(
        frame,
        x="风险组成",
        y="分数贡献",
        range_y=[0, 35],
        color="标准化风险",
        color_continuous_scale=["#2e7d32", "#ef9a3d", "#c62828"],
        range_color=[0, 1],
        text=frame["分数贡献"].map(lambda value: f"{value:.2f} 分"),
        hover_data={
            "标准化风险": ":.4f",
            "权重": ":.2%",
            "分数贡献": ":.2f",
        },
    )
    figure.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "标准化风险": st.column_config.NumberColumn(format="%.4f"),
            "权重": st.column_config.NumberColumn(format="%.2f"),
            "分数贡献": st.column_config.NumberColumn(format="%.2f 分"),
        },
    )


def _render_results(
    result: dict[str, Any],
    original_image: Image.Image,
) -> None:
    st.divider()
    decision = result["risk"]["decision"]
    icon, css_class = DECISION_STYLES[decision]
    st.markdown(
        f'<div class="decision-card {css_class}">{icon} 决策：{decision}'
        f"　风险分：{result['risk']['risk_score']:.1f}/100</div>",
        unsafe_allow_html=True,
    )
    st.caption(RISK_DISCLAIMER)

    st.subheader("原始模型结果")
    left, center, right = st.columns([1, 1, 1])
    with left:
        st.image(original_image, caption="原始图像", use_container_width=True)
    with center:
        st.image(
            result["original_gradcam"]["overlay"],
            caption="原始 Grad-CAM（不代表医学正确性）",
            use_container_width=True,
        )
    with right:
        prediction = result["original_prediction"]
        st.metric("预测类别", prediction["class_name"])
        st.metric("模型置信度", f"{prediction['confidence']:.2%}")
        st.metric("图像质量分", f"{result['quality']['quality_score']:.1f}/100")
        _probability_chart(prediction)

    st.subheader("扰动测试对比")
    tabs = st.tabs(
        [
            {
                "gaussian_blur": "高斯模糊",
                "gaussian_noise": "高斯噪声",
                "brightness": "亮度调整",
            }[item["test_name"]]
            for item in result["perturbation_results"]
        ]
    )
    for tab, item in zip(tabs, result["perturbation_results"], strict=True):
        with tab:
            image_col, cam_col, metrics_col = st.columns([1, 1, 1])
            with image_col:
                st.image(
                    item["image"],
                    caption="扰动后图像",
                    use_container_width=True,
                )
            with cam_col:
                st.image(
                    item["gradcam"]["overlay"],
                    caption="扰动后 Grad-CAM",
                    use_container_width=True,
                )
            with metrics_col:
                comparison = item["comparison"]
                prediction = item["prediction"]
                st.metric("扰动后类别", prediction["class_name"])
                st.metric("扰动后置信度", f"{prediction['confidence']:.2%}")
                st.metric(
                    "类别是否保持",
                    "是" if comparison["class_unchanged"] else "否",
                )
                st.metric(
                    "置信度绝对变化",
                    f"{comparison['confidence_change']:.4f}",
                )
                st.metric(
                    "Grad-CAM 余弦相似度",
                    f"{comparison['gradcam_cosine_similarity']:.4f}",
                )
                st.metric(
                    "Grad-CAM SSIM",
                    f"{comparison['gradcam_ssim']:.4f}",
                )

    st.subheader("风险评分组成")
    _risk_chart(result["risk"])
    st.info(RISK_DISCLAIMER)

    st.subheader("中文监测报告")
    st.markdown(result["report"])
    if result.get("storage_error"):
        st.warning(f"分析已完成，但记录保存失败：{result['storage_error']}")
    else:
        st.success(f"分析记录已保存，记录编号：{result['record_id']}")

    json_col, csv_col = st.columns(2)
    with json_col:
        st.download_button(
            "下载 JSON",
            data=analysis_json_bytes(result),
            file_name=f"medai_watch_{result.get('record_id') or 'unsaved'}.json",
            mime="application/json",
            use_container_width=True,
        )
    with csv_col:
        st.download_button(
            "下载 CSV",
            data=analysis_csv_bytes(result),
            file_name=f"medai_watch_{result.get('record_id') or 'unsaved'}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    _inject_styles()
    st.title("🩺 MedAI Watch")
    st.subheader("单病例医学影像 AI 稳定性监测")
    _render_disclaimer()
    try:
        container = get_cached_container()
    except Exception as exc:
        st.error(f"模型或服务初始化失败：{exc}")
        st.stop()

    _render_model_status(container)
    image, image_name, is_sample, preset_case = _input_image()
    selected_tests, parameters = _analysis_controls(preset_case)
    current_signature = (
        _analysis_signature(
            image,
            image_name,
            selected_tests,
            parameters,
        )
        if image is not None and selected_tests
        else None
    )
    if image is not None:
        st.image(
            image,
            caption="待分析图像",
            width=360,
        )
    if is_sample:
        st.caption(DEMO_DISCLAIMER)

    if st.button(
        "开始安全监控分析",
        type="primary",
        use_container_width=True,
        disabled=image is None or not selected_tests,
    ):
        for key in (
            "analysis_result",
            "analysis_image",
            "analysis_signature",
        ):
            st.session_state.pop(key, None)
        progress = st.progress(0.0, text="准备分析")

        def update_progress(value: float, message: str) -> None:
            progress.progress(value, text=message)

        try:
            result = container.agent.run(
                image,
                image_name=image_name,
                selected_tests=selected_tests,
                perturbation_parameters=parameters,
                progress_callback=update_progress,
            )
        except Exception as exc:
            progress.empty()
            st.error(f"分析失败：{exc}")
        else:
            st.session_state["analysis_result"] = result
            st.session_state["analysis_image"] = image.copy()
            st.session_state["analysis_signature"] = current_signature
            progress.empty()

    if (
        current_signature is not None
        and st.session_state.get("analysis_signature") == current_signature
        and "analysis_result" in st.session_state
        and "analysis_image" in st.session_state
    ):
        _render_results(
            st.session_state["analysis_result"],
            st.session_state["analysis_image"],
        )


if __name__ == "__main__":
    main()
