from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import MEDICAL_DISCLAIMER, RISK_DISCLAIMER
from services.container import get_cached_container
from services.perturbations import PERTURBATION_DISPLAY_NAMES


st.set_page_config(
    page_title="MedAI Watch｜监控仪表盘",
    page_icon="📊",
    layout="wide",
)

RECENT_COLUMN_NAMES = {
    "id": "记录编号",
    "created_at": "创建时间",
    "image_name": "图像名称",
    "predicted_class": "预测类别",
    "original_confidence": "原始置信度",
    "quality_score": "图像质量分",
    "risk_score": "风险分",
    "decision": "决策",
    "model_version": "模型版本",
}

PERTURBATION_COLUMN_NAMES = {
    "id": "扰动记录编号",
    "analysis_id": "分析记录编号",
    "perturbation_name": "扰动名称",
    "changed_class": "扰动后类别",
    "changed_confidence": "扰动后置信度",
    "confidence_change": "置信度变化",
    "class_unchanged": "类别是否保持",
    "gradcam_cosine_similarity": "Grad-CAM 余弦相似度",
    "gradcam_ssim": "Grad-CAM SSIM",
    "perturbed_image_path": "扰动图像路径",
    "perturbed_gradcam_path": "扰动 Grad-CAM 路径",
}


def _date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def main() -> None:
    st.title("📊 监控仪表盘")
    st.warning(f"研究原型：{MEDICAL_DISCLAIMER} {RISK_DISCLAIMER}")
    try:
        repository = get_cached_container().database
    except Exception as exc:
        st.error(f"服务初始化失败：{exc}")
        st.stop()

    all_rows = repository.get_history(limit=1000)
    if all_rows:
        dates = [_date(row["created_at"]) for row in all_rows]
        default_range = (min(dates), max(dates))
        classes = sorted({row["predicted_class"] for row in all_rows})
    else:
        default_range = (date.today(), date.today())
        classes = []

    st.subheader("筛选条件")
    date_col, class_col, decision_col = st.columns(3)
    with date_col:
        selected_dates = st.date_input(
            "日期范围",
            value=default_range,
        )
    with class_col:
        selected_class = st.selectbox("预测类别", ["全部", *classes])
    with decision_col:
        selected_decision = st.selectbox(
            "决策",
            ["全部", "通过", "谨慎使用", "建议人工复核"],
        )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    filters = {
        "start_date": start_date,
        "end_date": end_date,
        "predicted_class": (
            None if selected_class == "全部" else selected_class
        ),
        "decision": (
            None if selected_decision == "全部" else selected_decision
        ),
    }

    summary = repository.get_dashboard_summary(**filters)
    metric_columns = st.columns(4)
    metric_columns[0].metric("分析总数", summary["total_count"])
    metric_columns[1].metric("高风险案例", summary["high_risk_count"])
    metric_columns[2].metric(
        "高风险比例",
        f"{summary['high_risk_rate']:.1%}",
    )
    metric_columns[3].metric(
        "平均风险分",
        f"{summary['average_risk_score']:.1f}/100",
    )

    distribution_col, failures_col = st.columns(2)
    decisions = repository.get_decision_distribution(**filters)
    with distribution_col:
        st.subheader("决策分布")
        if decisions:
            decision_frame = pd.DataFrame(decisions)
            figure = px.pie(
                decision_frame,
                names="decision",
                values="count",
                labels={"decision": "决策", "count": "数量"},
                color="decision",
                color_discrete_map={
                    "通过": "#2e7d32",
                    "谨慎使用": "#ef8c25",
                    "建议人工复核": "#c62828",
                },
                hole=0.45,
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("当前筛选条件下暂无分析记录。")

    failures = repository.get_perturbation_failure_distribution(**filters)
    with failures_col:
        st.subheader("扰动失败分布")
        if failures:
            failure_frame = pd.DataFrame(failures)
            failure_frame["扰动类型"] = failure_frame[
                "perturbation_name"
            ].map(PERTURBATION_DISPLAY_NAMES).fillna(
                failure_frame["perturbation_name"]
            )
            figure = px.bar(
                failure_frame,
                x="扰动类型",
                y="failure_count",
                color="failure_rate",
                color_continuous_scale=["#f5f5f5", "#c62828"],
                labels={
                    "failure_count": "类别变化次数",
                    "failure_rate": "类别变化率",
                },
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("当前筛选条件下暂无扰动结果。")

    st.subheader("最近分析记录")
    rows = repository.get_history(**filters, limit=100)
    if not rows:
        st.info("暂无符合条件的分析记录。")
        return
    display_columns = [
        "id",
        "created_at",
        "image_name",
        "predicted_class",
        "original_confidence",
        "quality_score",
        "risk_score",
        "decision",
        "model_version",
    ]
    st.dataframe(
        pd.DataFrame(rows)[display_columns].rename(
            columns=RECENT_COLUMN_NAMES
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("分析详情")
    for row in rows[:20]:
        label = (
            f"#{row['id']}｜{row['created_at']}｜"
            f"{row['decision']}｜风险 {row['risk_score']:.1f}"
        )
        with st.expander(label):
            detail = repository.get_analysis(row["id"])
            if detail is None:
                st.warning("记录不存在。")
                continue
            st.markdown(detail["report"])
            st.markdown("**扰动明细**")
            detail_frame = pd.DataFrame(detail["perturbations"])
            if not detail_frame.empty:
                detail_frame["perturbation_name"] = detail_frame[
                    "perturbation_name"
                ].map(PERTURBATION_DISPLAY_NAMES).fillna(
                    detail_frame["perturbation_name"]
                )
                detail_frame["class_unchanged"] = detail_frame[
                    "class_unchanged"
                ].map({True: "是", False: "否"})
                detail_frame = detail_frame.rename(
                    columns=PERTURBATION_COLUMN_NAMES
                )
            st.dataframe(
                detail_frame,
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
