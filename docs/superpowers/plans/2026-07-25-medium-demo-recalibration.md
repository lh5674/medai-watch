# Medium Demo Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the deterministic medium-risk demo so it shows a material confidence change without a class change and computes the caution decision band.

**Architecture:** Keep the predictor, Grad-CAM implementation, risk engine, and persistence unchanged. Reconfigure the synthetic medium case to run one deterministic Gaussian-noise stress test, expand the Streamlit slider to represent that preset, and update tests and documentation to match the computed behavior.

**Tech Stack:** Python 3.14, Streamlit, PyTorch, Pillow, NumPy, pytest.

---

The directory is not a Git repository. Do not initialize one as part of this repair; use test checkpoints instead of commit steps.

### Task 1: Lock The Correct Demo Contract In Tests

**Files:**
- Modify: `tests/test_demo_cases.py`
- Modify: `tests/test_app_imports.py`
- Test: `tests/test_monitoring_agent.py`

- [ ] **Step 1: Update the expected medium metadata**

Change `test_demo_cases_are_clearly_labeled_and_complete` to expect:

```python
assert [case["id"] for case in cases] == [
    "stable_low_risk",
    "confidence_shift_medium_risk",
    "prediction_change_high_risk",
]
assert [case["name"] for case in cases] == [
    "稳定低风险演示",
    "置信度变化中风险演示",
    "预测变化高风险演示",
]
```

- [ ] **Step 2: Update the computed preset assertions**

Replace the medium-equals-stable assertion in
`test_demo_cases_define_computed_workflow_presets` with:

```python
assert medium["recommended_tests"] == ["gaussian_noise"]
assert medium["recommended_parameters"] == {
    "gaussian_noise": {"sigma": 0.25, "seed": 42}
}
```

Keep the stable and high-risk assertions unchanged.

- [ ] **Step 3: Lock the Streamlit range required by the preset**

Add these assertions to `test_analysis_page_contains_required_chinese_controls`:

```python
assert "max_value=0.30" in source
assert 'value=float(noise_defaults["sigma"])' in source
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```powershell
$env:TEMP = "E:\medai-watch\.runtime\tmp"
$env:TMP = $env:TEMP
E:\medai-watch\.venv\Scripts\python.exe -m pytest tests/test_demo_cases.py tests/test_app_imports.py tests/test_monitoring_agent.py::test_demo_presets_cover_all_three_decision_bands -q
```

Expected: failures show the old medium ID/name, the three-test preset, the
`0.20` slider maximum, and the existing class-changing medium behavior.

### Task 2: Recalibrate The Demo And UI

**Files:**
- Modify: `services/demo_cases.py`
- Modify: `app.py`

- [ ] **Step 1: Rename the medium pattern helper**

Rename `_confidence_drop_pattern` to `_confidence_shift_pattern` without
changing its deterministic pixel-generation formula.

- [ ] **Step 2: Replace the medium case metadata and preset**

Use this exact medium case entry in `get_demo_cases()`:

```python
{
    "id": "confidence_shift_medium_risk",
    "name": "置信度变化中风险演示",
    "description": "用于演示低对比度合成输入在噪声扰动下的置信度与解释变化。",
    "image": _confidence_shift_pattern(),
    "simulated": True,
    "disclaimer": DEMO_DISCLAIMER,
    "recommended_tests": ["gaussian_noise"],
    "recommended_parameters": {
        "gaussian_noise": {"sigma": 0.25, "seed": 42}
    },
}
```

Do not add prediction, confidence, similarity, score, or decision fields.

- [ ] **Step 3: Expand the interactive noise range**

Change the Gaussian-noise slider in `app.py` to:

```python
sigma = st.sidebar.slider(
    "噪声强度 σ",
    min_value=0.0,
    max_value=0.30,
    value=float(noise_defaults["sigma"]),
    step=0.01,
    key=f"noise_sigma_{preset_id}",
)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Task 1 Step 4.

Expected: all focused tests pass. The workflow regression computes `通过`,
`谨慎使用`, and `建议人工复核` in order; the medium perturbation keeps its
class and changes confidence by at least `0.05`.

### Task 3: Update Documentation And Verify The Application

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Correct the demo-flow documentation**

Replace the current two-line preset description with:

```markdown
默认演示模型下，“稳定低风险”使用三种默认扰动；
“置信度变化中风险”使用噪声强度 `0.25`、随机种子 `42` 的单项压力测试；
“预测变化高风险”使用亮度系数 `0.20` 的单项压力测试。三档决策均由当前
固定权重演示模型实时计算，案例中没有预存预测、置信度或风险分数；更换模型、
参数或评分配置后，实际决策可能变化。
```

- [ ] **Step 2: Run the complete test suite**

Run:

```powershell
$env:TEMP = "E:\medai-watch\.runtime\tmp"
$env:TMP = $env:TEMP
E:\medai-watch\.venv\Scripts\python.exe -m pytest -q
```

Expected: `156 passed` and zero failures after the review-hardening test.

- [ ] **Step 3: Run the end-to-end sample verifier**

Run:

```powershell
E:\medai-watch\.venv\Scripts\python.exe E:\medai-watch\scripts\verify_sample.py
```

Expected: exit code 0, a saved record ID, a computed decision, three persisted
perturbations for the verifier's default workflow, and a non-empty dashboard
summary.

- [ ] **Step 4: Run a headless Streamlit health check**

Start the app on an available local port, poll `/_stcore/health` until it returns
`ok`, then stop only the process started by this check.

Expected: the health endpoint returns `ok` and the process exits cleanly after
verification.
