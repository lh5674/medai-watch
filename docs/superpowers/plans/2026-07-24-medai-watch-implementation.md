# MedAI Watch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, deterministic Streamlit prototype that monitors prediction and Grad-CAM stability under controlled perturbations, stores results in SQLite, and reports only computed non-clinical findings in Chinese.

**Architecture:** A small orchestration class composes focused predictor, quality, perturbation, Grad-CAM, similarity, risk, report, asset, and database services. Configuration is centralized in dataclasses; Streamlit obtains a cached application container and uses the same services as tests and CLI verification.

**Tech Stack:** Python 3.11+, Streamlit, PyTorch, torchvision, OpenCV, Pillow, NumPy, Pandas, scikit-image, Plotly, SQLite, pytest.

---

### Task 1: Project configuration and package skeleton

**Files:**
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `agent/__init__.py`
- Create: `services/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Define test fixtures**

Create deterministic RGB and grayscale PIL fixtures and a temporary settings fixture in `tests/conftest.py`.

- [ ] **Step 2: Add typed configuration**

Define frozen dataclasses for model, preprocessing, quality, perturbation, risk, storage, and application settings. Risk weights must sum to 1 and thresholds must satisfy `0 <= pass_threshold < caution_threshold <= 100`.

- [ ] **Step 3: Add dependency and ignore manifests**

Pin minimum compatible versions for the required technology and exclude virtual environments, caches, SQLite runtime files, generated assets, and model weights while retaining directory marker files.

- [ ] **Step 4: Verify imports**

Run:

```powershell
python -c "from config.settings import get_settings; print(get_settings().risk.weights)"
```

Expected: exit 0 and a dictionary whose values sum to 1.

### Task 2: Perturbations and image-quality checks

**Files:**
- Create: `tests/test_perturbations.py`
- Create: `tests/test_image_quality.py`
- Create: `services/perturbations.py`
- Create: `services/image_quality.py`

- [ ] **Step 1: Write perturbation tests**

Cover RGB/grayscale size and mode preservation, deterministic Gaussian noise, odd blur-kernel validation, sigma validation, and brightness factor validation.

- [ ] **Step 2: Run tests and confirm RED**

Run `python -m pytest tests/test_perturbations.py -q`.

Expected: collection failure because `services.perturbations` does not exist.

- [ ] **Step 3: Implement perturbations**

Implement `add_gaussian_blur`, `add_gaussian_noise`, `adjust_brightness`, and `apply_perturbation` using safe float conversion, clipping, deterministic NumPy RNG, and mode restoration.

- [ ] **Step 4: Run perturbation tests and confirm GREEN**

Run the command from Step 2. Expected: all perturbation tests pass.

- [ ] **Step 5: Write image-quality tests**

Cover valid RGB/grayscale images, small dimensions, black/white extreme values, finite normalized scores, corrupted byte streams through `load_image`, and unsupported input types.

- [ ] **Step 6: Run tests and confirm RED**

Run `python -m pytest tests/test_image_quality.py -q`.

Expected: collection failure because `services.image_quality` does not exist.

- [ ] **Step 7: Implement quality service**

Implement safe PIL loading, RGB/grayscale conversion for metrics, Laplacian variance, mean brightness, contrast, extreme-pixel ratio, configurable issue thresholds, and a bounded weighted quality score.

- [ ] **Step 8: Run tests and confirm GREEN**

Run the command from Step 6. Expected: all image-quality tests pass.

### Task 3: Similarity and transparent risk engine

**Files:**
- Create: `tests/test_similarity.py`
- Create: `tests/test_risk_engine.py`
- Create: `services/similarity.py`
- Create: `services/risk_engine.py`

- [ ] **Step 1: Write similarity tests**

Cover identical heatmaps, unrelated heatmaps, constant equal/unequal heatmaps, empty arrays, mismatched sizes, finite results, and `[0, 1]` boundaries.

- [ ] **Step 2: Run similarity tests and confirm RED**

Run `python -m pytest tests/test_similarity.py -q`.

Expected: collection failure because `services.similarity` does not exist.

- [ ] **Step 3: Implement comparison metrics**

Implement safe heatmap normalization, cosine similarity, SSIM with automatic odd `win_size`, and `compare_results` returning JSON-safe Python scalars.

- [ ] **Step 4: Run similarity tests and confirm GREEN**

Run the command from Step 2. Expected: all similarity tests pass.

- [ ] **Step 5: Write risk tests**

Cover zero and maximum risk inputs, exact decision thresholds at 30 and 60, no-perturbation validation, component exposure, disclaimer presence, and score boundaries.

- [ ] **Step 6: Run risk tests and confirm RED**

Run `python -m pytest tests/test_risk_engine.py -q`.

Expected: collection failure because `services.risk_engine` does not exist.

- [ ] **Step 7: Implement risk engine**

Compute the four configured components, weighted contributions, bounded 0–100 score, decision label/icon, and mandatory prototype disclaimer.

- [ ] **Step 8: Run risk tests and confirm GREEN**

Run the command from Step 6. Expected: all risk tests pass.

### Task 4: Model adapter and Grad-CAM

**Files:**
- Create: `tests/test_predictor.py`
- Create: `tests/test_gradcam.py`
- Create: `services/predictor.py`
- Create: `services/gradcam_service.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Write predictor tests**

Cover deterministic demo predictions, JSON-safe output, probability sum, input validation, model load-once behavior, missing real checkpoint errors, incompatible checkpoint errors, and configured class-count mismatch.

- [ ] **Step 2: Run predictor tests and confirm RED**

Run `python -m pytest tests/test_predictor.py -q`.

Expected: collection failure because `services.predictor` does not exist.

- [ ] **Step 3: Implement predictor**

Implement fixed-weight `DemoConvNet`, torchvision classifier construction, strict checkpoint loading, exact configured preprocessing, device selection, `eval()`, `torch.inference_mode()`, timing, model status, and serialized prediction output.

- [ ] **Step 4: Run predictor tests and confirm GREEN**

Run the command from Step 2. Expected: all predictor tests pass.

- [ ] **Step 5: Write Grad-CAM tests**

Cover automatic final-convolution discovery, explicit dotted-layer resolution, normalized output shapes, overlay shape, original target class use, and clear errors for models without convolutional layers.

- [ ] **Step 6: Run Grad-CAM tests and confirm RED**

Run `python -m pytest tests/test_gradcam.py -q`.

Expected: collection failure because `services.gradcam_service` does not exist.

- [ ] **Step 7: Implement custom Grad-CAM**

Capture activations and gradients with hooks, run a target-class backward pass, average spatial gradients, apply ReLU, resize and normalize safely, generate a colored overlay, and always remove hooks.

- [ ] **Step 8: Run Grad-CAM tests and confirm GREEN**

Run the command from Step 6. Expected: all Grad-CAM tests pass.

### Task 5: Reporting, assets, and SQLite

**Files:**
- Create: `tests/test_report_generator.py`
- Create: `tests/test_database.py`
- Create: `services/report_generator.py`
- Create: `services/assets.py`
- Create: `services/database.py`
- Create: `data/.gitkeep`
- Create: `data/sample_images/.gitkeep`

- [ ] **Step 1: Write report tests**

Assert all eight Chinese sections, computed-only descriptions, perturbation names, decision-specific action, risk disclaimer, medical disclaimer, and simulation label when `is_demo` is true.

- [ ] **Step 2: Run report tests and confirm RED**

Run `python -m pytest tests/test_report_generator.py -q`.

Expected: collection failure because `services.report_generator` does not exist.

- [ ] **Step 3: Implement deterministic report generator**

Build text solely from supplied quality, prediction, comparison, and risk dictionaries; do not infer disease meaning or treatment.

- [ ] **Step 4: Run report tests and confirm GREEN**

Run the command from Step 2. Expected: all report tests pass.

- [ ] **Step 5: Write storage tests**

Use temporary folders to verify schema initialization, foreign keys, parameterized insertion, rollback, single-record retrieval, filtered history, dashboard summary, perturbation failure distribution, and persisted asset files.

- [ ] **Step 6: Run storage tests and confirm RED**

Run `python -m pytest tests/test_database.py -q`.

Expected: collection failure because storage modules do not exist.

- [ ] **Step 7: Implement asset and database repositories**

Encode PIL/NumPy images to PNG, allocate UUID case directories, initialize both requested tables and indexes, store a complete analysis transaction, and expose dashboard/history query methods returning dictionaries or DataFrames.

- [ ] **Step 8: Run storage tests and confirm GREEN**

Run the command from Step 6. Expected: all storage tests pass.

### Task 6: Agent workflow and demo cases

**Files:**
- Create: `tests/test_monitoring_agent.py`
- Create: `tests/test_demo_cases.py`
- Create: `agent/monitoring_agent.py`
- Create: `services/demo_cases.py`

- [ ] **Step 1: Write workflow tests**

Cover exact phase order, selected perturbations, original target class passed to every Grad-CAM call, serializable export payload, report generation, saved record ID, and meaningful failure when no test is selected.

- [ ] **Step 2: Run workflow tests and confirm RED**

Run `python -m pytest tests/test_monitoring_agent.py tests/test_demo_cases.py -q`.

Expected: collection failure because workflow modules do not exist.

- [ ] **Step 3: Implement deterministic demo sample generation**

Generate three fixed synthetic phantom-style images with labels “稳定低风险演示”, “置信度变化中风险演示”, and “预测变化高风险演示”; every metadata record must carry the simulation disclaimer. The medium case uses one deterministic Gaussian-noise test with `sigma=0.25` and `seed=42`.

- [ ] **Step 4: Implement monitoring workflow**

Compose quality, original prediction/CAM, each chosen perturbation and comparison, risk, report, asset persistence, database transaction, and export-safe payload. Accept progress callbacks but do not require Streamlit.

- [ ] **Step 5: Run workflow tests and confirm GREEN**

Run the command from Step 2. Expected: all workflow and demo tests pass.

### Task 7: Streamlit analysis and dashboard

**Files:**
- Create: `app.py`
- Create: `pages/1_单病例分析.py`
- Create: `pages/2_监控仪表盘.py`
- Create: `services/container.py`
- Create: `.streamlit/config.toml`
- Create: `tests/test_app_imports.py`

- [ ] **Step 1: Write import and UI-source tests**

Verify the container builds without Streamlit, Chinese disclaimer strings exist, all three decisions include icon plus text, and export helpers return JSON/CSV bytes.

- [ ] **Step 2: Run UI tests and confirm RED**

Run `python -m pytest tests/test_app_imports.py -q`.

Expected: failure because app modules do not exist.

- [ ] **Step 3: Implement cached service container**

Create one settings, predictor, Grad-CAM service, database, and agent instance per Streamlit resource cache. Real-model initialization errors become visible model status and never trigger simulated fallback.

- [ ] **Step 4: Implement Chinese single-case page**

Add wide layout, disclaimer, model card, uploader/sample selector, perturbation controls, progress callback, prediction/Grad-CAM/perturbation displays, metrics, risk visualization, report, and JSON/CSV downloads.

- [ ] **Step 5: Implement Chinese dashboard**

Add requested KPIs, filters, Plotly distributions, recent table, high-risk cases, and expandable report/perturbation details.

- [ ] **Step 6: Run UI tests and confirm GREEN**

Run the command from Step 2. Expected: all UI tests pass.

### Task 8: Documentation and full verification

**Files:**
- Create: `README.md`
- Create: `scripts/verify_sample.py`
- Modify: any file implicated by verification failures

- [ ] **Step 1: Write Chinese documentation**

Document all 16 requested sections, exact install/start commands, real checkpoint placement at `models/model.pth`, label and preprocessing settings in `config/settings.py`, Grad-CAM layer configuration, demo-mode warnings, and limitations.

- [ ] **Step 2: Add executable sample verifier**

Run one generated sample through all enabled perturbations, assert finite/bounded metrics, assert SQLite insertion and retrieval, query dashboard summaries, and print the record ID and decision.

- [ ] **Step 3: Provision Python and dependencies**

Create `.venv` using Python 3.11 and install `requirements.txt`.

- [ ] **Step 4: Run dependency/import checks**

Run `python -c "import streamlit, torch, torchvision, cv2, PIL, numpy, pandas, skimage, plotly; import app"`.

Expected: exit 0.

- [ ] **Step 5: Run full unit tests**

Run `python -m pytest -q`.

Expected: zero failures.

- [ ] **Step 6: Run complete sample analysis and database checks**

Run `python scripts/verify_sample.py`.

Expected: exit 0, one saved analysis, three saved perturbation records, and a non-empty dashboard summary.

- [ ] **Step 7: Run application startup check**

Start Streamlit headlessly on an available localhost port, wait for `/_stcore/health` to return `ok`, then terminate the spawned process.

- [ ] **Step 8: Review requirements against evidence**

Inspect the final tree and test output, identify any unverified item explicitly, and report only claims supported by the fresh commands above.
