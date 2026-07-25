# MedAI Watch Medium Demo Recalibration

## Problem

The medium-risk synthetic demo is intended to demonstrate a material confidence
change without changing the predicted class. Its current image begins at only
about 51.58% confidence in a binary classifier. The configured brightness
factor of 0.85 crosses the 50% class boundary, while the reported confidence
change is only about 0.0034. The regression test therefore fails, and the demo
overlaps semantically with the high-risk class-change scenario.

## Scope

Recalibrate only the synthetic medium demo and the Streamlit control range
needed to load its preset. Do not change the fixed demo model, preprocessing,
risk formula, decision thresholds, persistence schema, report contract, or
real-checkpoint integration.

The medium scenario will use:

- ID: `confidence_shift_medium_risk`
- Display name: `置信度变化中风险演示`
- Tests: `gaussian_noise` only
- Noise parameters: `sigma=0.25`, `seed=42`

The noise slider will accept values from 0.00 through 0.30 in increments of
0.01 so the preset is valid when selected in Streamlit. The stable scenario
will retain all three default perturbations. The high-risk scenario will retain
its brightness-only test with factor 0.20.

## Expected Behavior

With the existing deterministic model and preprocessing:

- The stable scenario receives the `通过` decision and keeps its class under
  all recommended perturbations.
- The medium scenario receives the `谨慎使用` decision, keeps its original
  class under Gaussian noise, and has an absolute confidence change of at
  least 0.05.
- The high scenario receives the `建议人工复核` decision and changes class
  under its recommended brightness perturbation.

The medium score must come from measured confidence and Grad-CAM instability,
not from a class change or a hard-coded decision.

## Components And Data Flow

`services/demo_cases.py` remains the only source of synthetic case metadata
and recommended parameters. `app.py` reads those values into the existing
sidebar controls. `MedicalAIWatchAgent` continues to execute the selected
perturbation, compare predictions and Grad-CAM outputs, and pass those computed
measurements to the unchanged `RiskEngine`.

No prefabricated prediction, confidence, similarity, risk score, or decision
will be stored in demo metadata.

## Error Handling

Existing perturbation validation remains authoritative: Gaussian noise must
stay in the service-supported range `[0, 1]`, and the UI constrains interactive
input to `[0, 0.30]`. No fallback or forced decision is introduced.

## Documentation

Update the README demo-flow section to describe the medium scenario as a
confidence-change case driven by deterministic Gaussian noise at `sigma=0.25`.
Remove the statement that both stable and medium scenarios use all three
default perturbations.

## Verification

1. Update demo metadata tests to assert the new ID, display name, single noise
   test, and exact deterministic parameters.
2. Use the existing three-band workflow regression as the behavioral test.
3. Add UI-source assertions and a Streamlit `AppTest` proving that selecting
   the medium case loads `sigma=0.25` within the supported slider range.
4. Run the focused demo and monitoring-agent tests.
5. Run the complete test suite.
6. Run the sample verifier and a headless Streamlit health check.

## Acceptance Criteria

- All three demo scenarios produce their intended decision bands from live
  computation.
- The medium scenario has no class change and confidence change `>= 0.05`.
- The medium preset loads without a Streamlit slider range error.
- No production model, risk, storage, or report behavior changes.
- The complete automated test suite passes.
