# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 2/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 3/4
- Dynamic concepts above raw-feature baseline: 2/6
- Layer 11 wins on coordination and cardinality: True
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 0

## Best BOOM Probe Views

- `cardinality_bucket`: layer 11, `all_context`, `series_mean`; test_accuracy=0.496, baseline_test_accuracy=0.763
- `coordination`: layer 11, `all_context`, `series_mean`; test_r2=0.047, baseline_test_r2=0.038, shuffled_test_r2=-0.062
- `current_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.143, baseline_test_r2=0.425, shuffled_test_r2=-0.011
- `current_sparsity`: layer 11, `first_decode`, `per_variate`; test_r2=0.347, baseline_test_r2=1.000, shuffled_test_r2=-0.101
- `domain`: layer 11, `all_context`, `series_mean`; test_accuracy=0.561, baseline_test_accuracy=0.411
- `frequency_bucket`: layer 5, `first_decode`, `series_mean`; test_accuracy=0.872, baseline_test_accuracy=0.474
- `future_burstiness`: layer 11, `all_context`, `series_mean`; test_r2=0.108, baseline_test_r2=0.048, shuffled_test_r2=-0.052
- `future_sparsity`: layer 11, `first_decode`, `per_variate`; test_r2=0.346, baseline_test_r2=1.000, shuffled_test_r2=-0.101
- `metric_type`: layer 10, `all_context`, `series_mean`; test_accuracy=0.618, baseline_test_accuracy=0.420
- `shift_risk`: layer 8, `final_context`, `per_variate`; test_r2=0.124, baseline_test_r2=-0.026, shuffled_test_r2=-0.515

## Localization And Geometry

- Best probes concentrate as follows: layer 5: 1 best probes, layer 8: 1 best probes, layer 10: 1 best probes, layer 11: 7 best probes.
- Future-facing dynamic concepts average at layer 10.25.

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=-0.906, corr(|strength|, forecast_change)=0.686
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.564, corr(|strength|, forecast_change)=0.762
- `future_sparsity` (steer): monotonic=False, corr(strength, probe_score)=1.000, corr(|strength|, forecast_change)=0.487
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=-0.726, corr(|strength|, forecast_change)=0.735

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-0.959, baseline_transfer_r2=-7164.140, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9866307.707, baseline_transfer_r2=-9874846.207, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-0.547, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8428343.923, baseline_transfer_r2=-8556833.481, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-0.693, baseline_transfer_r2=0.277, datasets=3
- `fev` / `shift_risk`: transfer_r2=-18214533.153, baseline_transfer_r2=-18498853.502, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
