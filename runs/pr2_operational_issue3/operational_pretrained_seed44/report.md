# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 1/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 3/4
- Dynamic concepts above raw-feature baseline: 1/6
- Layer 11 wins on coordination and cardinality: False
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 0

## Best BOOM Probe Views

- `cardinality_bucket`: layer 11, `all_context`, `series_mean`; test_accuracy=0.487, baseline_test_accuracy=0.751
- `coordination`: layer 7, `all_context`, `series_mean`; test_r2=0.068, baseline_test_r2=0.061, shuffled_test_r2=-0.045
- `current_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.069, baseline_test_r2=0.499, shuffled_test_r2=-0.012
- `current_sparsity`: layer 7, `all_context`, `series_mean`; test_r2=0.022, baseline_test_r2=1.000, shuffled_test_r2=-0.053
- `domain`: layer 7, `first_decode`, `series_mean`; test_accuracy=0.552, baseline_test_accuracy=0.338
- `frequency_bucket`: layer 5, `first_decode`, `series_mean`; test_accuracy=0.877, baseline_test_accuracy=0.375
- `future_burstiness`: layer 11, `all_context`, `series_mean`; test_r2=0.070, baseline_test_r2=0.174, shuffled_test_r2=-0.037
- `future_sparsity`: layer 7, `all_context`, `series_mean`; test_r2=0.022, baseline_test_r2=1.000, shuffled_test_r2=-0.053
- `metric_type`: layer 8, `first_decode`, `series_mean`; test_accuracy=0.613, baseline_test_accuracy=0.353
- `shift_risk`: layer 6, `final_context`, `per_variate`; test_r2=0.177, baseline_test_r2=0.006, shuffled_test_r2=-0.112

## Localization And Geometry

- Best probes concentrate as follows: layer 5: 1 best probes, layer 6: 1 best probes, layer 7: 4 best probes, layer 8: 1 best probes, layer 11: 3 best probes.
- Future-facing dynamic concepts average at layer 7.75.

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=0.303, corr(|strength|, forecast_change)=0.997
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.737, corr(|strength|, forecast_change)=0.849
- `future_sparsity` (steer): monotonic=False, corr(strength, probe_score)=0.247, corr(|strength|, forecast_change)=0.764
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=-0.690, corr(|strength|, forecast_change)=0.984

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-0.967, baseline_transfer_r2=-3920.586, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9839688.900, baseline_transfer_r2=-9874846.207, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-0.329, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8414794.741, baseline_transfer_r2=-8528499.194, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-0.347, baseline_transfer_r2=0.276, datasets=3
- `fev` / `shift_risk`: transfer_r2=-18340890.386, baseline_transfer_r2=-18498853.471, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
