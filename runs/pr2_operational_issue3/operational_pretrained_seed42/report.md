# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 0/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 3/4
- Dynamic concepts above raw-feature baseline: 0/6
- Layer 11 wins on coordination and cardinality: True
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 0

## Best BOOM Probe Views

- `cardinality_bucket`: layer 11, `all_context`, `series_mean`; test_accuracy=0.455, baseline_test_accuracy=0.787
- `coordination`: layer 11, `all_context`, `series_mean`; test_r2=0.035, baseline_test_r2=0.070, shuffled_test_r2=-0.044
- `current_burstiness`: layer 5, `all_context`, `per_variate`; test_r2=0.072, baseline_test_r2=0.450, shuffled_test_r2=-0.062
- `current_sparsity`: layer -1, `all_context`, `series_mean`; test_r2=-0.076, baseline_test_r2=1.000, shuffled_test_r2=-0.093
- `domain`: layer 6, `all_context`, `series_mean`; test_accuracy=0.571, baseline_test_accuracy=0.389
- `frequency_bucket`: layer 7, `first_decode`, `series_mean`; test_accuracy=0.841, baseline_test_accuracy=0.475
- `future_burstiness`: layer 10, `all_context`, `series_mean`; test_r2=0.078, baseline_test_r2=0.093, shuffled_test_r2=-0.095
- `future_sparsity`: layer -1, `all_context`, `series_mean`; test_r2=-0.076, baseline_test_r2=1.000, shuffled_test_r2=-0.093
- `metric_type`: layer 11, `all_context`, `series_mean`; test_accuracy=0.574, baseline_test_accuracy=0.344
- `shift_risk`: layer 10, `final_context`, `per_variate`; test_r2=0.027, baseline_test_r2=0.015, shuffled_test_r2=-0.019

## Localization And Geometry

- Best probes concentrate as follows: layer -1: 2 best probes, layer 5: 1 best probes, layer 6: 1 best probes, layer 7: 1 best probes, layer 10: 2 best probes, layer 11: 3 best probes.
- Future-facing dynamic concepts average at layer 7.50.

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=-0.846, corr(|strength|, forecast_change)=0.989
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.399, corr(|strength|, forecast_change)=0.746
- `future_sparsity` (steer): monotonic=False, corr(strength, probe_score)=0.424, corr(|strength|, forecast_change)=0.478
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=-0.347, corr(|strength|, forecast_change)=0.990

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-1.210, baseline_transfer_r2=-4623.239, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9716714.784, baseline_transfer_r2=-9900156.868, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-0.230, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8268192.233, baseline_transfer_r2=-8517400.713, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-0.223, baseline_transfer_r2=0.275, datasets=3
- `fev` / `shift_risk`: transfer_r2=-18091474.658, baseline_transfer_r2=-17742149.642, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
