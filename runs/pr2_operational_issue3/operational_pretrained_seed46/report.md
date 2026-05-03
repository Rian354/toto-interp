# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 1/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 3/4
- Dynamic concepts above raw-feature baseline: 1/6
- Layer 11 wins on coordination and cardinality: True
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 1

## Best BOOM Probe Views

- `cardinality_bucket`: layer 11, `all_context`, `series_mean`; test_accuracy=0.497, baseline_test_accuracy=0.775
- `coordination`: layer 11, `all_context`, `series_mean`; test_r2=0.100, baseline_test_r2=0.080, shuffled_test_r2=-0.085
- `current_burstiness`: layer 11, `all_context`, `series_mean`; test_r2=0.095, baseline_test_r2=0.631, shuffled_test_r2=-0.032
- `current_sparsity`: layer 11, `all_context`, `series_mean`; test_r2=0.060, baseline_test_r2=1.000, shuffled_test_r2=-0.039
- `domain`: layer 5, `first_decode`, `series_mean`; test_accuracy=0.530, baseline_test_accuracy=0.404
- `frequency_bucket`: layer 4, `first_decode`, `series_mean`; test_accuracy=0.862, baseline_test_accuracy=0.422
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.127, baseline_test_r2=0.013, shuffled_test_r2=-0.010
- `future_sparsity`: layer 11, `all_context`, `series_mean`; test_r2=0.060, baseline_test_r2=1.000, shuffled_test_r2=-0.039
- `metric_type`: layer 9, `final_context`, `series_mean`; test_accuracy=0.595, baseline_test_accuracy=0.367
- `shift_risk`: layer 10, `all_context`, `series_mean`; test_r2=0.095, baseline_test_r2=0.083, shuffled_test_r2=-0.030

## Localization And Geometry

- Best probes concentrate as follows: layer 4: 1 best probes, layer 5: 1 best probes, layer 9: 1 best probes, layer 10: 1 best probes, layer 11: 6 best probes.
- Future-facing dynamic concepts average at layer 10.75.

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=-0.183, corr(|strength|, forecast_change)=0.948
- `future_burstiness` (steer): monotonic=True, corr(strength, probe_score)=0.976, corr(|strength|, forecast_change)=0.749
- `future_sparsity` (steer): monotonic=False, corr(strength, probe_score)=-0.774, corr(|strength|, forecast_change)=0.994
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=0.365, corr(|strength|, forecast_change)=0.822

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-1.165, baseline_transfer_r2=-4602.924, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9817684.148, baseline_transfer_r2=-9900156.868, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-0.322, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8507902.457, baseline_transfer_r2=-8554915.472, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-0.339, baseline_transfer_r2=0.276, datasets=3
- `fev` / `shift_risk`: transfer_r2=-18441236.593, baseline_transfer_r2=-17091378.133, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
