# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 2/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 3/4
- Dynamic concepts above raw-feature baseline: 2/6
- Layer 11 wins on coordination and cardinality: False
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 0

## Best BOOM Probe Views

- `cardinality_bucket`: layer 11, `all_context`, `series_mean`; test_accuracy=0.462, baseline_test_accuracy=0.755
- `coordination`: layer 8, `all_context`, `series_mean`; test_r2=0.100, baseline_test_r2=0.104, shuffled_test_r2=-0.038
- `current_burstiness`: layer 9, `all_context`, `per_variate`; test_r2=0.162, baseline_test_r2=0.539, shuffled_test_r2=-0.005
- `current_sparsity`: layer 11, `all_context`, `series_mean`; test_r2=0.055, baseline_test_r2=1.000, shuffled_test_r2=-0.029
- `domain`: layer 9, `first_decode`, `series_mean`; test_accuracy=0.575, baseline_test_accuracy=0.444
- `frequency_bucket`: layer 4, `first_decode`, `series_mean`; test_accuracy=0.877, baseline_test_accuracy=0.525
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.219, baseline_test_r2=0.152, shuffled_test_r2=-0.038
- `future_sparsity`: layer 11, `all_context`, `series_mean`; test_r2=0.055, baseline_test_r2=1.000, shuffled_test_r2=-0.029
- `metric_type`: layer 5, `all_context`, `series_mean`; test_accuracy=0.608, baseline_test_accuracy=0.378
- `shift_risk`: layer 11, `all_context`, `per_variate`; test_r2=0.217, baseline_test_r2=-0.026, shuffled_test_r2=-0.024

## Localization And Geometry

- Best probes concentrate as follows: layer 4: 1 best probes, layer 5: 1 best probes, layer 8: 1 best probes, layer 9: 2 best probes, layer 11: 5 best probes.
- Future-facing dynamic concepts average at layer 10.25.

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=0.915, corr(|strength|, forecast_change)=0.976
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.452, corr(|strength|, forecast_change)=0.990
- `future_sparsity` (steer): monotonic=False, corr(strength, probe_score)=-0.022, corr(|strength|, forecast_change)=0.989
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=0.554, corr(|strength|, forecast_change)=0.996

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-0.820, baseline_transfer_r2=-7386.227, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9716969.021, baseline_transfer_r2=-9900156.868, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-0.316, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8482389.490, baseline_transfer_r2=-7898615.728, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-0.330, baseline_transfer_r2=0.275, datasets=3
- `fev` / `shift_risk`: transfer_r2=-18432965.625, baseline_transfer_r2=-18503450.360, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
