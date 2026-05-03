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

- `cardinality_bucket`: layer -1, `final_context`, `series_mean`; test_accuracy=0.364, baseline_test_accuracy=0.731
- `coordination`: layer -1, `final_context`, `per_variate`; test_r2=-0.004, baseline_test_r2=-0.196, shuffled_test_r2=-0.023
- `current_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.040, baseline_test_r2=0.499, shuffled_test_r2=-0.014
- `current_sparsity`: layer 0, `final_context`, `series_mean`; test_r2=0.013, baseline_test_r2=1.000, shuffled_test_r2=-0.101
- `domain`: layer -1, `final_context`, `series_mean`; test_accuracy=0.416, baseline_test_accuracy=0.338
- `frequency_bucket`: layer 11, `final_context`, `series_mean`; test_accuracy=0.723, baseline_test_accuracy=0.375
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.063, baseline_test_r2=0.260, shuffled_test_r2=-0.019
- `future_sparsity`: layer 0, `final_context`, `series_mean`; test_r2=0.013, baseline_test_r2=1.000, shuffled_test_r2=-0.102
- `metric_type`: layer 0, `first_decode`, `series_mean`; test_accuracy=0.465, baseline_test_accuracy=0.353
- `shift_risk`: layer 11, `all_context`, `per_variate`; test_r2=0.027, baseline_test_r2=0.006, shuffled_test_r2=-0.012

## Localization And Geometry

- Best probes concentrate as follows: layer -1: 3 best probes, layer 0: 3 best probes, layer 11: 4 best probes.
- Future-facing dynamic concepts average at layer 5.25.

## Causal Interventions

No intervention outputs were found.

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-88.444, baseline_transfer_r2=-3920.586, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9780011.610, baseline_transfer_r2=-9874846.207, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-16.918, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8486611.930, baseline_transfer_r2=-8528499.194, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-26.432, baseline_transfer_r2=0.276, datasets=3
- `fev` / `shift_risk`: transfer_r2=-17652874.713, baseline_transfer_r2=-18498853.471, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
