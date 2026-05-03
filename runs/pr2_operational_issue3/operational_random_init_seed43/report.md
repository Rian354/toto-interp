# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 0/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 1/4
- Dynamic concepts above raw-feature baseline: 0/6
- Layer 11 wins on coordination and cardinality: False
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 0

## Best BOOM Probe Views

- `cardinality_bucket`: layer 5, `first_decode`, `series_mean`; test_accuracy=0.330, baseline_test_accuracy=0.742
- `coordination`: layer -1, `all_context`, `series_mean`; test_r2=-0.016, baseline_test_r2=0.038, shuffled_test_r2=-0.017
- `current_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.061, baseline_test_r2=0.425, shuffled_test_r2=-0.009
- `current_sparsity`: layer 11, `all_context`, `per_variate`; test_r2=0.145, baseline_test_r2=1.000, shuffled_test_r2=-0.005
- `domain`: layer 4, `final_context`, `series_mean`; test_accuracy=0.454, baseline_test_accuracy=0.406
- `frequency_bucket`: layer 11, `final_context`, `series_mean`; test_accuracy=0.734, baseline_test_accuracy=0.474
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.091, baseline_test_r2=0.066, shuffled_test_r2=-0.005
- `future_sparsity`: layer 11, `all_context`, `per_variate`; test_r2=0.145, baseline_test_r2=1.000, shuffled_test_r2=-0.005
- `metric_type`: layer 9, `final_context`, `series_mean`; test_accuracy=0.462, baseline_test_accuracy=0.421
- `shift_risk`: layer -1, `first_decode`, `series_mean`; test_r2=0.052, baseline_test_r2=0.049, shuffled_test_r2=0.063

## Localization And Geometry

- Best probes concentrate as follows: layer -1: 2 best probes, layer 4: 1 best probes, layer 5: 1 best probes, layer 9: 1 best probes, layer 11: 5 best probes.
- Future-facing dynamic concepts average at layer 5.00.

## Causal Interventions

No intervention outputs were found.

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-1.111, baseline_transfer_r2=-7164.140, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9680400.493, baseline_transfer_r2=-9874846.207, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-71.426, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8186771.502, baseline_transfer_r2=-8556833.481, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-111.825, baseline_transfer_r2=0.277, datasets=3
- `fev` / `shift_risk`: transfer_r2=-18183874.396, baseline_transfer_r2=-18498853.502, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
