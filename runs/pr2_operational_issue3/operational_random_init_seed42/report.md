# Toto Interp Research Report

## Executive Summary

This run evaluates whether Toto contains linearly decodable observability-native concepts, whether those concepts localize to specific layers and token positions, and whether selected regime directions causally influence forecasts.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 0/4
- Operational concepts where pretrained linear beats random-init and FNO: 0/4
- Structural concepts above raw-feature baseline: 2/4
- Dynamic concepts above raw-feature baseline: 0/6
- Layer 11 wins on coordination and cardinality: False
- Cross-benchmark transfer wins (FEV and LSF): 0
- Monotonic intervention families: 0

## Best BOOM Probe Views

- `cardinality_bucket`: layer -1, `all_context`, `series_mean`; test_accuracy=0.300, baseline_test_accuracy=0.787
- `coordination`: layer -1, `all_context`, `series_mean`; test_r2=0.001, baseline_test_r2=0.070, shuffled_test_r2=-0.003
- `current_burstiness`: layer 0, `all_context`, `series_mean`; test_r2=-0.002, baseline_test_r2=0.444, shuffled_test_r2=-0.028
- `current_sparsity`: layer -1, `all_context`, `series_mean`; test_r2=-0.076, baseline_test_r2=1.000, shuffled_test_r2=-0.093
- `domain`: layer 0, `all_context`, `series_mean`; test_accuracy=0.406, baseline_test_accuracy=0.389
- `frequency_bucket`: layer 0, `first_decode`, `series_mean`; test_accuracy=0.727, baseline_test_accuracy=0.475
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.017, baseline_test_r2=0.004, shuffled_test_r2=-0.052
- `future_sparsity`: layer -1, `all_context`, `series_mean`; test_r2=-0.076, baseline_test_r2=1.000, shuffled_test_r2=-0.093
- `metric_type`: layer 2, `final_context`, `series_mean`; test_accuracy=0.446, baseline_test_accuracy=0.348
- `shift_risk`: layer 11, `final_context`, `per_variate`; test_r2=0.023, baseline_test_r2=0.015, shuffled_test_r2=-0.006

## Localization And Geometry

- Best probes concentrate as follows: layer -1: 4 best probes, layer 0: 3 best probes, layer 2: 1 best probes, layer 11: 2 best probes.
- Future-facing dynamic concepts average at layer 5.00.

## Causal Interventions

No intervention outputs were found.

## Zero-Shot Transfer

- `fev` / `coordination`: transfer_r2=-1.796, baseline_transfer_r2=-4623.239, datasets=3
- `fev` / `current_burstiness`: transfer_r2=-9746207.614, baseline_transfer_r2=-9900156.868, datasets=3
- `fev` / `current_sparsity`: transfer_r2=-0.185, baseline_transfer_r2=0.333, datasets=3
- `fev` / `future_burstiness`: transfer_r2=-8226936.757, baseline_transfer_r2=-8517400.713, datasets=3
- `fev` / `future_sparsity`: transfer_r2=-0.168, baseline_transfer_r2=0.275, datasets=3
- `fev` / `shift_risk`: transfer_r2=-17798941.050, baseline_transfer_r2=-17742149.642, datasets=3

## What This Run Demonstrates

This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best localization views, intervene on learned concept directions during forecasting, and test whether dynamic regime probes transfer to public FEV and LSF datasets.
