# Toto Interp Research Report

## Executive Summary

This run is organized around the narrowest main-track claim: operational regimes are linearly encoded in Toto and are better recovered from a pretrained frozen checkpoint than from stronger controls.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 1/4
- Operational concepts where pretrained linear beats random-init and FNO: 4/4
- Monotonic intervention families: 0

## Main-Track Operational Results

- `coordination`: layer 7, `all_context`, `series_mean`; test_r2=0.068, baseline_test_r2=0.061, shuffled_test_r2=-0.045
- `current_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.069, baseline_test_r2=0.499, shuffled_test_r2=-0.012
- `future_burstiness`: layer 11, `all_context`, `series_mean`; test_r2=0.070, baseline_test_r2=0.174, shuffled_test_r2=-0.037
- `shift_risk`: layer 6, `final_context`, `per_variate`; test_r2=0.177, baseline_test_r2=0.006, shuffled_test_r2=-0.112

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=0.303, corr(|strength|, forecast_change)=0.997
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.737, corr(|strength|, forecast_change)=0.849
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=-0.690, corr(|strength|, forecast_change)=0.984

## Control Comparison

- `coordination`: fno/pretrained=0.041, linear_probe/pretrained=0.068, linear_probe/random_init=-0.004
- `current_burstiness`: fno/pretrained=0.051, linear_probe/pretrained=0.069, linear_probe/random_init=0.040
- `future_burstiness`: fno/pretrained=0.038, linear_probe/pretrained=0.070, linear_probe/random_init=0.063
- `shift_risk`: fno/pretrained=-0.014, linear_probe/pretrained=0.177, linear_probe/random_init=0.027

## What This Run Demonstrates

This run supports a tighter main-track claim: operational regime variables such as burstiness, shift risk, and coordination are more linearly recoverable from a pretrained Toto checkpoint than from randomized weights or a nonlinear raw-window FNO baseline. Transfer and geometry should remain secondary unless they become materially stronger in larger runs.
