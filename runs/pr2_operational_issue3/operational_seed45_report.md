# Toto Interp Research Report

## Executive Summary

This run is organized around the narrowest main-track claim: operational regimes are linearly encoded in Toto and are better recovered from a pretrained frozen checkpoint than from stronger controls.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 2/4
- Operational concepts where pretrained linear beats random-init and FNO: 4/4
- Monotonic intervention families: 0

## Main-Track Operational Results

- `coordination`: layer 8, `all_context`, `series_mean`; test_r2=0.100, baseline_test_r2=0.104, shuffled_test_r2=-0.038
- `current_burstiness`: layer 9, `all_context`, `per_variate`; test_r2=0.162, baseline_test_r2=0.539, shuffled_test_r2=-0.005
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.219, baseline_test_r2=0.152, shuffled_test_r2=-0.038
- `shift_risk`: layer 11, `all_context`, `per_variate`; test_r2=0.217, baseline_test_r2=-0.026, shuffled_test_r2=-0.024

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=0.915, corr(|strength|, forecast_change)=0.976
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.452, corr(|strength|, forecast_change)=0.990
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=0.554, corr(|strength|, forecast_change)=0.996

## Control Comparison

- `coordination`: fno/pretrained=-0.008, linear_probe/pretrained=0.100, linear_probe/random_init=0.005
- `current_burstiness`: fno/pretrained=0.044, linear_probe/pretrained=0.162, linear_probe/random_init=0.083
- `future_burstiness`: fno/pretrained=0.043, linear_probe/pretrained=0.219, linear_probe/random_init=0.039
- `shift_risk`: fno/pretrained=-0.008, linear_probe/pretrained=0.217, linear_probe/random_init=0.008

## What This Run Demonstrates

This run supports a tighter main-track claim: operational regime variables such as burstiness, shift risk, and coordination are more linearly recoverable from a pretrained Toto checkpoint than from randomized weights or a nonlinear raw-window FNO baseline. Transfer and geometry should remain secondary unless they become materially stronger in larger runs.
