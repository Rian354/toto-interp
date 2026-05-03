# Toto Interp Research Report

## Executive Summary

This run is organized around the narrowest main-track claim: operational regimes are linearly encoded in Toto and are better recovered from a pretrained frozen checkpoint than from stronger controls.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 1/4
- Operational concepts where pretrained linear beats random-init and FNO: 4/4
- Monotonic intervention families: 1

## Main-Track Operational Results

- `coordination`: layer 11, `all_context`, `series_mean`; test_r2=0.100, baseline_test_r2=0.080, shuffled_test_r2=-0.085
- `current_burstiness`: layer 11, `all_context`, `series_mean`; test_r2=0.095, baseline_test_r2=0.631, shuffled_test_r2=-0.032
- `future_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.127, baseline_test_r2=0.013, shuffled_test_r2=-0.010
- `shift_risk`: layer 10, `all_context`, `series_mean`; test_r2=0.095, baseline_test_r2=0.083, shuffled_test_r2=-0.030

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=-0.183, corr(|strength|, forecast_change)=0.948
- `future_burstiness` (steer): monotonic=True, corr(strength, probe_score)=0.976, corr(|strength|, forecast_change)=0.749
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=0.365, corr(|strength|, forecast_change)=0.822

## Control Comparison

- `coordination`: fno/pretrained=0.050, linear_probe/pretrained=0.100, linear_probe/random_init=-0.003
- `current_burstiness`: fno/pretrained=0.021, linear_probe/pretrained=0.095, linear_probe/random_init=0.018
- `future_burstiness`: fno/pretrained=0.024, linear_probe/pretrained=0.127, linear_probe/random_init=0.042
- `shift_risk`: fno/pretrained=-0.008, linear_probe/pretrained=0.095, linear_probe/random_init=-0.008

## What This Run Demonstrates

This run supports a tighter main-track claim: operational regime variables such as burstiness, shift risk, and coordination are more linearly recoverable from a pretrained Toto checkpoint than from randomized weights or a nonlinear raw-window FNO baseline. Transfer and geometry should remain secondary unless they become materially stronger in larger runs.
