# Toto Interp Research Report

## Executive Summary

This run is organized around the narrowest main-track claim: operational regimes are linearly encoded in Toto and are better recovered from a pretrained frozen checkpoint than from stronger controls.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 2/4
- Operational concepts where pretrained linear beats random-init and FNO: 4/4
- Monotonic intervention families: 0

## Main-Track Operational Results

- `coordination`: layer 11, `all_context`, `series_mean`; test_r2=0.047, baseline_test_r2=0.038, shuffled_test_r2=-0.062
- `current_burstiness`: layer 11, `all_context`, `per_variate`; test_r2=0.143, baseline_test_r2=0.425, shuffled_test_r2=-0.011
- `future_burstiness`: layer 11, `all_context`, `series_mean`; test_r2=0.108, baseline_test_r2=0.048, shuffled_test_r2=-0.052
- `shift_risk`: layer 8, `final_context`, `per_variate`; test_r2=0.124, baseline_test_r2=-0.026, shuffled_test_r2=-0.515

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=-0.906, corr(|strength|, forecast_change)=0.686
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.564, corr(|strength|, forecast_change)=0.762
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=-0.726, corr(|strength|, forecast_change)=0.735

## Control Comparison

- `coordination`: fno/pretrained=0.000, linear_probe/pretrained=0.047, linear_probe/random_init=-0.016
- `current_burstiness`: fno/pretrained=0.036, linear_probe/pretrained=0.143, linear_probe/random_init=0.061
- `future_burstiness`: fno/pretrained=0.057, linear_probe/pretrained=0.108, linear_probe/random_init=0.091
- `shift_risk`: fno/pretrained=-0.003, linear_probe/pretrained=0.124, linear_probe/random_init=0.052

## What This Run Demonstrates

This run supports a tighter main-track claim: operational regime variables such as burstiness, shift risk, and coordination are more linearly recoverable from a pretrained Toto checkpoint than from randomized weights or a nonlinear raw-window FNO baseline. Transfer and geometry should remain secondary unless they become materially stronger in larger runs.
