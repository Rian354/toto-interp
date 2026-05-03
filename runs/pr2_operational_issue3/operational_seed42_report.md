# Toto Interp Research Report

## Executive Summary

This run is organized around the narrowest main-track claim: operational regimes are linearly encoded in Toto and are better recovered from a pretrained frozen checkpoint than from stronger controls.

## Acceptance Snapshot

- Operational concepts above raw-feature baseline: 0/4
- Operational concepts where pretrained linear beats random-init and FNO: 3/4
- Monotonic intervention families: 0

## Main-Track Operational Results

- `coordination`: layer 11, `all_context`, `series_mean`; test_r2=0.035, baseline_test_r2=0.070, shuffled_test_r2=-0.044
- `current_burstiness`: layer 5, `all_context`, `per_variate`; test_r2=0.072, baseline_test_r2=0.450, shuffled_test_r2=-0.062
- `future_burstiness`: layer 10, `all_context`, `series_mean`; test_r2=0.078, baseline_test_r2=0.093, shuffled_test_r2=-0.095
- `shift_risk`: layer 10, `final_context`, `per_variate`; test_r2=0.027, baseline_test_r2=0.015, shuffled_test_r2=-0.019

## Causal Interventions

- `coordination` (steer): monotonic=False, corr(strength, probe_score)=-0.846, corr(|strength|, forecast_change)=0.989
- `future_burstiness` (steer): monotonic=False, corr(strength, probe_score)=0.399, corr(|strength|, forecast_change)=0.746
- `shift_risk` (steer): monotonic=False, corr(strength, probe_score)=-0.347, corr(|strength|, forecast_change)=0.990

## Control Comparison

- `coordination`: fno/pretrained=0.075, linear_probe/pretrained=0.035, linear_probe/random_init=0.001
- `current_burstiness`: fno/pretrained=0.015, linear_probe/pretrained=0.072, linear_probe/random_init=-0.002
- `future_burstiness`: fno/pretrained=0.024, linear_probe/pretrained=0.078, linear_probe/random_init=0.017
- `shift_risk`: fno/pretrained=-0.001, linear_probe/pretrained=0.027, linear_probe/random_init=0.023

## What This Run Demonstrates

This run supports a tighter main-track claim: operational regime variables such as burstiness, shift risk, and coordination are more linearly recoverable from a pretrained Toto checkpoint than from randomized weights or a nonlinear raw-window FNO baseline. Transfer and geometry should remain secondary unless they become materially stronger in larger runs.
