# Toto Interp Codebase Report

## Overview

`toto-interp` is an independent interpretability research codebase built around
`Datadog/Toto-Open-Base-1.0`. Its purpose is to test whether a time-series
foundation model contains linearly decodable operational concepts and whether
those concept directions causally influence forecasts.

The codebase is deliberately separated from Toto itself. It installs the public
`toto-ts` package from PyPI and treats Toto as a consumed dependency, while all
research logic, dataset plumbing, probe fitting, interventions, transfer
evaluation, and reporting live in this repo.

In practical terms, this repo turns Toto into an interpretable research object.
It provides a full workflow from raw BOOM windows to a generated report that
summarizes what the run discovered.

## Core Research Question

The main question this codebase investigates is:

1. Does Toto linearly encode observability-native structural concepts such as
   metric type, domain, frequency bucket, and multivariate cardinality?
2. Does Toto linearly encode dynamic operating-regime concepts such as
   sparsity, burstiness, shift risk, and multivariate coordination?
3. Are those directions behaviorally meaningful, in the sense that ablating or
   steering them changes the forecasts Toto produces?

This is the time-series analogue of the concept-probe plus causal-intervention
loop used in modern mechanistic and representation-level interpretability work.

## What The Codebase Actually Does

### 1. Builds BOOM-based research windows

The package uses BOOM as the main in-domain research dataset because BOOM
contains observability semantics that are not present in generic forecasting
benchmarks.

Implemented BOOM capabilities include:

- loading BOOM taxonomy metadata directly from the public Hugging Face dataset
- splitting by series id rather than by window, which avoids leakage
- constructing train/val/test windows plus optional held-out late windows
- deriving structural labels from taxonomy metadata
- deriving dynamic regime labels from the last context patch and next true patch
- computing raw-statistics baselines from the last context patch only

This means the codebase can automatically transform BOOM into an
interpretability-ready dataset without requiring Datadog's Toto repo to be
checked out locally.

### 2. Extracts Toto activations on patch tokens

The tracing layer operates on Toto's internal patch-token sequence rather than
on raw timesteps. It captures:

- the patch embedding output
- transformer layer outputs across all core layers
- three token views:
  - all context tokens
  - the final context token
  - the first decode token
- two pooling views:
  - series-level pooled activations
  - per-variate activations

This gives the codebase a reusable activation substrate for asking where
concepts live in the model and whether current-context and next-step concepts
appear at different layers or positions.

### 3. Fits linear probes with meaningful baselines

The probe layer supports two major task families:

- categorical concepts through logistic regression
- continuous concepts through ridge regression

For every probe view, the codebase records:

- train/val/test metrics
- a raw-feature baseline
- a shuffled-label control
- a mean-difference vector for causal work

This matters because it allows the codebase to separate three different claims:

- the concept is decodable at all
- the concept is more decodable than simple raw last-patch statistics
- the direction has a plausible causal interpretation for intervention

### 4. Produces geometry and localization artifacts

After fitting probes, the codebase exports:

- `probe_results.csv`
- `probe_localization_summary.csv`
- `probe_geometry_cosine.csv`
- `probe_geometry_pca.csv`
- `probe_geometry_meta.json`

These files let a researcher inspect:

- which layer/token/pooling view is best for each concept
- whether future-facing concepts skew later than current-patch concepts
- whether layer 11 is especially informative for multivariate concepts
- whether concept vectors cluster in intuitive ways

### 5. Runs causal forecasting interventions

The intervention layer supports:

- additive steering
- projection ablation

Interventions are applied to specific residual-stream layers and token views
while Toto forecasts. The outputs include:

- per-strength summary metrics
- per-window metrics
- forecast change measurements
- probe-score movement under intervention
- selective performance metrics on concept-conditioned subsets

This is the part of the codebase that moves beyond passive interpretability.
It tests whether a linearly identified regime direction actually changes model
behavior in the expected direction.

### 6. Evaluates zero-shot transfer to public forecasting benchmarks

The transfer layer allows BOOM-trained dynamic probes to be evaluated on:

- FEV datasets from `autogluon/fev_datasets`
- LSF datasets through Toto's published LSF loader

This repo vendors the official public FEV task registry from Toto's evaluation
configuration so the transfer interface is useful even though `toto-ts` does
not ship that YAML with the package.

That makes the codebase more than a BOOM-only harness. It can test whether a
regime direction learned in an observability-native setting also measures
something stable on out-of-domain time-series benchmarks.

### 7. Generates research reports automatically

The reporting layer reads the outputs of the probe, intervention, and transfer
stages and generates:

- a Markdown report
- a JSON summary

The report includes:

- an executive summary
- a run-level acceptance snapshot
- best BOOM probe views
- localization summary
- intervention summary
- transfer summary

This makes the codebase usable as a repeatable research system rather than a
collection of ad hoc notebooks.

## Valuable Insights This Repo Can Produce

When run on real BOOM-scale settings rather than the minimal smoke fixture, the
repo is designed to answer questions like:

- Which operational concepts are already explicit in Toto's residual stream?
- Are structural observability concepts linearly encoded more strongly in
  pooled series views than in per-variate views?
- Are future-regime concepts concentrated later in the network than
  current-patch morphology concepts?
- Does Toto's space-wise layer improve multivariate concept decoding?
- Do intervention sweeps move forecast uncertainty or forecast shape in a way
  that tracks the learned concept direction?
- Do BOOM-derived dynamic probes transfer to non-observability time-series
  benchmarks?

Those are publishable, model-centric questions. The repo is not just a wrapper
around Toto inference. It is a research environment for discovering and testing
representational hypotheses about Toto.

## Upstream Functionality Ported Into This Repo

This codebase already ports the most important public utilities needed for an
independent research workflow:

- BOOM taxonomy access and partial snapshot downloading
- BOOM full-dataset downloading
- BOOM windowing utilities
- FEV task definitions from Toto's public evaluation configuration
- compatibility logic for using the published `toto-ts` package with an
  optional no-xFormers fallback

As a result, the repo no longer needs a vendored Toto checkout to be useful.

## End-to-End User Experience

A user can now:

1. `pip install -e .`
2. run `scripts/run_toto_pipeline.py`
3. wait for the pipeline to dump activations, fit probes, run interventions,
   run transfer, and emit a final report

This is the key practical accomplishment of the codebase: it converts a
somewhat scattered set of model and dataset interfaces into a single coherent
research pipeline.

## What The Smoke Run Demonstrates

The included smoke workflow proves the software stack is complete:

- activations are dumped
- probes are fit
- intervention outputs are produced
- transfer outputs are produced
- a final report is generated

The smoke metrics themselves are intentionally not meaningful because the run is
tiny by design. Their purpose is software validation, not scientific evidence.

## Current Limits

The most important remaining external requirement is LSF data availability.
FEV runs can execute against public Hugging Face data, but LSF transfer still
requires local benchmark CSVs supplied through `--lsf-path`.

That is a dataset availability issue, not a missing software capability: the
adapter and evaluation path are already implemented here.

## Bottom Line

This repo accomplishes something concrete and useful:

it makes Toto interpretable as a research subject.

More specifically, it provides an end-to-end, clone-free, reproducible
framework for measuring, localizing, intervening on, and reporting linear
concept representations in Toto using BOOM as the primary research dataset and
FEV/LSF as transfer environments.
