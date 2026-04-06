# Toto Interp

Independent interpretability research tooling for `Datadog/Toto-Open-Base-1.0`.

This repo keeps all research code independent of Toto itself. It consumes the published
`toto-ts` package via normal Python installation rather than requiring a vendored clone.

The package is designed around a BOOM-first research workflow:

- trace Toto residual-stream activations over patch tokens
- fit linear probes for observability-native concepts
- localize those concepts across layers and token positions
- intervene on learned directions during forecasting
- test zero-shot transfer on FEV and LSF datasets
- render a run report that summarizes what the pipeline discovered

Upstream/public functionality already ported into this repo:

- BOOM taxonomy loading and BOOM snapshot/window construction
- a standalone BOOM full-dataset downloader
- an official FEV task registry mirrored from Toto's public evaluation config
- an LSF downloader/validator/normalizer for the public CSV benchmark bundles
- optional local-dev fallback to a Toto checkout through `TOTO_REPO_PATH`

For a detailed architecture and capability report, see
[/Users/dhyeymavani/Documents/GitHub/toto-interp/docs/codebase_report.md](/Users/dhyeymavani/Documents/GitHub/toto-interp/docs/codebase_report.md).

## Local setup

1. `pip install -e .`

That installs this package plus the upstream `toto-ts` dependency.

If you ever want to test against an unpublished local Toto checkout instead, set
`TOTO_REPO_PATH=/path/to/toto`, but that is optional and not required for normal use.

## What Works End To End

- BOOM activation dumping without a Toto clone
- BOOM probe fitting for taxonomy and dynamic-regime labels
- causal steering/ablation runs for selected dynamic probes
- zero-shot transfer on public FEV datasets
- zero-shot transfer on local LSF datasets when CSVs are available
- single-command orchestration with a generated Markdown + JSON report

The BOOM dump script automatically snapshots only the needed BOOM series from Hugging Face.
If you want the entire public BOOM dataset locally, use:

- `python scripts/download_boom_dataset.py --output-dir data`

If you want the public LSF benchmark CSVs Toto expects, use:

- `python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets`

## Main entrypoints

- `python scripts/dump_toto_activations.py --output-dir runs/activations`
- `python scripts/fit_toto_probes.py --activation-files runs/activations/train_activations.pt runs/activations/val_activations.pt runs/activations/test_activations.pt --output-dir runs/probes`
- `python scripts/run_toto_interventions.py --probe-path runs/probes/artifacts/shift_risk__layer_10__final_context__series_mean.pt --output-dir runs/interventions`
- `python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer --dataset fev --fev-tasks entsoe_15T`
- `python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets`
- `python scripts/render_toto_report.py --probe-results-path runs/probes/probe_results.csv --intervention-dirs runs/interventions/shift_risk --transfer-dir runs/transfer --output-path runs/report.md --summary-json-path runs/report_summary.json`
- `python scripts/run_toto_pipeline.py --output-dir runs/full`

## End-to-end workflow

1. Dump BOOM activations:
   `python scripts/dump_toto_activations.py --output-dir runs/activations`
2. Fit BOOM probes:
   `python scripts/fit_toto_probes.py --activation-files runs/activations/train_activations.pt runs/activations/val_activations.pt runs/activations/test_activations.pt --output-dir runs/probes`
3. Run BOOM causal interventions:
   `python scripts/run_toto_interventions.py --probe-path runs/probes/artifacts/shift_risk__layer_10__final_context__series_mean.pt --output-dir runs/interventions`
4. Run zero-shot transfer on FEV:
   `python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer/fev --dataset fev --fev-tasks entsoe_15T epf_be`
5. Run zero-shot transfer on LSF:
   `python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer/lsf --dataset lsf --lsf-path data/lsf_datasets --lsf-datasets ETTh1 electricity --download-lsf`
6. Render a run report:
   `python scripts/render_toto_report.py --probe-results-path runs/probes/probe_results.csv --intervention-dirs runs/interventions/shift_risk runs/interventions/future_sparsity --transfer-dir runs/transfer --output-path runs/report.md --summary-json-path runs/report_summary.json`
7. Or run the whole workflow at once:
   `python scripts/run_toto_pipeline.py --output-dir runs/full --fev-tasks entsoe_15T epf_be --download-lsf --lsf-datasets ETTh1 electricity`

## Useful options

- `--device auto|cpu|cuda|mps` is supported across dump, intervention, transfer, and pipeline scripts.
- `--compile` enables Toto model compilation when available.
- `python scripts/run_toto_transfer.py --list-fev-tasks` prints the vendored official FEV task registry.
- `python scripts/run_toto_transfer.py --list-fev-tasks --fev-safe-only` restricts to paper-safe default tasks.
- `python scripts/run_toto_pipeline.py --reuse-existing` skips completed steps and only refreshes later outputs.
- `python scripts/download_lsf_datasets.py --validate-only` checks an existing local LSF layout.

## Transfer notes

- FEV transfer uses the public Hugging Face dataset `autogluon/fev_datasets`.
- LSF transfer uses Toto's published local LSF loader. This repo now ships a downloader and validator for the public CSV bundles Toto expects.
- `run_toto_transfer.py` can use the vendored official FEV task definitions or explicit dataset config names.
- `run_toto_transfer.py` automatically reuses the best continuous BOOM probe views from `probe_localization_summary.csv` when you point it at `--probe-dir`.
- Detailed LSF setup instructions and the public source links are in [docs/lsf_setup.md](/Users/dhyeymavani/Documents/GitHub/toto-interp/docs/lsf_setup.md).

## Smoke validation

This repo has been validated with:

- `python -m pytest tests -q`
- `python scripts/run_toto_pipeline.py --output-dir .smoke_pipeline --device cpu --context-length 128 --max-series-per-split 1 --max-windows-per-series 1 --num-samples 4 --fev-tasks entsoe_15T`

The smoke pipeline is intentionally tiny, so its metrics are not scientifically meaningful. Its purpose is to prove that the full orchestration path completes and produces activations, probes, interventions, transfer outputs, and a final report from a clean checkout.
