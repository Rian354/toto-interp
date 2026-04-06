# Toto Interp

Independent interpretability research tooling for `Datadog/Toto-Open-Base-1.0`.

This repo keeps all research code independent of Toto itself. It consumes the published
`toto-ts` package via normal Python installation rather than requiring a vendored clone.

## Local setup

1. `pip install -e .`

That installs this package plus the upstream `toto-ts` dependency.

If you ever want to test against an unpublished local Toto checkout instead, set
`TOTO_REPO_PATH=/path/to/toto`, but that is optional and not required for normal use.

## Main entrypoints

- `python scripts/dump_toto_activations.py --output-dir runs/activations`
- `python scripts/fit_toto_probes.py --activation-files runs/activations/train_activations.pt runs/activations/val_activations.pt runs/activations/test_activations.pt --output-dir runs/probes`
- `python scripts/run_toto_interventions.py --probe-path runs/probes/artifacts/shift_risk__layer_10__final_context__series_mean.pt --output-dir runs/interventions`
- `python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer --dataset fev --fev-configs ETT_15T`

## End-to-end workflow

1. Dump BOOM activations:
   `python scripts/dump_toto_activations.py --output-dir runs/activations`
2. Fit BOOM probes:
   `python scripts/fit_toto_probes.py --activation-files runs/activations/train_activations.pt runs/activations/val_activations.pt runs/activations/test_activations.pt --output-dir runs/probes`
3. Run BOOM causal interventions:
   `python scripts/run_toto_interventions.py --probe-path runs/probes/artifacts/shift_risk__layer_10__final_context__series_mean.pt --output-dir runs/interventions`
4. Run zero-shot transfer on FEV:
   `python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer/fev --dataset fev --fev-configs ETT_15T`
5. Run zero-shot transfer on LSF:
   `python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer/lsf --dataset lsf --lsf-path /path/to/lsf_data --lsf-datasets ETTh1 electricity`

## Transfer notes

- FEV transfer uses the public Hugging Face dataset `autogluon/fev_datasets`.
- LSF transfer uses Toto's published local LSF loader, so you need the benchmark CSVs available locally and passed with `--lsf-path`.
- `run_toto_transfer.py` automatically reuses the best continuous BOOM probe views from `probe_localization_summary.csv` when you point it at `--probe-dir`.
