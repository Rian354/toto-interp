# NeurIPS Preliminary Experiments

This repo now has a thin experiment runner for GPU-oriented preliminary runs that layers on top of the existing pipeline instead of replacing it:

- [scripts/run_neurips_prelim_experiments.py](/Users/saurabhatri/Dev/toto-interp/scripts/run_neurips_prelim_experiments.py)

The intent is to keep the PR incremental:

- preserve the original broader paper pipeline with `label_group=all`
- add a tighter operational-regime control track for the reviewer-facing claim
- keep all runs manifest-driven and reproducible

## Recommended Suites

Use `operational` for the narrow main-track result:

```bash
python scripts/run_neurips_prelim_experiments.py \
  --suite operational \
  --size gpu_prelim \
  --device cuda \
  --seeds 42 43 44 \
  --skip-transfer
```

This launches, per seed:

- `pretrained + linear_probe + operational`
- `random_init + linear_probe + operational`
- `pretrained + fno + operational`
- a combined control report

Use `paper` to keep the original broader LP run alive:

```bash
python scripts/run_neurips_prelim_experiments.py \
  --suite paper \
  --size gpu_prelim \
  --device cuda \
  --seeds 42 \
  --skip-transfer
```

Use `all` when you want both tracks together:

```bash
python scripts/run_neurips_prelim_experiments.py \
  --suite all \
  --size prelim \
  --device cuda \
  --seeds 42 \
  --skip-transfer
```

## Presets

- `pilot`: fast smoke-scale comparison
- `prelim`: moderate preliminary result run
- `gpu_prelim`: larger GPU-oriented preliminary run

The preset controls:

- `max_series_per_split`
- `max_windows_per_series`
- `num_samples`
- FNO width/layers/modes/epochs/batch size

## Outputs

Runs are written under `runs/neurips_prelim` by default.

Operational suite outputs:

- `operational_pretrained_seed{seed}`
- `operational_random_init_seed{seed}`
- `operational_fno_seed{seed}`
- `operational_seed{seed}_report.md`
- `operational_seed{seed}_report.json`

Broader paper outputs:

- `paper_full_seed{seed}`

Each pipeline run also writes a `pipeline_manifest.json` with device, seed, context length, window budget, and report settings.

## Notes

- Use `--dry-run` first on a new machine to inspect exact commands.
- Keep `--skip-transfer` on for the first GPU pass unless the operational decoding/control results are already stable.
- The operational report is the right artifact for the tighter reviewer-facing claim.
- The broader paper run should remain in parallel so this PR reads as an extension, not a rewrite.
