from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SizePreset:
    max_series_per_split: int
    max_windows_per_series: int
    num_samples: int
    fno_epochs: int
    fno_batch_size: int
    fno_width: int
    fno_layers: int
    fno_modes: int


SIZE_PRESETS: dict[str, SizePreset] = {
    "pilot": SizePreset(
        max_series_per_split=8,
        max_windows_per_series=4,
        num_samples=8,
        fno_epochs=10,
        fno_batch_size=8,
        fno_width=16,
        fno_layers=2,
        fno_modes=8,
    ),
    "prelim": SizePreset(
        max_series_per_split=16,
        max_windows_per_series=6,
        num_samples=8,
        fno_epochs=12,
        fno_batch_size=8,
        fno_width=16,
        fno_layers=2,
        fno_modes=8,
    ),
    "gpu_prelim": SizePreset(
        max_series_per_split=24,
        max_windows_per_series=8,
        num_samples=16,
        fno_epochs=16,
        fno_batch_size=16,
        fno_width=24,
        fno_layers=3,
        fno_modes=12,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeurIPS-oriented preliminary experiment suites without changing the core pipeline."
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/neurips_prelim"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--suite", choices=("operational", "paper", "all"), default="all")
    parser.add_argument("--size", choices=tuple(SIZE_PRESETS), default="prelim")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str((ROOT / "scripts" / name).resolve())


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def render_operational_report(
    *,
    output_root: Path,
    seed: int,
    dry_run: bool,
) -> None:
    pretrained_dir = output_root / f"operational_pretrained_seed{seed}"
    random_init_dir = output_root / f"operational_random_init_seed{seed}"
    fno_dir = output_root / f"operational_fno_seed{seed}" / "probes"
    intervention_dirs = [
        pretrained_dir / "interventions" / "future_burstiness",
        pretrained_dir / "interventions" / "shift_risk",
        pretrained_dir / "interventions" / "coordination",
    ]
    if not dry_run:
        intervention_dirs = [path for path in intervention_dirs if path.exists()]
    command = [
        sys.executable,
        script_path("render_toto_report.py"),
        "--probe-results-path",
        str(pretrained_dir / "probes" / "probe_results.csv"),
        str(random_init_dir / "probes" / "probe_results.csv"),
        str(fno_dir / "probe_results.csv"),
        "--report-focus",
        "operational",
        "--primary-method",
        "linear_probe",
        "--primary-weight-source",
        "pretrained",
        "--intervention-dirs",
        *[str(path) for path in intervention_dirs],
        "--output-path",
        str(output_root / f"operational_seed{seed}_report.md"),
        "--summary-json-path",
        str(output_root / f"operational_seed{seed}_report.json"),
    ]
    run_command(command, dry_run=dry_run)


def run_operational_suite(
    *,
    output_root: Path,
    device: str,
    preset: SizePreset,
    context_length: int,
    seed: int,
    skip_transfer: bool,
    reuse_existing: bool,
    dry_run: bool,
) -> None:
    pretrained_dir = output_root / f"operational_pretrained_seed{seed}"
    random_init_dir = output_root / f"operational_random_init_seed{seed}"
    fno_dir = output_root / f"operational_fno_seed{seed}"

    common_pipeline_args = [
        "--device",
        device,
        "--context-length",
        str(context_length),
        "--max-series-per-split",
        str(preset.max_series_per_split),
        "--max-windows-per-series",
        str(preset.max_windows_per_series),
        "--num-samples",
        str(preset.num_samples),
        "--label-group",
        "operational",
        "--report-focus",
        "operational",
        "--seed",
        str(seed),
    ]
    if skip_transfer:
        common_pipeline_args.append("--skip-transfer")
    if reuse_existing:
        common_pipeline_args.append("--reuse-existing")

    run_command(
        [
            sys.executable,
            script_path("run_toto_pipeline.py"),
            "--output-dir",
            str(pretrained_dir),
            "--weight-source",
            "pretrained",
            "--max-intervention-probes",
            "4",
            *common_pipeline_args,
        ],
        dry_run=dry_run,
    )

    run_command(
        [
            sys.executable,
            script_path("run_toto_pipeline.py"),
            "--output-dir",
            str(random_init_dir),
            "--weight-source",
            "random_init",
            "--max-intervention-probes",
            "0",
            *common_pipeline_args,
        ],
        dry_run=dry_run,
    )

    run_command(
        [
            sys.executable,
            script_path("fit_toto_probes.py"),
            "--activation-files",
            str(pretrained_dir / "activations" / "train_activations.pt"),
            str(pretrained_dir / "activations" / "val_activations.pt"),
            str(pretrained_dir / "activations" / "test_activations.pt"),
            "--window-files",
            str(pretrained_dir / "activations" / "train_windows.pt"),
            str(pretrained_dir / "activations" / "val_windows.pt"),
            str(pretrained_dir / "activations" / "test_windows.pt"),
            "--output-dir",
            str(fno_dir / "probes"),
            "--method",
            "fno",
            "--label-group",
            "operational",
            "--device",
            device,
            "--seed",
            str(seed),
            "--fno-width",
            str(preset.fno_width),
            "--fno-layers",
            str(preset.fno_layers),
            "--fno-modes",
            str(preset.fno_modes),
            "--epochs",
            str(preset.fno_epochs),
            "--batch-size",
            str(preset.fno_batch_size),
            "--model-id",
            "Datadog/Toto-Open-Base-1.0",
            "--weight-source",
            "pretrained",
        ],
        dry_run=dry_run,
    )

    render_operational_report(output_root=output_root, seed=seed, dry_run=dry_run)


def run_paper_suite(
    *,
    output_root: Path,
    device: str,
    preset: SizePreset,
    context_length: int,
    seed: int,
    skip_transfer: bool,
    reuse_existing: bool,
    dry_run: bool,
) -> None:
    paper_dir = output_root / f"paper_full_seed{seed}"
    command = [
        sys.executable,
        script_path("run_toto_pipeline.py"),
        "--output-dir",
        str(paper_dir),
        "--device",
        device,
        "--context-length",
        str(context_length),
        "--max-series-per-split",
        str(preset.max_series_per_split),
        "--max-windows-per-series",
        str(preset.max_windows_per_series),
        "--num-samples",
        str(preset.num_samples),
        "--label-group",
        "all",
        "--report-focus",
        "full",
        "--weight-source",
        "pretrained",
        "--max-intervention-probes",
        "4",
        "--seed",
        str(seed),
    ]
    if skip_transfer:
        command.append("--skip-transfer")
    if reuse_existing:
        command.append("--reuse-existing")
    run_command(command, dry_run=dry_run)


def write_plan_manifest(args: argparse.Namespace, preset: SizePreset) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": args.suite,
        "device": args.device,
        "size": args.size,
        "preset": asdict(preset),
        "context_length": args.context_length,
        "seeds": args.seeds,
        "skip_transfer": args.skip_transfer,
        "reuse_existing": args.reuse_existing,
        "dry_run": args.dry_run,
    }
    (args.output_root / "experiment_plan.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    preset = SIZE_PRESETS[args.size]
    write_plan_manifest(args, preset)

    for seed in args.seeds:
        if args.suite in {"operational", "all"}:
            run_operational_suite(
                output_root=args.output_root,
                device=args.device,
                preset=preset,
                context_length=args.context_length,
                seed=seed,
                skip_transfer=args.skip_transfer,
                reuse_existing=args.reuse_existing,
                dry_run=args.dry_run,
            )
        if args.suite in {"paper", "all"}:
            run_paper_suite(
                output_root=args.output_root,
                device=args.device,
                preset=preset,
                context_length=args.context_length,
                seed=seed,
                skip_transfer=args.skip_transfer,
                reuse_existing=args.reuse_existing,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
