from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_INTERVENTION_LABELS = ("future_sparsity", "future_burstiness", "shift_risk", "coordination")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Toto interpretability research pipeline end to end.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--weight-source", choices=("pretrained", "random_init", "checkpoint"), default="pretrained")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--randomize-scope", choices=("full", "selected_layers", "head_only"), default="full")
    parser.add_argument("--randomize-layers", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-series-per-split", type=int, default=8)
    parser.add_argument("--max-windows-per-series", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-intervention-probes", type=int, default=4)
    parser.add_argument("--probe-method", choices=("linear_probe", "fno"), default="linear_probe")
    parser.add_argument("--label-group", choices=("all", "taxonomy", "dynamic", "operational"), default="all")
    parser.add_argument("--report-focus", choices=("full", "operational"), default="full")
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--fev-tasks", nargs="*", default=["entsoe_15T", "epf_be"])
    parser.add_argument("--lsf-datasets", nargs="*", default=["ETTh1"])
    parser.add_argument("--lsf-path", type=Path, default=None)
    parser.add_argument("--download-lsf", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str((ROOT / "scripts" / name).resolve())


def run_step(command: list[str], *, cwd: Path, done_path: Path | None = None, reuse_existing: bool = False) -> None:
    if reuse_existing and done_path is not None and done_path.exists():
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def select_probe_paths(probe_dir: Path, max_count: int) -> list[Path]:
    results_path = probe_dir / "probe_results.csv"
    df = pd.read_csv(results_path)
    df = df[
        (df["task_type"] == "continuous")
        & (df.get("method", "linear_probe") == "linear_probe")
        & (df["label"].isin(ALLOWED_INTERVENTION_LABELS))
        & (df["layer"] >= 0)
    ]
    selected: list[Path] = []
    for label in ALLOWED_INTERVENTION_LABELS:
        rows = df[df["label"] == label]
        if rows.empty:
            continue
        metric_column = "test_r2" if "test_r2" in rows.columns else rows.columns[-1]
        valid = rows[metric_column].dropna()
        if valid.empty:
            continue
        selected.append(Path(rows.loc[valid.idxmax()]["artifact_path"]))
    return selected[:max_count]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = args.output_dir / "activations"
    probes_dir = args.output_dir / "probes"
    interventions_root = args.output_dir / "interventions"
    transfer_dir = args.output_dir / "transfer"
    report_path = args.output_dir / "report.md"
    report_json_path = args.output_dir / "report_summary.json"

    common_model_args = [
        "--model-id",
        args.model_id,
        "--device",
        args.device,
        "--weight-source",
        args.weight_source,
        "--randomize-scope",
        args.randomize_scope,
    ]
    if args.checkpoint_path is not None:
        common_model_args.extend(["--checkpoint-path", str(args.checkpoint_path)])
    if args.randomize_layers:
        common_model_args.extend(["--randomize-layers", *[str(layer) for layer in args.randomize_layers]])
    if args.compile:
        common_model_args.append("--compile")

    run_step(
        [
            sys.executable,
            script_path("dump_toto_activations.py"),
            "--output-dir",
            str(activations_dir),
            "--seed",
            str(args.seed),
            "--context-length",
            str(args.context_length),
            "--max-series-per-split",
            str(args.max_series_per_split),
            "--max-windows-per-series",
            str(args.max_windows_per_series),
            *common_model_args,
        ],
        cwd=ROOT,
        done_path=activations_dir / "activation_dump_summary.json",
        reuse_existing=args.reuse_existing,
    )

    run_step(
        [
            sys.executable,
            script_path("fit_toto_probes.py"),
            "--activation-files",
            str(activations_dir / "train_activations.pt"),
            str(activations_dir / "val_activations.pt"),
            str(activations_dir / "test_activations.pt"),
            "--window-files",
            str(activations_dir / "train_windows.pt"),
            str(activations_dir / "val_windows.pt"),
            str(activations_dir / "test_windows.pt"),
            "--output-dir",
            str(probes_dir),
            "--method",
            args.probe_method,
            "--label-group",
            args.label_group,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--model-id",
            args.model_id,
            "--weight-source",
            args.weight_source,
            "--fno-modes",
            str(args.fno_modes),
            "--fno-width",
            str(args.fno_width),
            "--fno-layers",
            str(args.fno_layers),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--randomize-scope",
            args.randomize_scope,
            *(
                ["--checkpoint-path", str(args.checkpoint_path)]
                if args.checkpoint_path is not None
                else []
            ),
            *(
                ["--randomize-layers", *[str(layer) for layer in args.randomize_layers]]
                if args.randomize_layers
                else []
            ),
        ],
        cwd=ROOT,
        done_path=probes_dir / "probe_results.csv",
        reuse_existing=args.reuse_existing,
    )

    intervention_dirs: list[Path] = []
    if args.probe_method == "linear_probe":
        for probe_path in select_probe_paths(probes_dir, args.max_intervention_probes):
            label = probe_path.stem.split("__")[0]
            output_dir = interventions_root / label
            intervention_dirs.append(output_dir)
            run_step(
                [
                    sys.executable,
                    script_path("run_toto_interventions.py"),
                    "--probe-path",
                    str(probe_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    str(args.seed),
                    "--context-length",
                    str(args.context_length),
                    "--max-series",
                    str(args.max_series_per_split),
                    "--max-windows-per-series",
                    str(max(1, min(4, args.max_windows_per_series))),
                    "--num-samples",
                    str(args.num_samples),
                    *common_model_args,
                ],
                cwd=ROOT,
                done_path=output_dir / "intervention_summary.csv",
                reuse_existing=args.reuse_existing,
            )

    transfer_generated = False
    if not args.skip_transfer and args.probe_method == "linear_probe":
        resolved_lsf_path = args.lsf_path or (ROOT / "data" / "lsf_datasets")
        include_lsf = args.download_lsf or args.lsf_path is not None or resolved_lsf_path.exists()
        transfer_command = [
            sys.executable,
            script_path("run_toto_transfer.py"),
            "--probe-dir",
            str(probes_dir),
            "--output-dir",
            str(transfer_dir),
            "--context-length",
            str(args.context_length),
            "--max-series",
            str(args.max_series_per_split),
            "--max-windows-per-series",
            str(args.max_windows_per_series),
            "--fev-tasks",
            *args.fev_tasks,
            *common_model_args,
        ]
        if include_lsf:
            transfer_command.extend(
                [
                    "--dataset",
                    "both",
                    "--lsf-path",
                    str(resolved_lsf_path),
                    "--lsf-datasets",
                    *args.lsf_datasets,
                ]
            )
            if args.download_lsf:
                transfer_command.append("--download-lsf")
        else:
            transfer_command.extend(["--dataset", "fev"])

        run_step(
            transfer_command,
            cwd=ROOT,
            done_path=transfer_dir / "transfer_summary.csv",
            reuse_existing=args.reuse_existing,
        )
        transfer_generated = True

    run_step(
        [
            sys.executable,
            script_path("render_toto_report.py"),
            "--probe-results-path",
            str(probes_dir / "probe_results.csv"),
            "--report-focus",
            args.report_focus,
            "--primary-method",
            "linear_probe",
            "--primary-weight-source",
            args.weight_source,
            "--intervention-dirs",
            *[str(path) for path in intervention_dirs],
            "--output-path",
            str(report_path),
            "--summary-json-path",
            str(report_json_path),
            *(["--transfer-dir", str(transfer_dir)] if transfer_generated else []),
        ],
        cwd=ROOT,
        done_path=report_path,
        reuse_existing=False,
    )

    manifest = {
        "model_id": args.model_id,
        "device": args.device,
        "seed": args.seed,
        "context_length": args.context_length,
        "max_series_per_split": args.max_series_per_split,
        "max_windows_per_series": args.max_windows_per_series,
        "num_samples": args.num_samples,
        "max_intervention_probes": args.max_intervention_probes,
        "skip_transfer": args.skip_transfer,
        "activations_dir": str(activations_dir),
        "probes_dir": str(probes_dir),
        "intervention_dirs": [str(path) for path in intervention_dirs],
        "transfer_dir": str(transfer_dir) if transfer_generated else None,
        "report_path": str(report_path),
        "report_summary_path": str(report_json_path),
        "probe_method": args.probe_method,
        "label_group": args.label_group,
        "report_focus": args.report_focus,
        "weight_source": args.weight_source,
        "checkpoint_path": None if args.checkpoint_path is None else str(args.checkpoint_path),
    }
    (args.output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
