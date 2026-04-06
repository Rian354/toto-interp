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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-series-per-split", type=int, default=8)
    parser.add_argument("--max-windows-per-series", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-intervention-probes", type=int, default=4)
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

    common_model_args = ["--model-id", args.model_id, "--device", args.device]
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
            "--output-dir",
            str(probes_dir),
        ],
        cwd=ROOT,
        done_path=probes_dir / "probe_results.csv",
        reuse_existing=args.reuse_existing,
    )

    intervention_dirs: list[Path] = []
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
    if not args.skip_transfer:
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
        "activations_dir": str(activations_dir),
        "probes_dir": str(probes_dir),
        "intervention_dirs": [str(path) for path in intervention_dirs],
        "transfer_dir": str(transfer_dir) if transfer_generated else None,
        "report_path": str(report_path),
        "report_summary_path": str(report_json_path),
    }
    (args.output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
