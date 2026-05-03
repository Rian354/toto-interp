#!/usr/bin/env python3
"""Aggregate PR2-style operational runs (pretrained LP + random-init LP + FNO) with Issue #3-matched linear rows.

Also emits probe cost tables (train time, VRAM, FLOPs, latency) and FNO learning curves by reading
``artifact_metadata`` from saved ``.pt`` files — same variables as ``aggregate_random_weight_fno.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.issue3_matching import extract_issue3_matched_linear_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--skip-cost-metrics",
        action="store_true",
        help="Skip timing/FLOPs/learning-curve CSVs (Issue #3 tables only).",
    )
    return p.parse_args()


def mean_ci(values: np.ndarray, ci: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, float("nan")
    z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96
    half = float(z * values.std(ddof=1) / np.sqrt(values.size))
    return mean, half


def load_probe_best_per_label(path: Path) -> pd.DataFrame:
    """One best row per (label, method, weight_source[, seed]) for stable timing/metrics."""
    df = pd.read_csv(path)
    if df.empty:
        return df
    group_cols = ["label", "method"]
    if "weight_source" in df.columns:
        group_cols.append("weight_source")
    if "seed" in df.columns:
        group_cols.append("seed")

    picked: list[pd.Series] = []
    for _, g in df.groupby(group_cols, dropna=False):
        if "task_type" in g.columns:
            tt = str(g["task_type"].iloc[0]).lower()
            metric = "test_r2" if tt == "continuous" else "test_accuracy"
        elif "test_accuracy" in g.columns and g["test_accuracy"].notna().any():
            metric = "test_accuracy"
        else:
            metric = "test_r2"
        if metric not in g.columns or not g[metric].notna().any():
            continue
        idx = g[metric].idxmax()
        picked.append(g.loc[idx])

    if not picked:
        return df.iloc[0:0].copy()
    return pd.DataFrame(picked).reset_index(drop=True)


def extract_artifact_history_timing(artifact_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    import torch

    artifact = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    meta = getattr(artifact, "artifact_metadata", {}) or {}
    history = meta.get("history", [])
    timing = meta.get("timing", {})
    frame = pd.DataFrame(history)
    if frame.empty:
        frame = pd.DataFrame(columns=["epoch", "epoch_time_s", "train_loss", "val_loss"])
    timing_floats = {k: float(v) for k, v in timing.items() if isinstance(v, (int, float, np.floating))}
    return frame, timing_floats


def collect_pr2_cost_tables(
    runs_root: Path, seeds: list[int]
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    """Timing rows, non-empty learning-curve frames, and metric rows for cost efficiency."""
    timing_rows: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    arms: list[tuple[str, str]] = [
        ("pretrained_lp", "operational_pretrained_seed{seed}/probes/probe_results.csv"),
        ("random_init_lp", "operational_random_init_seed{seed}/probes/probe_results.csv"),
        ("fno", "operational_fno_seed{seed}/probes/probe_results.csv"),
    ]

    for seed in seeds:
        for run_arm, rel in arms:
            csv_path = runs_root / rel.format(seed=seed)
            if not csv_path.exists():
                continue
            df = load_probe_best_per_label(csv_path)
            if df.empty:
                continue
            for _, r in df.iterrows():
                tt = str(r.get("task_type", "")).lower()
                metric_col = "test_r2" if tt == "continuous" else "test_accuracy"
                if metric_col not in r.index:
                    metric_col = "test_r2" if "test_r2" in r.index else "test_accuracy"
                ap = Path(str(r["artifact_path"]))
                if not ap.exists():
                    alt = csv_path.parent / "artifacts" / ap.name
                    if alt.exists():
                        ap = alt
                    else:
                        continue
                hist, timing = extract_artifact_history_timing(ap)
                if not hist.empty:
                    h = hist.copy()
                    h["seed"] = seed
                    h["label"] = str(r["label"])
                    h["run_arm"] = run_arm
                    histories.append(h)
                timing_rows.append({"seed": seed, "label": str(r["label"]), "run_arm": run_arm, **timing})
                mval = r[metric_col] if metric_col in r.index else float("nan")
                if pd.notna(mval):
                    metric_rows.append(
                        {
                            "seed": seed,
                            "label": str(r["label"]),
                            "run_arm": run_arm,
                            "metric_col": metric_col,
                            "metric_value": float(mval),
                            "train_total_time_s": float(timing.get("train_total_time_s", float("nan"))),
                        }
                    )

    return timing_rows, histories, metric_rows


def write_cost_outputs(
    out_dir: Path,
    timing_rows: list[dict[str, object]],
    histories: list[pd.DataFrame],
    metric_rows: list[dict[str, object]],
    ci: float,
) -> dict[str, str]:
    """Write timing / curves / cost-efficiency CSVs; return output paths."""
    paths: dict[str, str] = {}
    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        p = out_dir / "probe_timing_raw.csv"
        timing_df.to_csv(p, index=False)
        paths["probe_timing_raw"] = str(p)

        timing_agg: list[dict[str, object]] = []
        for (label, run_arm), frame in timing_df.groupby(["label", "run_arm"], dropna=False):
            for col in [
                "train_total_time_s",
                "inference_batch_latency_ms",
                "inference_sample_latency_ms",
                "approx_total_flops",
                "approx_forward_flops_per_sample",
                "parameter_count",
                "peak_vram_bytes",
                "peak_vram_reserved_bytes",
            ]:
                if col not in frame.columns:
                    continue
                values = frame[col].to_numpy(dtype=float)
                mean, half = mean_ci(values, ci)
                timing_agg.append(
                    {
                        "label": str(label),
                        "run_arm": str(run_arm),
                        "field": col,
                        "n": int(np.isfinite(values).sum()),
                        "mean": mean,
                        "ci_half": half,
                    }
                )
        p2 = out_dir / "probe_timing_mean_ci.csv"
        pd.DataFrame(timing_agg).sort_values(["label", "run_arm", "field"]).to_csv(p2, index=False)
        paths["probe_timing_mean_ci"] = str(p2)

    if histories:
        curves = pd.concat(histories, ignore_index=True)
        p3 = out_dir / "fno_learning_curves.csv"
        curves.to_csv(p3, index=False)
        paths["fno_learning_curves"] = str(p3)

        curve_agg: list[dict[str, object]] = []
        gb = ["run_arm", "label", "epoch"] if "run_arm" in curves.columns else ["label", "epoch"]
        for key, frame in curves.groupby(gb, dropna=False):
            if len(gb) == 3:
                run_arm, label, epoch = key
            else:
                label, epoch = key
                run_arm = None
            for col in ["train_loss", "val_loss", "epoch_time_s"]:
                if col not in frame.columns:
                    continue
                values = frame[col].to_numpy(dtype=float)
                mean, half = mean_ci(values, ci)
                row = {
                    "label": str(label),
                    "epoch": float(epoch),
                    "field": col,
                    "n": int(np.isfinite(values).sum()),
                    "mean": mean,
                    "ci_half": half,
                }
                if run_arm is not None:
                    row["run_arm"] = str(run_arm)
                curve_agg.append(row)
        sort_keys = ["run_arm", "label", "epoch", "field"] if "run_arm" in curves.columns else ["label", "epoch", "field"]
        p4 = out_dir / "fno_learning_curves_mean_ci.csv"
        pd.DataFrame(curve_agg).sort_values(sort_keys).to_csv(p4, index=False)
        paths["fno_learning_curves_mean_ci"] = str(p4)

    if metric_rows:
        mr = pd.DataFrame(metric_rows)
        cost_rows: list[dict[str, object]] = []
        for (run_arm, label), frame in mr.groupby(["run_arm", "label"], dropna=False):
            mcol = str(frame["metric_col"].iloc[0])
            denom = frame["train_total_time_s"].to_numpy(dtype=float)
            num = frame["metric_value"].to_numpy(dtype=float)
            ratio = num / denom
            mean, half = mean_ci(ratio, ci)
            cost_rows.append(
                {
                    "run_arm": str(run_arm),
                    "label": str(label),
                    "field": f"{mcol}_per_train_s",
                    "n": int(np.isfinite(ratio).sum()),
                    "mean": mean,
                    "ci_half": half,
                }
            )
        p5 = out_dir / "cost_efficiency_mean_ci.csv"
        pd.DataFrame(cost_rows).sort_values(["run_arm", "label"]).to_csv(p5, index=False)
        paths["cost_efficiency_mean_ci"] = str(p5)

    return paths


def load_fno_best_per_label(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].astype(str) == "fno"]
    if df.empty:
        return df
    if "test_accuracy" in df.columns and df["test_accuracy"].notna().any():
        metric = "test_accuracy"
    else:
        metric = "test_r2"
    idx = df.groupby(["label"], dropna=False)[metric].idxmax()
    idx = idx.dropna()
    if idx.empty:
        return df.iloc[0:0].copy()
    return df.loc[idx].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wide_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        layout = [
            ("pretrained_lp", args.runs_root / f"operational_pretrained_seed{seed}" / "probes" / "probe_results.csv"),
            ("random_init_lp", args.runs_root / f"operational_random_init_seed{seed}" / "probes" / "probe_results.csv"),
            ("fno", args.runs_root / f"operational_fno_seed{seed}" / "probes" / "probe_results.csv"),
        ]
        for run_name, csv_path in layout:
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["seed"] = seed
            df["run_arm"] = run_name

            if run_name == "fno":
                best = load_fno_best_per_label(csv_path)
                for _, row in best.iterrows():
                    r = row.to_dict()
                    r["seed"] = seed
                    r["run_arm"] = run_name
                    wide_rows.append(r)
                continue

            matched = extract_issue3_matched_linear_rows(df)
            if matched.empty:
                continue
            matched = matched.copy()
            matched["seed"] = seed
            matched["run_arm"] = run_name
            for _, row in matched.iterrows():
                wide_rows.append(row.to_dict())

    wide_df = pd.DataFrame(wide_rows)
    wide_path = args.out_dir / "pr2_issue3_matched_per_seed.csv"
    wide_df.to_csv(wide_path, index=False)

    # Mean CI for operational labels on matched LP arms + FNO (per label).
    agg_rows: list[dict[str, object]] = []
    if not wide_df.empty:
        for (arm, label), frame in wide_df.groupby(["run_arm", "label"], dropna=False):
            if frame.empty:
                continue
            if str(arm) in {"pretrained_lp", "random_init_lp"}:
                mcol = str(frame.iloc[0].get("issue3_matched_metric_name", "test_r2"))
                if mcol not in frame.columns:
                    mcol = "test_r2" if frame["test_r2"].notna().any() else "test_accuracy"
            else:
                if "test_accuracy" in frame.columns and frame["test_accuracy"].notna().any():
                    mcol = "test_accuracy"
                else:
                    mcol = "test_r2"
            values = frame[mcol].to_numpy(dtype=float)
            mean, half = mean_ci(values, args.ci)
            agg_rows.append(
                {
                    "run_arm": str(arm),
                    "label": str(label),
                    "metric": mcol,
                    "n": int(np.isfinite(values).sum()),
                    "mean": mean,
                    "ci_half": half,
                }
            )

    pd.DataFrame(agg_rows).sort_values(["label", "run_arm"]).to_csv(args.out_dir / "pr2_issue3_matched_mean_ci.csv", index=False)

    cost_paths: dict[str, str] = {}
    notes: dict[str, str] = {}
    if not args.skip_cost_metrics:
        timing_rows, histories, metric_rows = collect_pr2_cost_tables(args.runs_root, list(args.seeds))
        cost_paths = write_cost_outputs(args.out_dir, timing_rows, histories, metric_rows, args.ci)
        notes = {
            "learning_curves": "FNO only (epoch history in fno_learning_curves*.csv); linear probes use sklearn (no epochs).",
            "linear_vram": "Linear artifacts use CPU sklearn; peak_vram_bytes is 0 unless future instrumentation adds GPU.",
        }

    (args.out_dir / "aggregate_pr2.json").write_text(
        json.dumps(
            {
                "runs_root": str(args.runs_root),
                "seeds": args.seeds,
                "wide_csv": str(wide_path),
                "cost_outputs": cost_paths,
                "notes": notes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
