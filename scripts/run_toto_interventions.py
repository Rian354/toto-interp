from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.bootstrap import ensure_toto_importable

ensure_toto_importable()

from toto.inference.forecaster import TotoForecaster
from toto_interp import InterventionConfig, ProbeArtifact, TraceConfig, apply_intervention, extract_activations
from toto_interp.boom import (
    build_boom_windows,
    build_masked_timeseries,
    ensure_boom_snapshot,
    load_boom_taxonomy,
    split_boom_series_ids,
)
from toto_interp.loader import load_toto_with_fallback
from toto_interp.loader import resolve_device
from toto_interp.metrics import mase, wape, weighted_quantile_loss
from toto_interp.types import WindowExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal intervention sweeps on Toto concept directions.")
    parser.add_argument("--probe-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-series", type=int, default=64)
    parser.add_argument("--max-windows-per-series", type=int, default=4)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--mode", choices=("ablate", "steer"), default="steer")
    parser.add_argument("--strengths", type=float, nargs="+", default=[-0.05, -0.02, 0.02, 0.05])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--random-direction", action="store_true")
    return parser.parse_args()


def probe_score_from_batch(batch, probe: ProbeArtifact) -> float:
    features = batch.activations
    standardized = (features - probe.feature_mean) / probe.feature_std
    score = standardized @ probe.coef[0].unsqueeze(-1)
    score = score.squeeze(-1) + probe.intercept[0]
    return float(score.mean().item())


def choose_subset_tag(window: WindowExample, probe: ProbeArtifact) -> str:
    if probe.positive_threshold is None or probe.negative_threshold is None:
        return "all"
    value = float(window.labels[probe.label_spec.name])
    if value >= probe.positive_threshold:
        return "high"
    if value <= probe.negative_threshold:
        return "low"
    return "middle"


def choose_samples_per_batch(num_samples: int, preferred: int = 10) -> int:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    samples_per_batch = min(num_samples, preferred)
    while samples_per_batch > 1 and num_samples % samples_per_batch != 0:
        samples_per_batch -= 1
    return samples_per_batch


def forecast_window(
    forecaster: TotoForecaster,
    window: WindowExample,
    *,
    num_samples: int,
    prediction_length: int,
    use_kv_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = build_masked_timeseries(window.context, prediction_length).to(forecaster.model.device)
    samples_per_batch = choose_samples_per_batch(num_samples)
    forecast = forecaster.forecast(
        inputs=masked,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
        use_kv_cache=use_kv_cache,
    )
    samples = forecast.samples
    if samples is None:
        raise ValueError("Intervention evaluation requires sampled forecasts.")
    median = forecast.median
    iqr = forecast.quantile(0.9) - forecast.quantile(0.1)
    return median.cpu(), iqr.cpu(), samples.cpu()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probe = ProbeArtifact.load(args.probe_path)
    if probe.label_spec.task_type != "continuous":
        raise ValueError("run_toto_interventions.py currently supports continuous regime probes only.")
    if probe.layer < 0:
        raise ValueError("run_toto_interventions.py only supports transformer-layer probes, not patch embeddings.")

    device = resolve_device(args.device)
    model = load_toto_with_fallback(args.model_id, map_location="cpu", device=device)
    if args.compile and hasattr(model, "compile"):
        model.compile()
    backbone = model.model
    forecaster = TotoForecaster(backbone)
    patch_size = int(backbone.patch_embed.patch_size)

    if args.context_length % patch_size != 0:
        raise ValueError(f"context-length must be divisible by patch size ({patch_size}).")

    taxonomy = load_boom_taxonomy()
    split_ids = split_boom_series_ids(taxonomy, seed=args.seed)
    series_ids = split_ids[args.split][: args.max_series]
    snapshot_path = ensure_boom_snapshot(series_ids)
    windows = build_boom_windows(
        series_ids=series_ids,
        split=args.split,
        snapshot_path=snapshot_path,
        taxonomy=taxonomy,
        context_length=args.context_length,
        patch_size=patch_size,
        max_windows_per_series=args.max_windows_per_series,
        include_heldout_late=True,
    )

    direction = probe.mean_difference_vector
    if direction is None:
        raise ValueError("Probe artifact does not contain a mean-difference vector for causal interventions.")
    if args.random_direction:
        direction = torch.randn_like(direction)

    trace_config = TraceConfig(
        layers=(probe.layer,) if probe.layer >= 0 else (),
        token_positions=("first_decode",),
        pooling_modes=(probe.pooling_mode,),
        capture_patch_embedding=probe.layer == -1,
        use_kv_cache=not args.disable_kv_cache,
    )

    baseline_rows: list[dict[str, object]] = []
    for window in windows:
        base_median, base_iqr, base_samples = forecast_window(
            forecaster,
            window,
            num_samples=args.num_samples,
            prediction_length=patch_size,
            use_kv_cache=not args.disable_kv_cache,
        )
        subset_tag = choose_subset_tag(window, probe)
        base_activation_batch = extract_activations(model, [window], trace_config)
        base_probe_score = probe_score_from_batch(base_activation_batch, probe)

        quantile_predictions = {
            0.1: base_samples.quantile(0.1, dim=-1),
            0.5: base_samples.quantile(0.5, dim=-1),
            0.9: base_samples.quantile(0.9, dim=-1),
        }
        baseline_rows.append(
            {
                "window_id": window.window_id,
                "subset": subset_tag,
                "strength": 0.0,
                "mode": "baseline",
                "probe_score": base_probe_score,
                "forecast_regime_value": float(window.labels[probe.label_spec.name]),
                "wape": wape(window.next_patch, base_median.squeeze(0)),
                "mase": mase(window.context, window.next_patch, base_median.squeeze(0)),
                "wql": weighted_quantile_loss(window.next_patch, quantile_predictions),
                "median_spread": float(base_iqr.mean().item()),
                "median_forecast_change": 0.0,
            }
        )

        for strength in args.strengths:
            intervention = InterventionConfig(
                layer_indices=(probe.layer,) if probe.layer >= 0 else (-1,),
                token_position=probe.token_position,  # type: ignore[arg-type]
                mode=args.mode,  # type: ignore[arg-type]
                vector=direction,
                strength=float(strength),
                decode_steps=(1,),
            )

            with apply_intervention(backbone, intervention):
                intervention_median, intervention_iqr, intervention_samples = forecast_window(
                    forecaster,
                    window,
                    num_samples=args.num_samples,
                    prediction_length=patch_size,
                    use_kv_cache=not args.disable_kv_cache,
                )
                intervention_activation_batch = extract_activations(model, [window], trace_config)
                intervention_probe_score = probe_score_from_batch(intervention_activation_batch, probe)

            intervention_quantiles = {
                0.1: intervention_samples.quantile(0.1, dim=-1),
                0.5: intervention_samples.quantile(0.5, dim=-1),
                0.9: intervention_samples.quantile(0.9, dim=-1),
            }
            baseline_rows.append(
                {
                    "window_id": window.window_id,
                    "subset": subset_tag,
                    "strength": strength,
                    "mode": args.mode,
                    "probe_score": intervention_probe_score,
                    "forecast_regime_value": float(window.labels[probe.label_spec.name]),
                    "wape": wape(window.next_patch, intervention_median.squeeze(0)),
                    "mase": mase(window.context, window.next_patch, intervention_median.squeeze(0)),
                    "wql": weighted_quantile_loss(window.next_patch, intervention_quantiles),
                    "median_spread": float(intervention_iqr.mean().item()),
                    "median_forecast_change": float((intervention_median - base_median).abs().mean().item()),
                }
            )

    results_df = pd.DataFrame(baseline_rows)
    results_df.to_csv(args.output_dir / "intervention_window_metrics.csv", index=False)

    summary_df = (
        results_df.groupby(["mode", "strength", "subset"], dropna=False)
        .agg(
            probe_score=("probe_score", "mean"),
            wape=("wape", "mean"),
            mase=("mase", "mean"),
            wql=("wql", "mean"),
            median_spread=("median_spread", "mean"),
            median_forecast_change=("median_forecast_change", "mean"),
            window_count=("window_id", "count"),
        )
        .reset_index()
    )
    summary_df.to_csv(args.output_dir / "intervention_summary.csv", index=False)

    with open(args.output_dir / "intervention_meta.json", "w") as handle:
        json.dump(
            {
                "probe_path": str(args.probe_path),
                "label": probe.label_spec.name,
                "mode": args.mode,
                "strengths": args.strengths,
                "split": args.split,
                "window_count": len(windows),
                "device": device,
                "use_kv_cache": not args.disable_kv_cache,
                "random_direction": args.random_direction,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
