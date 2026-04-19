from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp import ActivationBatch, fit_probe
from toto_interp.defaults import (
    default_dynamic_label_specs,
    default_label_specs,
    default_operational_label_specs,
    default_taxonomy_label_specs,
)
from toto_interp.fno import FNOConfig, fit_fno_probe
from toto_interp.loader import resolve_device
from toto_interp.types import WindowDataset

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.linear_model\._base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear interpretability probes on Toto activation dumps.")
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--window-files", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("linear_probe", "fno"), default="linear_probe")
    parser.add_argument(
        "--label-group",
        choices=("all", "taxonomy", "dynamic", "operational"),
        default="all",
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--weight-source", choices=("pretrained", "random_init", "checkpoint"), default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--randomize-scope", choices=("full", "selected_layers", "head_only"), default=None)
    parser.add_argument("--randomize-layers", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def choose_label_specs(group: str):
    if group == "taxonomy":
        return default_taxonomy_label_specs()
    if group == "dynamic":
        return default_dynamic_label_specs()
    if group == "operational":
        return default_operational_label_specs()
    return default_label_specs()


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=1e3, neginf=-1e3)
    vectors = np.clip(vectors, -1e3, 1e3)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-6)
    unit = vectors / norms
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return unit @ unit.T


def select_best_localization_row(frame: pd.DataFrame) -> pd.Series | None:
    if "test_accuracy" in frame.columns:
        valid_accuracy = frame["test_accuracy"].dropna()
        if not valid_accuracy.empty:
            return frame.loc[valid_accuracy.idxmax()]

    if "test_r2" in frame.columns:
        valid_r2 = frame["test_r2"].dropna()
        if not valid_r2.empty:
            return frame.loc[valid_r2.idxmax()]

    return None


def resolve_window_files(args: argparse.Namespace) -> list[Path]:
    if args.window_files:
        return list(args.window_files)
    inferred: list[Path] = []
    for activation_path in args.activation_files:
        name = activation_path.name
        if not name.endswith("_activations.pt"):
            raise ValueError(
                "Could not infer window dataset path from activation file "
                f"{activation_path}. Provide --window-files explicitly."
            )
        candidate = activation_path.with_name(name.replace("_activations.pt", "_windows.pt"))
        if not candidate.exists():
            raise FileNotFoundError(f"Expected inferred window dataset file at {candidate}")
        inferred.append(candidate)
    return inferred


def resolve_metadata(args: argparse.Namespace, source_metadata: dict[str, object]) -> dict[str, object]:
    return {
        "model_id": args.model_id or str(source_metadata.get("model_id", "Datadog/Toto-Open-Base-1.0")),
        "weight_source": args.weight_source or str(source_metadata.get("weight_source", "pretrained")),
        "checkpoint_path": (
            None
            if args.checkpoint_path is None
            else str(args.checkpoint_path)
        )
        or source_metadata.get("checkpoint_path"),
        "randomize_scope": args.randomize_scope or source_metadata.get("randomize_scope"),
        "randomize_layers": tuple(args.randomize_layers or source_metadata.get("randomize_layers") or ()),
        "seed": args.seed if args.seed != 0 else int(source_metadata.get("seed", 0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = [ActivationBatch.load(path) for path in args.activation_files]
    activation_batch = ActivationBatch.concatenate(batches)
    label_specs = choose_label_specs(args.label_group)
    metadata = resolve_metadata(args, activation_batch.source_metadata)

    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    vector_names: list[str] = []

    if args.method == "linear_probe":
        unique_layers = sorted(set(activation_batch.layer_indices.tolist()))
        unique_token_positions = sorted(set(activation_batch.token_positions))
        unique_pooling_modes = sorted(set(activation_batch.pooling_modes))

        for label_spec in label_specs:
            allowed_pooling_modes = unique_pooling_modes
            if label_spec.task_type == "categorical":
                allowed_pooling_modes = [mode for mode in unique_pooling_modes if mode == "series_mean"]

            for layer in unique_layers:
                for token_position in unique_token_positions:
                    for pooling_mode in allowed_pooling_modes:
                        subset = activation_batch.subset(
                            layer=layer,
                            token_position=token_position,
                            pooling_mode=pooling_mode,
                        )
                        if len(subset) == 0:
                            continue

                        try:
                            artifact = fit_probe(
                                subset,
                                label_spec,
                                method="linear_probe",
                                model_id=str(metadata["model_id"]),
                                weight_source=str(metadata["weight_source"]),
                                backbone_train_mode="frozen",
                                checkpoint_path=(
                                    None if metadata["checkpoint_path"] is None else str(metadata["checkpoint_path"])
                                ),
                                randomize_scope=(
                                    None if metadata["randomize_scope"] is None else str(metadata["randomize_scope"])
                                ),
                                randomize_layers=tuple(metadata["randomize_layers"]),
                                seed=int(metadata["seed"]),
                            )
                        except ValueError:
                            continue

                        artifact_path = artifact_dir / (
                            f"{label_spec.name}__layer_{layer}__{token_position}__{pooling_mode}.pt"
                        )
                        artifact.save(artifact_path)

                        row = {
                            "label": label_spec.name,
                            "task_type": label_spec.task_type,
                            "method": artifact.method,
                            "model_id": artifact.model_id,
                            "weight_source": artifact.weight_source,
                            "backbone_train_mode": artifact.backbone_train_mode,
                            "checkpoint_path": artifact.checkpoint_path,
                            "randomize_scope": artifact.randomize_scope,
                            "randomize_layers": json.dumps(list(artifact.randomize_layers or ())),
                            "seed": artifact.seed,
                            "layer": layer,
                            "token_position": token_position,
                            "pooling_mode": pooling_mode,
                            "artifact_path": str(artifact_path),
                        }
                        row.update(artifact.metrics)
                        row.update({f"baseline_{k}": v for k, v in artifact.baseline_metrics.items()})
                        row.update({f"shuffled_{k}": v for k, v in artifact.shuffled_metrics.items()})
                        rows.append(row)

                        if artifact.mean_difference_vector is not None:
                            vector_name = f"{label_spec.name}__layer_{layer}__{token_position}__{pooling_mode}"
                            vectors.append(artifact.mean_difference_vector.cpu().numpy())
                            vector_names.append(vector_name)
                            vector_rows.append(
                                {
                                    "name": vector_name,
                                    "label": label_spec.name,
                                    "layer": layer,
                                    "token_position": token_position,
                                    "pooling_mode": pooling_mode,
                                }
                            )
    else:
        window_datasets = [WindowDataset.load(path) for path in resolve_window_files(args)]
        window_dataset = WindowDataset.concatenate(window_datasets)
        fno_config = FNOConfig(
            modes=args.fno_modes,
            width=args.fno_width,
            layers=args.fno_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=resolve_device(args.device),
            seed=int(metadata["seed"]),
        )
        for label_spec in label_specs:
            try:
                artifact = fit_fno_probe(
                    window_dataset,
                    label_spec,
                    config=fno_config,
                    model_id=str(metadata["model_id"]),
                    weight_source=str(metadata["weight_source"]),
                    backbone_train_mode="frozen",
                    checkpoint_path=None if metadata["checkpoint_path"] is None else str(metadata["checkpoint_path"]),
                )
            except ValueError:
                continue

            artifact_path = artifact_dir / f"{label_spec.name}__fno__window.pt"
            artifact.save(artifact_path)
            row = {
                "label": label_spec.name,
                "task_type": label_spec.task_type,
                "method": artifact.method,
                "model_id": artifact.model_id,
                "weight_source": artifact.weight_source,
                "backbone_train_mode": artifact.backbone_train_mode,
                "checkpoint_path": artifact.checkpoint_path,
                "randomize_scope": artifact.randomize_scope,
                "randomize_layers": json.dumps(list(artifact.randomize_layers or ())),
                "seed": artifact.seed,
                "layer": artifact.layer,
                "token_position": artifact.token_position,
                "pooling_mode": artifact.pooling_mode,
                "artifact_path": str(artifact_path),
            }
            row.update(artifact.metrics)
            row.update({f"baseline_{k}": v for k, v in artifact.baseline_metrics.items()})
            row.update({f"shuffled_{k}": v for k, v in artifact.shuffled_metrics.items()})
            rows.append(row)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["method", "label", "layer", "token_position", "pooling_mode"])
    results_df.to_csv(args.output_dir / "probe_results.csv", index=False)

    if vectors:
        vector_matrix = np.nan_to_num(np.stack(vectors), nan=0.0, posinf=1e3, neginf=-1e3)
        vector_matrix = np.clip(vector_matrix, -1e3, 1e3)
        cosine_df = pd.DataFrame(cosine_similarity_matrix(vector_matrix), index=vector_names, columns=vector_names)
        cosine_df.to_csv(args.output_dir / "probe_geometry_cosine.csv")

        pca_df = pd.DataFrame(vector_rows)
        meta: dict[str, object] = {"vector_count": len(vectors)}
        has_variance = vector_matrix.shape[0] >= 2 and bool(np.any(np.var(vector_matrix, axis=0) > 0))
        if has_variance:
            pca = PCA(n_components=min(3, len(vectors), vector_matrix.shape[1]))
            projected = pca.fit_transform(vector_matrix)
            for idx in range(projected.shape[1]):
                pca_df[f"pc_{idx + 1}"] = projected[:, idx]
            meta["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        else:
            meta["explained_variance_ratio"] = []
            meta["pca_skipped"] = "insufficient finite variance"
        pca_df.to_csv(args.output_dir / "probe_geometry_pca.csv", index=False)
        with open(args.output_dir / "probe_geometry_meta.json", "w") as handle:
            json.dump(meta, handle, indent=2)

    if not results_df.empty and args.method == "linear_probe":
        localization_rows: list[pd.Series] = []
        for _, frame in results_df.groupby(["label", "token_position", "pooling_mode"], dropna=False):
            selected = select_best_localization_row(frame)
            if selected is not None:
                localization_rows.append(selected)
        if localization_rows:
            localization_summary = pd.DataFrame(localization_rows).reset_index(drop=True)
            localization_summary.to_csv(args.output_dir / "probe_localization_summary.csv", index=False)


if __name__ == "__main__":
    main()
