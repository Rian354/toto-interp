from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp import ActivationBatch, fit_probe
from toto_interp.defaults import default_dynamic_label_specs, default_label_specs, default_taxonomy_label_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear interpretability probes on Toto activation dumps.")
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-group",
        choices=("all", "taxonomy", "dynamic"),
        default="all",
    )
    return parser.parse_args()


def choose_label_specs(group: str):
    if group == "taxonomy":
        return default_taxonomy_label_specs()
    if group == "dynamic":
        return default_dynamic_label_specs()
    return default_label_specs()


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-6)
    unit = vectors / norms
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = [ActivationBatch.load(path) for path in args.activation_files]
    activation_batch = ActivationBatch.concatenate(batches)
    label_specs = choose_label_specs(args.label_group)

    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    vector_names: list[str] = []

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
                        artifact = fit_probe(subset, label_spec)
                    except ValueError:
                        continue

                    artifact_path = artifact_dir / (
                        f"{label_spec.name}__layer_{layer}__{token_position}__{pooling_mode}.pt"
                    )
                    artifact.save(artifact_path)

                    row = {
                        "label": label_spec.name,
                        "task_type": label_spec.task_type,
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

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["label", "layer", "token_position", "pooling_mode"])
    results_df.to_csv(args.output_dir / "probe_results.csv", index=False)

    if vectors:
        vector_matrix = np.nan_to_num(np.stack(vectors), nan=0.0, posinf=0.0, neginf=0.0)
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

    if not results_df.empty:
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
