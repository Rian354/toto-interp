from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

TokenPosition = Literal["all_context", "final_context", "first_decode"]
PoolingMode = Literal["per_variate", "series_mean"]
TaskType = Literal["categorical", "continuous"]
InterventionMode = Literal["ablate", "steer"]


@dataclass(frozen=True)
class TraceConfig:
    """
    Configuration for activation tracing.

    The implementation treats Toto's internal sequence as patch tokens, not raw
    timesteps. `layers` therefore refers to transformer block outputs, and
    `capture_patch_embedding=True` stores the patch-embedding activations with
    layer index -1.
    """

    layers: tuple[int, ...] = tuple(range(12))
    token_positions: tuple[TokenPosition, ...] = ("all_context", "final_context", "first_decode")
    pooling_modes: tuple[PoolingMode, ...] = ("per_variate", "series_mean")
    capture_patch_embedding: bool = True
    use_kv_cache: bool = True
    cache_format: Literal["pt"] = "pt"


@dataclass(frozen=True)
class WindowExample:
    """
    Single forecasting window used for probing and interventions.
    """

    series_id: str
    window_id: str
    split: str
    context: torch.Tensor
    next_patch: torch.Tensor
    patch_size: int
    freq: str
    item_id: str
    num_target_variates: int
    labels: dict[str, Any]


@dataclass
class ActivationBatch:
    """
    Flattened activation records aligned with window-level metadata and labels.
    """

    activations: torch.Tensor
    raw_features: torch.Tensor
    raw_feature_names: tuple[str, ...]
    layer_indices: torch.Tensor
    patch_indices: torch.Tensor
    variate_indices: torch.Tensor
    token_positions: list[str]
    pooling_modes: list[str]
    series_ids: list[str]
    window_ids: list[str]
    splits: list[str]
    labels: dict[str, list[Any]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.activations.shape[0])

    def save(self, path: str | Path) -> None:
        payload = {
            "activations": self.activations,
            "raw_features": self.raw_features,
            "raw_feature_names": self.raw_feature_names,
            "layer_indices": self.layer_indices,
            "patch_indices": self.patch_indices,
            "variate_indices": self.variate_indices,
            "token_positions": self.token_positions,
            "pooling_modes": self.pooling_modes,
            "series_ids": self.series_ids,
            "window_ids": self.window_ids,
            "splits": self.splits,
            "labels": self.labels,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationBatch":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**payload)

    def subset(
        self,
        *,
        layer: int | None = None,
        token_position: str | None = None,
        pooling_mode: str | None = None,
        split: str | None = None,
    ) -> "ActivationBatch":
        mask = torch.ones(len(self), dtype=torch.bool)
        if layer is not None:
            mask &= self.layer_indices == layer
        if token_position is not None:
            mask &= torch.tensor([value == token_position for value in self.token_positions], dtype=torch.bool)
        if pooling_mode is not None:
            mask &= torch.tensor([value == pooling_mode for value in self.pooling_modes], dtype=torch.bool)
        if split is not None:
            mask &= torch.tensor([value == split for value in self.splits], dtype=torch.bool)

        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return ActivationBatch(
            activations=self.activations[indices],
            raw_features=self.raw_features[indices],
            raw_feature_names=self.raw_feature_names,
            layer_indices=self.layer_indices[indices],
            patch_indices=self.patch_indices[indices],
            variate_indices=self.variate_indices[indices],
            token_positions=[self.token_positions[i] for i in indices.tolist()],
            pooling_modes=[self.pooling_modes[i] for i in indices.tolist()],
            series_ids=[self.series_ids[i] for i in indices.tolist()],
            window_ids=[self.window_ids[i] for i in indices.tolist()],
            splits=[self.splits[i] for i in indices.tolist()],
            labels={name: [values[i] for i in indices.tolist()] for name, values in self.labels.items()},
        )

    def label_array(self, label_name: str) -> np.ndarray:
        values = self.labels[label_name]
        if len(values) == 0:
            return np.empty((0,))
        sample = values[0]
        if isinstance(sample, str):
            return np.asarray(values, dtype=object)
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def concatenate(batches: list["ActivationBatch"]) -> "ActivationBatch":
        if not batches:
            raise ValueError("Cannot concatenate an empty list of ActivationBatch objects.")

        label_names = tuple(batches[0].labels.keys())
        raw_feature_names = batches[0].raw_feature_names
        return ActivationBatch(
            activations=torch.cat([batch.activations for batch in batches], dim=0),
            raw_features=torch.cat([batch.raw_features for batch in batches], dim=0),
            raw_feature_names=raw_feature_names,
            layer_indices=torch.cat([batch.layer_indices for batch in batches], dim=0),
            patch_indices=torch.cat([batch.patch_indices for batch in batches], dim=0),
            variate_indices=torch.cat([batch.variate_indices for batch in batches], dim=0),
            token_positions=[item for batch in batches for item in batch.token_positions],
            pooling_modes=[item for batch in batches for item in batch.pooling_modes],
            series_ids=[item for batch in batches for item in batch.series_ids],
            window_ids=[item for batch in batches for item in batch.window_ids],
            splits=[item for batch in batches for item in batch.splits],
            labels={name: [item for batch in batches for item in batch.labels[name]] for name in label_names},
        )


@dataclass(frozen=True)
class LabelSpec:
    name: str
    task_type: TaskType
    classes: tuple[str, ...] | None = None
    positive_quantile: float = 0.9
    negative_quantile: float = 0.1


@dataclass
class ProbeArtifact:
    """
    Serializable representation of a linear probe fit on a single activation view.
    """

    label_spec: LabelSpec
    layer: int
    token_position: str
    pooling_mode: str
    coef: torch.Tensor
    intercept: torch.Tensor
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    shuffled_metrics: dict[str, float]
    class_names: tuple[str, ...] | None = None
    mean_difference_vector: torch.Tensor | None = None
    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None
    raw_feature_mean: torch.Tensor | None = None
    raw_feature_std: torch.Tensor | None = None
    raw_baseline_feature_names: tuple[str, ...] | None = None
    raw_baseline_coef: torch.Tensor | None = None
    raw_baseline_intercept: torch.Tensor | None = None
    positive_threshold: float | None = None
    negative_threshold: float | None = None

    def save(self, path: str | Path) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbeArtifact":
        return torch.load(path, map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class InterventionConfig:
    layer_indices: tuple[int, ...]
    token_position: TokenPosition
    mode: InterventionMode
    vector: torch.Tensor
    strength: float = 0.0
    normalize_by_residual: bool = True
    decode_steps: tuple[int, ...] | None = None
