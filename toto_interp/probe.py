from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from .types import ActivationBatch, LabelSpec, ProbeArtifact, TaskType


def _unique_value(values: list[Any], name: str) -> Any:
    unique_values = sorted(set(values))
    if len(unique_values) != 1:
        raise ValueError(f"fit_probe expects a single {name}; found {unique_values}. Use ActivationBatch.subset first.")
    return unique_values[0]


def _split_mask(splits: list[str], split_name: str) -> np.ndarray:
    return np.asarray([split == split_name for split in splits], dtype=bool)


def _standardize_train(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    r2 = float("nan") if y_true.shape[0] < 2 else float(r2_score(y_true, y_pred))
    return {
        f"{prefix}_r2": r2,
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _predict_categorical(
    features: np.ndarray,
    *,
    coef: torch.Tensor,
    intercept: torch.Tensor,
    class_names: tuple[str, ...],
) -> np.ndarray:
    coef_np = coef.cpu().numpy()
    intercept_np = intercept.cpu().numpy()
    if coef_np.shape[0] == 1 and len(class_names) == 2:
        decision = features @ coef_np[0] + intercept_np[0]
        return np.where(decision > 0, class_names[1], class_names[0])

    logits = features @ coef_np.T + intercept_np
    indices = logits.argmax(axis=1)
    return np.asarray([class_names[index] for index in indices], dtype=object)


def score_probe(
    activation_batch: ActivationBatch,
    probe_artifact: ProbeArtifact,
    *,
    prefix: str = "transfer",
) -> dict[str, float]:
    subset = activation_batch.subset(
        layer=probe_artifact.layer,
        token_position=probe_artifact.token_position,
        pooling_mode=probe_artifact.pooling_mode,
    )
    if len(subset) == 0:
        raise ValueError("No activation records matched the probe view for scoring.")

    X = subset.activations.cpu().numpy()
    X_raw = subset.raw_features.cpu().numpy()
    y = subset.label_array(probe_artifact.label_spec.name)

    valid_mask = np.ones(len(subset), dtype=bool)
    if probe_artifact.label_spec.task_type == "categorical":
        if probe_artifact.class_names is None:
            raise ValueError("Categorical probe artifacts require class_names for scoring.")
        valid_mask &= np.asarray([str(value) in probe_artifact.class_names for value in y], dtype=bool)
    else:
        valid_mask &= np.isfinite(y.astype(np.float64))

    if not valid_mask.any():
        raise ValueError(f"No valid labels available to score probe {probe_artifact.label_spec.name!r}.")

    X = X[valid_mask]
    X_raw = X_raw[valid_mask]
    y = y[valid_mask]

    if probe_artifact.feature_mean is None or probe_artifact.feature_std is None:
        raise ValueError("Probe artifact is missing activation standardization statistics.")

    X_z = _standardize(X, probe_artifact.feature_mean.cpu().numpy(), probe_artifact.feature_std.cpu().numpy())

    metrics: dict[str, float]
    if probe_artifact.label_spec.task_type == "categorical":
        y_true = np.asarray(y, dtype=object)
        y_pred = _predict_categorical(
            X_z,
            coef=probe_artifact.coef,
            intercept=probe_artifact.intercept,
            class_names=probe_artifact.class_names or (),
        )
        metrics = _categorical_metrics(y_true, y_pred, prefix)

        if (
            probe_artifact.raw_feature_mean is not None
            and probe_artifact.raw_feature_std is not None
            and probe_artifact.raw_baseline_coef is not None
            and probe_artifact.raw_baseline_intercept is not None
            and probe_artifact.class_names is not None
        ):
            X_raw_z = _standardize(
                X_raw,
                probe_artifact.raw_feature_mean.cpu().numpy(),
                probe_artifact.raw_feature_std.cpu().numpy(),
            )
            baseline_pred = _predict_categorical(
                X_raw_z,
                coef=probe_artifact.raw_baseline_coef,
                intercept=probe_artifact.raw_baseline_intercept,
                class_names=probe_artifact.class_names,
            )
            metrics.update({f"baseline_{k}": v for k, v in _categorical_metrics(y_true, baseline_pred, prefix).items()})
    else:
        y_true = y.astype(np.float64)
        coef_np = probe_artifact.coef[0].cpu().numpy()
        intercept_np = float(probe_artifact.intercept[0].cpu().item())
        y_pred = X_z @ coef_np + intercept_np
        metrics = _continuous_metrics(y_true, y_pred, prefix)

        if (
            probe_artifact.raw_feature_mean is not None
            and probe_artifact.raw_feature_std is not None
            and probe_artifact.raw_baseline_coef is not None
            and probe_artifact.raw_baseline_intercept is not None
        ):
            X_raw_z = _standardize(
                X_raw,
                probe_artifact.raw_feature_mean.cpu().numpy(),
                probe_artifact.raw_feature_std.cpu().numpy(),
            )
            baseline_coef = probe_artifact.raw_baseline_coef[0].cpu().numpy()
            baseline_intercept = float(probe_artifact.raw_baseline_intercept[0].cpu().item())
            baseline_pred = X_raw_z @ baseline_coef + baseline_intercept
            metrics.update(
                {f"baseline_{k}": v for k, v in _continuous_metrics(y_true, baseline_pred, prefix).items()}
            )

    metrics[f"{prefix}_count"] = float(X.shape[0])
    return metrics


def fit_probe(
    activation_batch: ActivationBatch,
    label_spec: LabelSpec,
    task_type: TaskType | None = None,
) -> ProbeArtifact:
    """
    Fit a linear probe on a single activation view.

    The batch must already be narrowed to a single `(layer, token_position,
    pooling_mode)` view via `ActivationBatch.subset(...)`.
    """

    resolved_task_type = task_type or label_spec.task_type
    if resolved_task_type != label_spec.task_type:
        raise ValueError("task_type must match label_spec.task_type")

    if len(activation_batch) == 0:
        raise ValueError("Cannot fit a probe on an empty ActivationBatch.")

    layer = int(_unique_value(activation_batch.layer_indices.tolist(), "layer"))
    token_position = str(_unique_value(activation_batch.token_positions, "token position"))
    pooling_mode = str(_unique_value(activation_batch.pooling_modes, "pooling mode"))

    X = activation_batch.activations.cpu().numpy()
    X_raw = activation_batch.raw_features.cpu().numpy()
    y = activation_batch.label_array(label_spec.name)
    splits = activation_batch.splits

    valid_mask = np.ones(len(activation_batch), dtype=bool)
    class_names: tuple[str, ...] | None = None
    if resolved_task_type == "categorical":
        candidate_classes = label_spec.classes or tuple(sorted({str(value) for value in y.tolist()}))
        valid_mask &= np.asarray([str(value) in candidate_classes for value in y], dtype=bool)
        class_names = tuple(candidate_classes)
    else:
        valid_mask &= np.isfinite(y.astype(np.float64))

    if not valid_mask.any():
        raise ValueError(f"No valid examples found for label {label_spec.name!r}.")

    X = X[valid_mask]
    X_raw = X_raw[valid_mask]
    y = y[valid_mask]
    filtered_splits = [split for split, keep in zip(splits, valid_mask.tolist()) if keep]

    train_mask = _split_mask(filtered_splits, "train")
    val_mask = _split_mask(filtered_splits, "val")
    test_mask = _split_mask(filtered_splits, "test")
    if not train_mask.any():
        raise ValueError("fit_probe requires at least one training example.")

    feature_mean, feature_std = _standardize_train(X[train_mask])
    raw_mean, raw_std = _standardize_train(X_raw[train_mask])
    X_z = _standardize(X, feature_mean, feature_std)
    X_raw_z = _standardize(X_raw, raw_mean, raw_std)

    metrics: dict[str, float] = {}
    baseline_metrics: dict[str, float] = {}
    shuffled_metrics: dict[str, float] = {}
    mean_difference_vector: torch.Tensor | None = None
    positive_threshold: float | None = None
    negative_threshold: float | None = None

    rng = np.random.default_rng(0)

    if resolved_task_type == "categorical":
        train_targets = np.asarray(y[train_mask], dtype=object)
        if len(np.unique(train_targets)) < 2:
            raise ValueError("Categorical probes require at least two classes in the training split.")

        logistic = LogisticRegression(
            class_weight="balanced",
            max_iter=4000,
            random_state=0,
        )
        logistic.fit(X_z[train_mask], train_targets)

        raw_logistic = LogisticRegression(
            class_weight="balanced",
            max_iter=4000,
            random_state=0,
        )
        raw_logistic.fit(X_raw_z[train_mask], train_targets)

        shuffled_logistic = LogisticRegression(
            class_weight="balanced",
            max_iter=4000,
            random_state=0,
        )
        shuffled_targets = np.asarray(train_targets, dtype=object)
        rng.shuffle(shuffled_targets)
        shuffled_logistic.fit(X_z[train_mask], shuffled_targets)

        for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            if not split_mask.any():
                continue
            metrics.update(
                _categorical_metrics(
                    np.asarray(y[split_mask], dtype=object),
                    logistic.predict(X_z[split_mask]),
                    split_name,
                )
            )
            baseline_metrics.update(
                _categorical_metrics(
                    np.asarray(y[split_mask], dtype=object),
                    raw_logistic.predict(X_raw_z[split_mask]),
                    split_name,
                )
            )
            shuffled_metrics.update(
                _categorical_metrics(
                    np.asarray(y[split_mask], dtype=object),
                    shuffled_logistic.predict(X_z[split_mask]),
                    split_name,
                )
            )

        if class_names is not None and len(class_names) == 2:
            positive_class = class_names[1]
            negative_class = class_names[0]
            positive_examples = X[train_mask][train_targets == positive_class]
            negative_examples = X[train_mask][train_targets == negative_class]
            if len(positive_examples) and len(negative_examples):
                mean_difference_vector = torch.tensor(
                    positive_examples.mean(axis=0) - negative_examples.mean(axis=0),
                    dtype=torch.float32,
                )

        coef = torch.as_tensor(logistic.coef_, dtype=torch.float32)
        intercept = torch.as_tensor(logistic.intercept_, dtype=torch.float32)
        class_names = tuple(str(name) for name in logistic.classes_)
    else:
        targets = y.astype(np.float64)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_z[train_mask], targets[train_mask])

        raw_ridge = Ridge(alpha=1.0)
        raw_ridge.fit(X_raw_z[train_mask], targets[train_mask])

        shuffled_ridge = Ridge(alpha=1.0)
        shuffled_targets = np.asarray(targets[train_mask], dtype=np.float64)
        rng.shuffle(shuffled_targets)
        shuffled_ridge.fit(X_z[train_mask], shuffled_targets)

        for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            if not split_mask.any():
                continue
            metrics.update(_continuous_metrics(targets[split_mask], ridge.predict(X_z[split_mask]), split_name))
            baseline_metrics.update(
                _continuous_metrics(targets[split_mask], raw_ridge.predict(X_raw_z[split_mask]), split_name)
            )
            shuffled_metrics.update(
                _continuous_metrics(targets[split_mask], shuffled_ridge.predict(X_z[split_mask]), split_name)
            )

        positive_threshold = float(np.quantile(targets[train_mask], label_spec.positive_quantile))
        negative_threshold = float(np.quantile(targets[train_mask], label_spec.negative_quantile))

        positive_examples = X[train_mask][targets[train_mask] >= positive_threshold]
        negative_examples = X[train_mask][targets[train_mask] <= negative_threshold]
        if len(positive_examples) and len(negative_examples):
            mean_difference_vector = torch.tensor(
                positive_examples.mean(axis=0) - negative_examples.mean(axis=0),
                dtype=torch.float32,
            )

        coef = torch.as_tensor(np.atleast_2d(ridge.coef_), dtype=torch.float32)
        intercept = torch.as_tensor(np.atleast_1d(ridge.intercept_), dtype=torch.float32)

    return ProbeArtifact(
        label_spec=label_spec,
        layer=layer,
        token_position=token_position,
        pooling_mode=pooling_mode,
        coef=coef,
        intercept=intercept,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        shuffled_metrics=shuffled_metrics,
        class_names=class_names,
        mean_difference_vector=mean_difference_vector,
        feature_mean=torch.as_tensor(feature_mean, dtype=torch.float32),
        feature_std=torch.as_tensor(feature_std, dtype=torch.float32),
        raw_feature_mean=torch.as_tensor(raw_mean, dtype=torch.float32),
        raw_feature_std=torch.as_tensor(raw_std, dtype=torch.float32),
        raw_baseline_feature_names=activation_batch.raw_feature_names,
        raw_baseline_coef=torch.as_tensor(np.atleast_2d(raw_logistic.coef_ if resolved_task_type == "categorical" else raw_ridge.coef_), dtype=torch.float32),
        raw_baseline_intercept=torch.as_tensor(np.atleast_1d(raw_logistic.intercept_ if resolved_task_type == "categorical" else raw_ridge.intercept_), dtype=torch.float32),
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )
