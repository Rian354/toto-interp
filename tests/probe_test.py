from __future__ import annotations

import torch

from toto_interp import ActivationBatch, LabelSpec, fit_probe, score_probe
from toto_interp.labels import RAW_FEATURE_NAMES


def _make_synthetic_batch() -> ActivationBatch:
    total = 30
    splits = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    activations = []
    raw_features = []
    continuous_labels = []
    categorical_labels = []

    for idx in range(total):
        value = float(idx)
        activations.append(torch.tensor([value, value * 0.5, -value], dtype=torch.float32))
        raw_features.append(torch.zeros(len(RAW_FEATURE_NAMES), dtype=torch.float32))
        continuous_labels.append(value)
        categorical_labels.append("high" if value >= 15 else "low")

    return ActivationBatch(
        activations=torch.stack(activations),
        raw_features=torch.stack(raw_features),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.full((total,), 10, dtype=torch.long),
        patch_indices=torch.zeros(total, dtype=torch.long),
        variate_indices=torch.full((total,), -1, dtype=torch.long),
        token_positions=["final_context"] * total,
        pooling_modes=["series_mean"] * total,
        series_ids=[f"series-{idx}" for idx in range(total)],
        window_ids=[f"window-{idx}" for idx in range(total)],
        splits=splits,
        labels={
            "shift_risk": continuous_labels,
            "band": categorical_labels,
        },
    )


def test_fit_probe_supports_continuous_and_categorical_labels():
    batch = _make_synthetic_batch()

    continuous_probe = fit_probe(batch, LabelSpec(name="shift_risk", task_type="continuous"))
    categorical_probe = fit_probe(
        batch,
        LabelSpec(name="band", task_type="categorical", classes=("low", "high")),
    )

    assert continuous_probe.metrics["test_r2"] > 0.9
    assert continuous_probe.mean_difference_vector is not None
    assert continuous_probe.positive_threshold is not None
    assert continuous_probe.negative_threshold is not None

    assert categorical_probe.metrics["test_accuracy"] > 0.9
    assert categorical_probe.class_names == ("high", "low") or categorical_probe.class_names == ("low", "high")


def test_score_probe_reuses_frozen_probe_and_raw_baseline():
    batch = _make_synthetic_batch()
    probe = fit_probe(batch, LabelSpec(name="shift_risk", task_type="continuous"))

    metrics = score_probe(batch, probe, prefix="transfer")

    assert metrics["transfer_r2"] > 0.9
    assert "baseline_transfer_r2" in metrics
    assert metrics["transfer_count"] == float(len(batch))
