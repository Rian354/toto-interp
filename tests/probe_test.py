from __future__ import annotations

import torch

from toto_interp import ActivationBatch, LabelSpec, WindowDataset, fit_fno_probe, fit_probe, score_probe
from toto_interp.fno import FNOConfig
from toto_interp.labels import RAW_FEATURE_NAMES
from toto_interp.types import WindowExample


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


def test_window_dataset_from_windows_pads_variable_variate_counts():
    base_window = WindowExample(
        series_id="series-a",
        window_id="series-a:0",
        split="train",
        context=torch.ones(2, 8),
        next_patch=torch.ones(2, 4),
        patch_size=4,
        freq="1min",
        item_id="item-a",
        num_target_variates=2,
        labels={"shift_risk": 0.1, "band": "low"},
    )
    second_window = WindowExample(
        series_id="series-b",
        window_id="series-b:0",
        split="val",
        context=torch.ones(3, 8),
        next_patch=torch.ones(3, 4),
        patch_size=4,
        freq="1min",
        item_id="item-b",
        num_target_variates=3,
        labels={"shift_risk": 0.2, "band": "high"},
    )

    dataset = WindowDataset.from_windows([base_window, second_window])

    assert dataset.contexts.shape == (2, 3, 8)
    assert dataset.next_patches.shape == (2, 3, 4)
    assert dataset.variate_mask.tolist() == [[True, True, False], [True, True, True]]


def test_fit_fno_probe_returns_method_tagged_artifact():
    windows = []
    for idx in range(24):
        split = "train" if idx < 16 else "val" if idx < 20 else "test"
        value = float(idx)
        windows.append(
            WindowExample(
                series_id=f"series-{idx}",
                window_id=f"series-{idx}:0",
                split=split,
                context=torch.full((2, 8), value, dtype=torch.float32),
                next_patch=torch.full((2, 4), value + 1.0, dtype=torch.float32),
                patch_size=4,
                freq="1min",
                item_id=f"item-{idx}",
                num_target_variates=2,
                labels={"shift_risk": value, "band": "high" if idx >= 12 else "low"},
            )
        )
    dataset = WindowDataset.from_windows(windows)

    artifact = fit_fno_probe(
        dataset,
        LabelSpec(name="shift_risk", task_type="continuous"),
        config=FNOConfig(epochs=2, batch_size=8, width=8, modes=4, layers=2, seed=0),
    )

    assert artifact.method == "fno"
    assert artifact.layer == -2
    assert "test_r2" in artifact.metrics
    assert artifact.raw_baseline_feature_names == RAW_FEATURE_NAMES
