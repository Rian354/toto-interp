from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from toto_interp.transfer import build_fev_windows_from_dataset, build_lsf_windows


def test_build_fev_windows_from_dataset_supports_auto_target_detection():
    dataset = Dataset.from_dict(
        {
            "id": ["series-a", "series-b"],
            "timestamp": [
                pd.date_range("2024-01-01", periods=24, freq="h").to_numpy(),
                pd.date_range("2024-01-01", periods=24, freq="h").to_numpy(),
            ],
            "cpu": [list(range(24)), list(range(24, 48))],
            "mem": [list(range(100, 124)), list(range(124, 148))],
        }
    )
    dataset.set_format("numpy")

    windows = build_fev_windows_from_dataset(
        dataset,
        dataset_name="synthetic_fev",
        context_length=8,
        patch_size=4,
        max_series=1,
        max_windows_per_series=3,
    )

    assert len(windows) == 2
    assert windows[0].num_target_variates == 2
    assert windows[0].labels["benchmark_name"] == "fev"
    assert windows[0].labels["dataset_name"] == "synthetic_fev"


def test_build_lsf_windows_reads_local_custom_dataset(tmp_path: Path):
    data_dir = tmp_path / "electricity"
    data_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
            {
            "date": pd.date_range("2024-01-01", periods=240, freq="h"),
            "load": list(range(240)),
            "OT": list(range(500, 740)),
        }
    )
    frame.to_csv(data_dir / "electricity.csv", index=False)

    windows = build_lsf_windows(
        dataset_name="electricity",
        lsf_path=tmp_path,
        context_length=32,
        patch_size=8,
        max_series=1,
        max_windows_per_series=3,
    )

    assert len(windows) == 2
    assert windows[0].labels["benchmark_name"] == "lsf"
    assert windows[0].labels["dataset_name"] == "electricity"
    assert windows[0].num_target_variates == 2
