"""Canonical (layer, token_position, pooling_mode) readouts for Issue #3-style comparisons.

Use these rows for apples-to-apples tables instead of argmax over the full probe grid.
Reference metrics are copied from ``postrun_status.md`` (paper / Issue #3 baselines).

**Reproducing Issue #3 scale** (see ``postrun_status.md`` /
``slurm/dump_standardized_activations.sh``): 500 series/split, 4 windows, 16 samples,
context 1024, and ``label_group="all"`` (taxonomy + dynamic) so every label in
``ISSUE3_LINEAR_SPECS`` is trained.
"""

from __future__ import annotations

# --- Issue #3 experiment scale (paper / standardized activations) ---

ISSUE3_CONTEXT_LENGTH = 1024
ISSUE3_MAX_SERIES_PER_SPLIT = 500
ISSUE3_MAX_WINDOWS_PER_SERIES = 4
ISSUE3_NUM_SAMPLES = 16
ISSUE3_LABEL_GROUP = "all"

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Issue3LinearSpec:
    label: str
    layer: int
    token_position: str
    pooling_mode: str
    issue3_test_metric: float
    """Reference value: test_accuracy for categoricals, test_r2 for continuous labels."""


# Token choices follow internal ``postrun_status.md`` / probe conventions.
ISSUE3_LINEAR_SPECS: tuple[Issue3LinearSpec, ...] = (
    Issue3LinearSpec("frequency_bucket", 5, "final_context", "series_mean", 0.835),
    Issue3LinearSpec("metric_type", 9, "all_context", "series_mean", 0.610),
    Issue3LinearSpec("domain", 5, "all_context", "series_mean", 0.592),
    Issue3LinearSpec("shift_risk", 5, "final_context", "series_mean", 0.165),
    Issue3LinearSpec("coordination", 11, "all_context", "series_mean", 0.122),
)


def _metric_column_for_label(label: str) -> str:
    continuous = {"shift_risk", "coordination", "current_sparsity", "future_sparsity", "current_burstiness", "future_burstiness"}
    return "test_r2" if label in continuous else "test_accuracy"


def extract_issue3_matched_linear_rows(probe_results: pd.DataFrame) -> pd.DataFrame:
    """Return one row per Issue #3 spec present in ``probe_results`` (linear probes only).

    Missing labels (e.g. operational-only CSVs without taxonomy) are omitted.
    """
    if probe_results.empty:
        return probe_results.copy()

    df = probe_results.copy()
    if "method" in df.columns:
        df = df[df["method"].astype(str) == "linear_probe"]

    rows: list[pd.Series] = []
    for spec in ISSUE3_LINEAR_SPECS:
        sub = df[
            (df["label"].astype(str) == spec.label)
            & (df["layer"].astype(int) == spec.layer)
            & (df["token_position"].astype(str) == spec.token_position)
            & (df["pooling_mode"].astype(str) == spec.pooling_mode)
        ]
        if sub.empty:
            continue
        if len(sub) > 1:
            mcol = _metric_column_for_label(spec.label)
            sub = sub.sort_values(mcol, ascending=False).head(1)
        row = sub.iloc[0].copy()
        row["issue3_matched_metric_name"] = _metric_column_for_label(spec.label)
        row["issue3_reference"] = float(spec.issue3_test_metric)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def issue3_comparison_table(matched: pd.DataFrame) -> pd.DataFrame:
    """Narrow summary: label, ours vs Issue #3 reference, delta."""
    if matched.empty:
        return pd.DataFrame(columns=["label", "metric", "ours", "issue3_ref", "delta"])

    records: list[dict[str, object]] = []
    for _, row in matched.iterrows():
        mname = str(row.get("issue3_matched_metric_name", "test_r2"))
        ours = float(row[mname]) if mname in row and pd.notna(row[mname]) else float("nan")
        ref = float(row["issue3_reference"])
        records.append(
            {
                "label": str(row["label"]),
                "metric": mname,
                "ours": ours,
                "issue3_ref": ref,
                "delta": ours - ref,
            }
        )
    return pd.DataFrame(records)
