from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp import TraceConfig, extract_activations, load_toto_with_fallback
from toto_interp.boom import (
    build_boom_windows,
    ensure_boom_snapshot,
    load_boom_taxonomy,
    split_boom_series_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Toto activation traces for BOOM research windows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-windows-per-series", type=int, default=16)
    parser.add_argument("--max-series-per-split", type=int, default=0)
    parser.add_argument("--include-heldout-late", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_toto_with_fallback(args.model_id, map_location="cpu")
    backbone = model.model
    patch_size = int(backbone.patch_embed.patch_size)

    if args.context_length % patch_size != 0:
        raise ValueError(f"context-length must be divisible by patch size ({patch_size}).")

    taxonomy = load_boom_taxonomy()
    split_ids = split_boom_series_ids(taxonomy, seed=args.seed)
    if args.max_series_per_split > 0:
        split_ids = {
            split_name: series_ids[: args.max_series_per_split]
            for split_name, series_ids in split_ids.items()
        }

    snapshot_path = ensure_boom_snapshot(
        [series_id for series_ids in split_ids.values() for series_id in series_ids]
    )

    trace_config = TraceConfig(use_kv_cache=not args.disable_kv_cache)
    summary: dict[str, object] = {
        "model_id": args.model_id,
        "patch_size": patch_size,
        "context_length": args.context_length,
        "trace_config": {
            "layers": trace_config.layers,
            "token_positions": trace_config.token_positions,
            "pooling_modes": trace_config.pooling_modes,
            "use_kv_cache": trace_config.use_kv_cache,
        },
        "splits": {},
    }

    for split_name, series_ids in split_ids.items():
        windows = build_boom_windows(
            series_ids=series_ids,
            split=split_name,
            snapshot_path=snapshot_path,
            taxonomy=taxonomy,
            context_length=args.context_length,
            patch_size=patch_size,
            max_windows_per_series=args.max_windows_per_series,
            include_heldout_late=args.include_heldout_late,
        )
        batch = extract_activations(model, windows, trace_config)
        batch_path = args.output_dir / f"{split_name}_activations.pt"
        batch.save(batch_path)
        summary["splits"][split_name] = {
            "series_count": len(series_ids),
            "window_count": len(windows),
            "activation_count": len(batch),
            "path": str(batch_path),
        }

    with open(args.output_dir / "activation_dump_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
