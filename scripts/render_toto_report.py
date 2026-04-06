from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown summary report from Toto interpretability outputs.")
    parser.add_argument("--probe-results-path", type=Path, required=True)
    parser.add_argument("--intervention-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--transfer-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-json-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(
        probe_results_path=args.probe_results_path,
        intervention_dirs=args.intervention_dirs,
        transfer_dir=args.transfer_dir,
        output_markdown_path=args.output_path,
        output_summary_path=args.summary_json_path,
    )


if __name__ == "__main__":
    main()
