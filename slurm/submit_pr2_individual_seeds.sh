#!/usr/bin/env bash
# Submit one Slurm job per seed (separate job IDs; can run in parallel up to partition limits).
# Uses slurm/run_pr2_operational_one.sh (same work as the array driver, without a job array).
#
# Examples:
#   ./slurm/submit_pr2_individual_seeds.sh 43 44 45 46
#   REPO=/path/to/toto-interp ./slurm/submit_pr2_individual_seeds.sh
#
# Default seeds: 43 44 45 46 (override with arguments).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if (($# > 0)); then
  SEEDS=("$@")
else
  SEEDS=(43 44 45 46)
fi

for s in "${SEEDS[@]}"; do
  echo "Submitting PR2 one-seed job SEED=${s}"
  sbatch --export=ALL,SEED="${s}" "${REPO}/slurm/run_pr2_operational_one.sh"
done
