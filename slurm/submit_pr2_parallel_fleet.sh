#!/usr/bin/env bash
# Submit five PR2 jobs (seeds 42–46) on **scavenger** into RUNS_ROOT **pr2_parallel_issue3**
# by default — disjoint from `runs/pr2_operational_issue3` so ic-express jobs keep writing safely.
#
# Does **not** cancel other jobs.
#
# Usage:
#   ./slurm/submit_pr2_parallel_fleet.sh
#   RUNS_ROOT=/path ./slurm/submit_pr2_parallel_fleet.sh 47 48 49 50 51

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/pr2_parallel_issue3}"

if (($# > 0)); then
  SEEDS=("$@")
else
  SEEDS=(42 43 44 45 46)
fi

echo "RUNS_ROOT=${RUNS_ROOT}"
for s in "${SEEDS[@]}"; do
  echo "Submitting parallel scavenger PR2 SEED=${s}"
  sbatch --export=ALL,SEED="${s}",RUNS_ROOT="${RUNS_ROOT}" "${REPO}/slurm/run_pr2_parallel_scavenger.sh"
done
