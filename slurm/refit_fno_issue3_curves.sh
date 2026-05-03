#!/bin/bash
# Refit FNO probes only (uses cached activations) so artifact_metadata contains history + timing.
# Run from repo root after deploying toto_interp/fno.py with learning-curve logging.
#
#   sbatch slurm/refit_fno_issue3_curves.sh
#
#SBATCH --job-name=fno-refit-issue3
#SBATCH --partition=scavenger
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=logs/fno_refit_issue3_%j.out
#SBATCH --error=logs/fno_refit_issue3_%j.err

set -euo pipefail

REPO="${REPO:-/scratch/rianatri/toto-interp}"
VENV="${VENV:-/scratch/rianatri/tmp/toto_interp_login_venv2}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/pr2_operational_issue3}"

cd "${REPO}"
mkdir -p logs
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

for SEED in 42 43; do
  PRE="${RUNS_ROOT}/operational_pretrained_seed${SEED}"
  OUT="${RUNS_ROOT}/operational_fno_seed${SEED}/probes"
  if [[ ! -f "${PRE}/activations/train_activations.pt" ]]; then
    echo "SKIP seed ${SEED}: missing activations under ${PRE}/activations"
    continue
  fi
  TS="$(date +%Y%m%d_%H%M%S)"
  if [[ -d "${OUT}" ]]; then
    mv "${OUT}" "${RUNS_ROOT}/operational_fno_seed${SEED}/probes_backup_${TS}"
  fi
  mkdir -p "${OUT}"
  echo "=== Refit FNO seed=${SEED} -> ${OUT} ==="
  python scripts/fit_toto_probes.py \
    --activation-files \
      "${PRE}/activations/train_activations.pt" \
      "${PRE}/activations/val_activations.pt" \
      "${PRE}/activations/test_activations.pt" \
    --window-files \
      "${PRE}/activations/train_windows.pt" \
      "${PRE}/activations/val_windows.pt" \
      "${PRE}/activations/test_windows.pt" \
    --output-dir "${OUT}" \
    --method fno \
    --label-group all \
    --device cuda \
    --seed "${SEED}" \
    --fno-width 24 --fno-layers 3 --fno-modes 12 \
    --epochs 16 --batch-size 16 \
    --model-id Datadog/Toto-Open-Base-1.0 \
    --weight-source pretrained
done

python scripts/aggregate_pr2_operational.py \
  --runs-root "${RUNS_ROOT}" \
  --out-dir "${RUNS_ROOT}/issue3_aggregate" \
  --seeds 42 43

echo "Done refit + aggregate at $(date)"
