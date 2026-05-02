#!/bin/bash
# Random-init linear + FNO only, **Issue #3–matched** data scale and labels:
#   500 series/split, 4 windows, 16 samples, context 1024, label-group all (taxonomy+dynamic).
# VRAM / train time / FNO learning curves: aggregate with
#   python scripts/aggregate_random_weight_fno.py --runs-root "$RUNS_ROOT" --out-dir ... --seeds "$SEED"
#
#SBATCH --job-name=toto-i3-rf
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/issue3_random_fno_%j.out
#SBATCH --error=logs/issue3_random_fno_%j.err
#
# Usage: sbatch --export=ALL,SEED=42 slurm/run_issue3_random_fno_one.sh
# Optional: RUNS_ROOT, REPO, VENV (same conventions as run_pr2_operational_suite.sh)

set -euo pipefail

REPO="${REPO:-/scratch/rianatri/toto-interp}"
VENV="${VENV:-/scratch/rianatri/tmp/toto_interp_login_venv2}"
SEED="${SEED:?Set SEED (e.g. sbatch --export=ALL,SEED=42)}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/issue3_random_fno}"

export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"
mkdir -p "${HF_HOME}"

if [[ -f "${REPO}/.hf_token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${REPO}/.hf_token")"
fi
if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${HF_HOME}/token")"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HOME:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${HOME}/.cache/huggingface/token")"
  export HF_HOME="${HOME}/.cache/huggingface"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: No Hugging Face token (HF_TOKEN, ${REPO}/.hf_token, HF_HOME/token, or ~/.cache/huggingface/token)" >&2
  exit 1
fi

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${REPO}/logs"
echo "Issue#3 random+FNO seed=${SEED} runs_root=${RUNS_ROOT} on $(hostname) at $(date)"

CMD=(
  python "${REPO}/scripts/run_random_weight_fno_suite.py"
  --output-root "${RUNS_ROOT}"
  --size issue3
  --seed "${SEED}"
  --device cuda
)
if [[ "${REUSE_EXISTING:-0}" == "1" ]]; then
  CMD+=(--reuse-existing)
fi
if [[ "${SKIP_TRANSFER:-0}" == "1" ]]; then
  CMD+=(--skip-transfer)
fi

"${CMD[@]}"

echo "Done seed ${SEED} at $(date)"
echo "Aggregate costs: python scripts/aggregate_random_weight_fno.py --runs-root ${RUNS_ROOT} --out-dir ${RUNS_ROOT}/aggregate --seeds ${SEED}"
