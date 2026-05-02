#!/bin/bash
# PR #2 operational suite (Issue #3 data scale): pretrained LP, random-init LP, FNO.
#
# ic-express: single H100 node, max 8h — light CPU/RAM + generic gpu:1 helps the scheduler pack/start jobs.
# If a seed exceeds 8h, resubmit the IllinoisComputes-GPU variant (24h) or split with --reuse-existing.
#
# Aggregate: python scripts/aggregate_pr2_operational.py --runs-root "$RUNS_ROOT" --out-dir ...
#
#SBATCH --job-name=toto-pr2-icx
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
# One seed at a time in this array (`%1`). For **separate job IDs per seed** (often parallel), use:
#   ./slurm/submit_pr2_individual_seeds.sh 42 43 44 45 46
# or:  sbatch --export=ALL,SEED=43 slurm/run_pr2_operational_one.sh
#SBATCH --array=42-46%1
#SBATCH --output=logs/pr2_icexpress_%A_%a.out
#SBATCH --error=logs/pr2_icexpress_%A_%a.err
#
# HF auth: ${HF_HOME}/token (from huggingface-cli login) or ${REPO}/.hf_token — see script body.
# Optional: sbatch --export=ALL,REUSE_EXISTING=1 to skip completed steps after a partial run.

set -euo pipefail

REPO="${REPO:-/scratch/rianatri/toto-interp}"
VENV="${VENV:-/scratch/rianatri/tmp/toto_interp_login_venv2}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/pr2_operational_issue3}"
SIZE="${SIZE:-issue3}"
CONTEXT="${CONTEXT:-1024}"

export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"
mkdir -p "${HF_HOME}"

if [[ -f "${REPO}/.hf_token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${REPO}/.hf_token")"
fi
# `huggingface-cli login` writes here when HF_HOME matches (no separate .hf_token needed).
if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${HF_HOME}/token")"
fi
# Batch jobs often have HF in $HOME but not under $REPO (interactive `huggingface-cli login`).
if [[ -z "${HF_TOKEN:-}" && -n "${HOME:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${HOME}/.cache/huggingface/token")"
  export HF_HOME="${HOME}/.cache/huggingface"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: No Hugging Face token. Set HF_TOKEN, or: huggingface-cli login, or ${REPO}/.hf_token, or ~/.cache/huggingface/token" >&2
  exit 1
fi

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${REPO}/logs"
echo "PR2 ic-express seed=${SEED} size=${SIZE} context=${CONTEXT} runs_root=${RUNS_ROOT} on $(hostname) at $(date)"

NEURIPS_ARGS=(
  "${REPO}/scripts/run_neurips_prelim_experiments.py"
  --output-root "${RUNS_ROOT}"
  --suite operational
  --size "${SIZE}"
  --seeds "${SEED}"
  --device cuda
  --context-length "${CONTEXT}"
)
if [[ "${REUSE_EXISTING:-0}" == "1" ]]; then
  NEURIPS_ARGS+=(--reuse-existing)
fi
python "${NEURIPS_ARGS[@]}"

echo "Done seed ${SEED} at $(date)"
