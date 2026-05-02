#!/bin/bash
# Same as run_pr2_operational_suite.sh but one seed per job (no array).
# Submit:  sbatch --export=ALL,SEED=43 slurm/run_pr2_operational_one.sh
# Logs:    logs/pr2_one_<jobid>.out (use sacct -j <id> to map seed from command line)
#
#SBATCH --job-name=toto-pr2-1
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/pr2_one_%j.out
#SBATCH --error=logs/pr2_one_%j.err
#
# HF auth: ${HF_HOME}/token or ${REPO}/.hf_token (see run_pr2_operational_suite.sh)

set -euo pipefail

REPO="${REPO:-/scratch/rianatri/toto-interp}"
VENV="${VENV:-/scratch/rianatri/tmp/toto_interp_login_venv2}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/pr2_operational_issue3}"
SIZE="${SIZE:-issue3}"
CONTEXT="${CONTEXT:-1024}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SEED="${SLURM_ARRAY_TASK_ID}"
elif [[ -n "${SEED:-}" ]]; then
  :
else
  echo "ERROR: Pass seed:  sbatch --export=ALL,SEED=43 ${REPO}/slurm/run_pr2_operational_one.sh" >&2
  exit 1
fi

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
echo "PR2 one-seed jobid=${SLURM_JOB_ID} seed=${SEED} size=${SIZE} context=${CONTEXT} runs_root=${RUNS_ROOT} on $(hostname) at $(date)"

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
