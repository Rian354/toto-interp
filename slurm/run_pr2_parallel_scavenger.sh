#!/bin/bash
# Parallel “shadow” PR2 runs on **scavenger** (many GPU nodes) while other jobs use the main RUNS_ROOT.
# Writes under RUNS_ROOT **per seed** only — safe to run **five seeds at once** if each job has a distinct SEED.
#
# Defaults separate from ic-express fleet:
#   RUNS_ROOT -> ${REPO}/runs/pr2_parallel_issue3  (override via sbatch --export)
#
# Optimizations: PYTHONUNBUFFERED + python -u so logs show progress; avoids silent-looking hangs.
#
# Submit fleet:
#   ./slurm/submit_pr2_parallel_fleet.sh
#
#SBATCH --job-name=toto-pr2-par
#SBATCH --partition=scavenger
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/pr2_par_%j.out
#SBATCH --error=logs/pr2_par_%j.err

set -euo pipefail

REPO="${REPO:-/scratch/rianatri/toto-interp}"
VENV="${VENV:-/scratch/rianatri/tmp/toto_interp_login_venv2}"
# Separate tree so we never race the ic-express jobs still writing pr2_operational_issue3/.
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/pr2_parallel_issue3}"
SIZE="${SIZE:-issue3}"
CONTEXT="${CONTEXT:-1024}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SEED="${SLURM_ARRAY_TASK_ID}"
elif [[ -n "${SEED:-}" ]]; then
  :
else
  echo "ERROR: Pass seed: sbatch --export=ALL,SEED=43 ${REPO}/slurm/run_pr2_parallel_scavenger.sh" >&2
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
  echo "ERROR: No Hugging Face token" >&2
  exit 1
fi

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${REPO}/logs"
mkdir -p "${RUNS_ROOT}"
echo "PR2 parallel scavenger jobid=${SLURM_JOB_ID} seed=${SEED} size=${SIZE} context=${CONTEXT} runs_root=${RUNS_ROOT} node=$(hostname) at $(date)"

NEURIPS_ARGS=(
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
python -u "${REPO}/scripts/run_neurips_prelim_experiments.py" "${NEURIPS_ARGS[@]}"

echo "Done seed ${SEED} at $(date)"
