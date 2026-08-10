#!/bin/bash
#SBATCH --job-name=pc-l1c5-construct
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=normal
#SBATCH --account=trllmout
#SBATCH --time=03:00:00
#SBATCH --output=/project/trllmout/physcog-runs/openvla-oft-l1c5/.physcog-agent/jobs/l1c5-construction-%j.out
#SBATCH --error=/project/trllmout/physcog-runs/openvla-oft-l1c5/.physcog-agent/jobs/l1c5-construction-%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
module load slurm "nvhpc-hpcx-cuda12/23.11"

if [[ -z "${SLURM_JOB_ID:-}" ]] || [[ "$(hostname)" == slogin-* ]]; then
  echo "L1-C5 construction must run on a Superpod Slurm compute node." >&2
  exit 2
fi

REMOTE_REPO="${L1C5_REMOTE_REPO:-/project/trllmout/physcog-runs/openvla-oft-l1c5}"
cd "${REMOTE_REPO}"
export PATH="/home/drwqyhappy/.conda/envs/openvla_oft/bin:${PATH}"
export LIBERO_ROOT="/home/drwqyhappy/04-mycode/LIBERO"
export PYTHONPATH="${REMOTE_REPO}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RENDER_GPU_DEVICE_ID=0
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/project/trllmout/physcog-runs/numba-cache/${USER:-physcog}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/project/trllmout/physcog-runs/uv-cache/${USER:-physcog}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/trllmout/physcog-runs/xdg-cache/${USER:-physcog}}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/project/trllmout/physcog-runs/jax-cache/${USER:-physcog}}"
mkdir -p \
  "${NUMBA_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}"

export NUM_TRIALS="${L1C5_CONSTRUCTION_STATES:-50}"
export CALIBRATION_NUM_STATES="${L1C5_CALIBRATION_STATES:-8}"
export PREVIEW_NUM_STATES="${NUM_TRIALS}"
export PREVIEW_SAVE_IMAGE_STATES="${L1C5_PREVIEW_SAVE_IMAGE_STATES:-3}"
export POLICY_MODEL_FAMILY=pi05

printf '__L1C5_COMPUTE_NODE__=%s\n' "$(hostname)"
printf '__L1C5_COMMIT__=%s\n' "$(git rev-parse HEAD)"
printf '__L1C5_DESIGN_SHA256__=%s\n' "$(sha256sum experiments/robot/libero/tasks/l1c5_design_prereg.json | awk '{print $1}')"

bash experiments/robot/libero/tasks/run_l1c5_orange_juice_basket.sh construct

mkdir -p experiments/logs/l1c5_construction
sha256sum \
  experiments/robot/libero/tasks/l1c5_design_prereg.json \
  experiments/robot/libero/tasks/l1c5_eb_states.hdf5 \
  experiments/robot/libero/tasks/l1c5_er_states.hdf5 \
  experiments/robot/libero/tasks/l1c5_ec_states.hdf5 \
  experiments/robot/libero/tasks/l1c5_state_bundle.json \
  experiments/logs/l1c5_native_preflight.json \
  experiments/logs/l1c5_exact_state_preview.csv \
  experiments/logs/l1c5_exact_state_preview_timeline.csv \
  experiments/logs/l1c5_calibration.csv \
  experiments/logs/l1c5_safe_reference.csv \
  > experiments/logs/l1c5_construction/artifact_sha256.txt

printf '__L1C5_CONSTRUCTION_EXIT_CODE__=0\n'
