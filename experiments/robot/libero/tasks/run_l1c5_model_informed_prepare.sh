#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "L1-C5-MI scene preparation must run on Superpod, not the local Mac." >&2
  exit 2
fi

for path in \
  experiments/robot/libero/tasks/l1c5_mi_v1_eb_states.hdf5 \
  experiments/robot/libero/tasks/l1c5_mi_v1_er_states.hdf5 \
  experiments/robot/libero/tasks/l1c5_mi_v1_ec_states.hdf5 \
  experiments/robot/libero/tasks/l1c5_mi_v1_source_indices.json \
  experiments/robot/libero/tasks/l1c5_mi_v1_state_bundle.json \
  experiments/logs/l1c5_mi_v1_prepare \
  review/L1-C5-MI-v1_task/initialization \
  review/L1-C5-MI-v1_task/safe_reference; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite L1-C5-MI preparation output: ${path}" >&2
    exit 2
  fi
done

export LIBERO_ROOT="${LIBERO_ROOT:-/home/drwqyhappy/04-mycode/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/project/trllmout/physcog-runs/numba-cache/${USER:-physcog}}"
mkdir -p "${NUMBA_CACHE_DIR}"

python experiments/robot/libero/tasks/prepare_l1c5_model_informed_scene.py \
  --num_states "${L1C_MI_PREPARE_STATES:-50}" \
  --safe_states "${L1C_MI_SAFE_STATES:-8}"
