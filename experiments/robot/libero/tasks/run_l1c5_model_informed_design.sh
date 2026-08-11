#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "screen" ]]; then
  echo "Usage: $0 screen" >&2
  exit 2
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "L1-C5-MI physical screening must run on Superpod, not the local Mac." >&2
  exit 2
fi

TASKS_DIR="experiments/robot/libero/tasks"
LOG_DIR="experiments/logs/l1c5_mi_v1_design"
PREVIEW_DIR="review/L1-C5-MI-v1_task/candidate_screen"
mkdir -p "${LOG_DIR}"
if [[ -e "${PREVIEW_DIR}" ]]; then
  echo "Refusing to overwrite existing candidate-screen images: ${PREVIEW_DIR}" >&2
  exit 2
fi

export LIBERO_ROOT="${LIBERO_ROOT:-/home/drwqyhappy/04-mycode/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/project/trllmout/physcog-runs/numba-cache/${USER:-physcog}}"
mkdir -p "${NUMBA_CACHE_DIR}"

python "${TASKS_DIR}/screen_l1c5_model_informed_er.py" \
  --num_states "${L1C_MI_SCREEN_STATES:-8}" \
  --policy_model_family pi05 \
  --out_csv "${LOG_DIR}/candidate_physical_screen.csv" \
  --safe_csv "${LOG_DIR}/candidate_static_safe_target_screen.csv" \
  --out_json "${LOG_DIR}/candidate_screen.json" \
  --out_report "${LOG_DIR}/candidate_screen.md" \
  --preview_dir "${PREVIEW_DIR}"
