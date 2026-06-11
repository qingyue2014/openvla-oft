#!/usr/bin/env bash
set -euo pipefail

# Capability check for using the LIBERO cookie box as a paper/cardboard-box proxy.
# This intentionally uses no PhysCog safety oracle: it only tests whether the
# policy can follow "pick up the cookie box and place it on the plate".
#
# Run from the OpenVLA-OFT repository root on a GPU node.
#
# Usage:
#   experiments/robot/libero/tasks/run_cookie_box_to_plate.sh list
#   experiments/robot/libero/tasks/run_cookie_box_to_plate.sh eval

MODE="${1:-eval}"

# Default to the LIBERO-Long (libero_10) checkpoint: it is the only released
# OFT checkpoint whose training data includes grasping packaged boxes and
# placing objects on plates. TASK_SUITE_NAME must match the checkpoint because
# the eval script uses it as the action un-normalization key (norm_stats).
# Fallback BDDL swaps the cookie box for the cream cheese box, which libero_10
# policies have actually grasped during training:
#   BDDL_FILE=experiments/robot/libero/tasks/PHYSCOG_L1B2_corridor_carry_cream_cheese.bddl
BDDL_FILE="${BDDL_FILE:-experiments/robot/libero/tasks/PHYSCOG_L1B2_corridor_carry.bddl}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
# CUDA inference (GPU 0) and MuJoCo EGL rendering on the same GPU causes
# SIGABRT in read_pixels after the first inference call. Steer the renderer
# to GPU 1 to avoid the interference. Override with RENDER_GPU=0 if the
# machine has only one GPU (the crash will recur, but nothing can be done
# without separate render/inference processes).
RENDER_GPU="${RENDER_GPU:-1}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-10}"
RUN_ID_NOTE="${RUN_ID_NOTE:-cookie-box-to-plate-capability}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"

if [[ -z "${LIBERO_ROOT}" ]]; then
  if [[ -d "../LIBERO/libero" ]]; then
    LIBERO_ROOT="$(cd ../LIBERO && pwd)"
  elif [[ -d "../libero/libero" ]]; then
    LIBERO_ROOT="$(cd ../libero && pwd)"
  fi
fi

if [[ -n "${LIBERO_ROOT}" ]]; then
  export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
fi

# On single-GPU nodes CUDA inference and EGL rendering share the same device
# and interfere (SIGABRT in read_pixels after first inference call). Use osmesa
# (CPU-based offscreen rendering) to avoid any GPU/EGL dependency entirely.
# On multi-GPU nodes you can override back to EGL: MUJOCO_GL=egl RENDER_GPU=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

run_list() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --bddl_file "${BDDL_FILE}" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --list_bodies_only True \
    --num_trials_per_task 1 \
    --run_id_note "${RUN_ID_NOTE}"
}

run_eval() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --bddl_file "${BDDL_FILE}" \
    --safety_oracle none \
    --num_trials_per_task "${NUM_TRIALS}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --render_gpu_device_id "${RENDER_GPU}" \
    --run_id_note "${RUN_ID_NOTE}"
}

case "${MODE}" in
  list)
    run_list
    ;;
  eval)
    run_eval
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
