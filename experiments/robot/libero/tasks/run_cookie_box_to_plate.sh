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

BDDL_FILE="${BDDL_FILE:-experiments/robot/libero/tasks/PHYSCOG_L1B2_corridor_carry.bddl}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
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

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

run_list() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --bddl_file "${BDDL_FILE}" \
    --task_suite_name libero_spatial \
    --list_bodies_only True \
    --num_trials_per_task 1 \
    --run_id_note "${RUN_ID_NOTE}"
}

run_eval() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --bddl_file "${BDDL_FILE}" \
    --safety_oracle none \
    --num_trials_per_task "${NUM_TRIALS}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
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
