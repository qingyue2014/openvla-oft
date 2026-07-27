#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the PhysCogSafe L1-B2 task6 workflow.
# Run from the OpenVLA-OFT repository root on a GPU node.
#
# Usage:
#   experiments/robot/libero/tasks/run_l1b2_task6.sh check
#   experiments/robot/libero/tasks/run_l1b2_task6.sh check_safe
#   experiments/robot/libero/tasks/run_l1b2_task6.sh debug
#   experiments/robot/libero/tasks/run_l1b2_task6.sh eval
#   experiments/robot/libero/tasks/run_l1b2_task6.sh eval_safe
#   experiments/robot/libero/tasks/run_l1b2_task6.sh control
#   experiments/robot/libero/tasks/run_l1b2_task6.sh all

MODE="${1:-all}"

STATE_PATH="${STATE_PATH:-experiments/robot/libero/tasks/l1b2_task6_initial_states.hdf5}"
SAFE_STATE_PATH="${SAFE_STATE_PATH:-experiments/robot/libero/tasks/l1b2_task6_matched_safe_initial_states.hdf5}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
# EVAL_SEED only affects the eval process (seed repeats); initial states stay fixed.
EVAL_SEED="${EVAL_SEED:-7}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-violation}"
MAX_VIOLATION_VIDEOS="${MAX_VIOLATION_VIDEOS:-10}"
MAX_SUCCESS_VIDEOS="${MAX_SUCCESS_VIDEOS:-10}"
MAX_FAILURE_VIDEOS="${MAX_FAILURE_VIDEOS:-10}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L1-B2-task6-cookie-ramekin}"
SAFE_RUN_ID_NOTE="${SAFE_RUN_ID_NOTE:-L1-B2-task6-matched-safe}"
if [[ -n "${RUN_ID_SUFFIX}" ]]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-${RUN_ID_SUFFIX}"
  SAFE_RUN_ID_NOTE="${SAFE_RUN_ID_NOTE}-${RUN_ID_SUFFIX}"
fi

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

run_check() {
  experiments/robot/libero/tasks/check_l1b2_task6_layout.sh "${STATE_PATH}"
}

run_check_safe() {
  rm -f "${SAFE_STATE_PATH}"
  python experiments/robot/libero/tasks/generate_l1b2_initial_states.py \
    --variant task6_matched_safe \
    --output "${SAFE_STATE_PATH}" \
    --num_states "${NUM_TRIALS}"
}

run_debug() {
  python experiments/robot/libero/tasks/debug_l1b2_task6_init.py \
    --state_path "${STATE_PATH}"
}

run_eval() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 6 \
    --initial_states_path "${STATE_PATH}" \
    --safety_oracle held_object_corridor \
    --held_object_body akita_black_bowl_1_main \
    --corridor_body "cookies_1_main,glazed_rim_porcelain_ramekin_1_main" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --seed "${EVAL_SEED}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --max_violation_videos "${MAX_VIOLATION_VIDEOS}" \
    --max_success_videos "${MAX_SUCCESS_VIDEOS}" \
    --max_failure_videos "${MAX_FAILURE_VIDEOS}" \
    --run_id_note "${RUN_ID_NOTE}"
}

run_eval_safe() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 6 \
    --initial_states_path "${SAFE_STATE_PATH}" \
    --safety_oracle held_object_corridor \
    --held_object_body akita_black_bowl_1_main \
    --corridor_body "cookies_1_main,glazed_rim_porcelain_ramekin_1_main" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --seed "${EVAL_SEED}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --max_violation_videos "${MAX_VIOLATION_VIDEOS}" \
    --max_success_videos "${MAX_SUCCESS_VIDEOS}" \
    --max_failure_videos "${MAX_FAILURE_VIDEOS}" \
    --run_id_note "${SAFE_RUN_ID_NOTE}"
}

case "${MODE}" in
  check)
    run_check
    ;;
  check_safe)
    run_check_safe
    ;;
  debug)
    run_debug
    ;;
  eval)
    run_eval
    ;;
  eval_safe)
    run_eval_safe
    ;;
  control)
    run_check_safe
    run_eval_safe
    ;;
  all)
    run_check
    run_debug
    run_eval
    run_check_safe
    run_eval_safe
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: check, check_safe, debug, eval, eval_safe, control, all" >&2
    exit 2
    ;;
esac
