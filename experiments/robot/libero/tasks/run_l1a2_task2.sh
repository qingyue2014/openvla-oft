#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the PhysCogSafe L1-A2 occluded-bowl grasp workflow.
# Run from the OpenVLA-OFT repository root on a GPU node.
#
# Usage:
#   experiments/robot/libero/tasks/run_l1a2_task2.sh preview
#   experiments/robot/libero/tasks/run_l1a2_task2.sh check
#   experiments/robot/libero/tasks/run_l1a2_task2.sh eval
#   experiments/robot/libero/tasks/run_l1a2_task2.sh all
#   experiments/robot/libero/tasks/run_l1a2_task2.sh matched

MODE="${1:-all}"

STATE_PATH="${STATE_PATH:-experiments/robot/libero/tasks/l1a2_task2_initial_states.hdf5}"
SAFE_STATE_PATH="${SAFE_STATE_PATH:-experiments/robot/libero/tasks/l1a2_task2_matched_safe_initial_states.hdf5}"
PREVIEW_DIR="${PREVIEW_DIR:-experiments/robot/libero/tasks/l1a2_debug}"
SAFE_PREVIEW_DIR="${SAFE_PREVIEW_DIR:-experiments/robot/libero/tasks/l1a2_safe_debug}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L1-A2-task2-cookie-in-bowl-occlusion}"
TASK_DESCRIPTION_OVERRIDE="${TASK_DESCRIPTION_OVERRIDE:-}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-violation}"

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
  rm -f "${STATE_PATH}"
  python experiments/robot/libero/tasks/generate_l1a2_initial_states.py \
    --variant task2_cookie_in_bowl \
    --output "${STATE_PATH}" \
    --num_states "${NUM_TRIALS}" \
    --preview_dir "${PREVIEW_DIR}"
}

run_safe_check() {
  rm -f "${SAFE_STATE_PATH}"
  python experiments/robot/libero/tasks/generate_l1a2_initial_states.py \
    --variant task2_matched_safe_control \
    --output "${SAFE_STATE_PATH}" \
    --num_states "${NUM_TRIALS}" \
    --preview_dir "${SAFE_PREVIEW_DIR}"
}

run_preview() {
  python experiments/robot/libero/tasks/generate_l1a2_initial_states.py \
    --variant task2_cookie_in_bowl \
    --num_states 5 \
    --preview_dir "${PREVIEW_DIR}" \
    --preview_only
}

run_safe_preview() {
  python experiments/robot/libero/tasks/generate_l1a2_initial_states.py \
    --variant task2_matched_safe_control \
    --num_states 5 \
    --preview_dir "${SAFE_PREVIEW_DIR}" \
    --preview_only
}

run_eval_with_state() {
  local state_path="$1"
  local run_id_note="$2"
  local extra_args=()
  if [[ -n "${TASK_DESCRIPTION_OVERRIDE}" ]]; then
    extra_args+=(--task_description_override "${TASK_DESCRIPTION_OVERRIDE}")
  fi

  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 2 \
    --initial_states_path "${state_path}" \
    --safety_oracle none \
    --held_object_body akita_black_bowl_1_main \
    --distractor_body cookies_1_main \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --run_id_note "${run_id_note}" \
    "${extra_args[@]}"
}

run_eval() {
  run_eval_with_state "${STATE_PATH}" "${RUN_ID_NOTE}"
}

run_safe_eval() {
  run_eval_with_state "${SAFE_STATE_PATH}" "${RUN_ID_NOTE}-matched-safe"
}

case "${MODE}" in
  preview)
    run_preview
    ;;
  safe_preview)
    run_safe_preview
    ;;
  check)
    run_check
    ;;
  safe_check)
    run_safe_check
    ;;
  eval)
    run_eval
    ;;
  safe_eval)
    run_safe_eval
    ;;
  all)
    run_check
    run_eval
    ;;
  safe_all)
    run_safe_check
    run_safe_eval
    ;;
  matched)
    run_safe_check
    run_safe_eval
    run_check
    run_eval
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
