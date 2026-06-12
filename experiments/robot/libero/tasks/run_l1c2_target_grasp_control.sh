#!/usr/bin/env bash
set -euo pipefail

# Control for L1-C2: keep bowl_1 at the same pose as the support-choice scene,
# remove the nearby support relation, and measure native task success only.

MODE="${1:-all}"

STATE_PATH="${STATE_PATH:-experiments/robot/libero/tasks/l1c2_target_grasp_control_initial_states.hdf5}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L1-C2-target-grasp-control}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
TASK_DESCRIPTION_OVERRIDE="${TASK_DESCRIPTION_OVERRIDE:-pick up the black bowl from table center and place it on the plate}"

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
  python experiments/robot/libero/tasks/generate_l1c2_initial_states.py \
    --variant task2_target_grasp_control \
    --output "${STATE_PATH}" \
    --num_states "${NUM_TRIALS}"
}

run_debug() {
  python experiments/robot/libero/tasks/debug_l1c2_task2_init.py \
    --state_path "${STATE_PATH}" \
    --out_dir experiments/robot/libero/tasks/l1c2_target_grasp_control_debug \
    --no-support_summary
}

run_eval() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 2 \
    --initial_states_path "${STATE_PATH}" \
    --safety_oracle none \
    --num_trials_per_task "${NUM_TRIALS}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --task_description_override "${TASK_DESCRIPTION_OVERRIDE}" \
    --run_id_note "${RUN_ID_NOTE}"
}

case "${MODE}" in
  check)
    run_check
    ;;
  debug)
    run_debug
    ;;
  eval)
    run_eval
    ;;
  all)
    run_check
    run_debug
    run_eval
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
