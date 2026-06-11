#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the PhysCogSafe L1-B2 task6 workflow.
# Run from the OpenVLA-OFT repository root on a GPU node.
#
# Usage:
#   experiments/robot/libero/tasks/run_l1b2_task6.sh check
#   experiments/robot/libero/tasks/run_l1b2_task6.sh debug
#   experiments/robot/libero/tasks/run_l1b2_task6.sh eval
#   experiments/robot/libero/tasks/run_l1b2_task6.sh all

MODE="${1:-all}"

STATE_PATH="${STATE_PATH:-experiments/robot/libero/tasks/l1b2_task6_initial_states.hdf5}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L1-B2-task6-cookie-ramekin}"

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

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

run_check() {
  experiments/robot/libero/tasks/check_l1b2_task6_layout.sh "${STATE_PATH}"
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
    echo "Expected one of: check, debug, eval, all" >&2
    exit 2
    ;;
esac
