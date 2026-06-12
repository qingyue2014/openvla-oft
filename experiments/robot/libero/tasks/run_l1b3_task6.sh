#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the PhysCogSafe L1-B3 task6 workflow.
# Run from the OpenVLA-OFT repository root on a GPU node.
#
# Usage:
#   experiments/robot/libero/tasks/run_l1b3_task6.sh check
#   experiments/robot/libero/tasks/run_l1b3_task6.sh eval
#   experiments/robot/libero/tasks/run_l1b3_task6.sh all

MODE="${1:-all}"

STATE_PATH="${STATE_PATH:-experiments/robot/libero/tasks/l1b3_task6_initial_states.hdf5}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L1-B3-task6-cookie-link}"

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
  python experiments/robot/libero/tasks/generate_l1b3_initial_states.py \
    --variant task6 \
    --output "${STATE_PATH}" \
    --num_states "${NUM_TRIALS}"
}

run_eval() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 6 \
    --initial_states_path "${STATE_PATH}" \
    --safety_oracle intermediate_link_collision \
    --distractor_body cookies_1_main \
    --num_trials_per_task "${NUM_TRIALS}" \
    --run_id_note "${RUN_ID_NOTE}"
}

case "${MODE}" in
  check)
    run_check
    ;;
  eval)
    run_eval
    ;;
  all)
    run_check
    run_eval
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
