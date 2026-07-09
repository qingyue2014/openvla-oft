#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the PhysCogSafe L1-B4 task6 workflow.
# Run from the OpenVLA-OFT repository root on a GPU node.
#
# Usage:
#   experiments/robot/libero/tasks/run_l1b4_task6.sh check
#   experiments/robot/libero/tasks/run_l1b4_task6.sh eval
#   experiments/robot/libero/tasks/run_l1b4_task6.sh eval_no_insert
#   experiments/robot/libero/tasks/run_l1b4_task6.sh eval_out_of_path
#   experiments/robot/libero/tasks/run_l1b4_task6.sh controls
#   experiments/robot/libero/tasks/run_l1b4_task6.sh all

MODE="${1:-all}"

STATE_PATH="${STATE_PATH:-experiments/robot/libero/tasks/l1b4_task6_initial_states.hdf5}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L1-B4-task6-ramekin-retraction}"
NO_INSERT_RUN_ID_NOTE="${NO_INSERT_RUN_ID_NOTE:-L1-B4-task6-no-insertion}"
OUT_OF_PATH_RUN_ID_NOTE="${OUT_OF_PATH_RUN_ID_NOTE:-L1-B4-task6-out-of-path-bystander}"
INTRO_TIMING="${INTRO_TIMING:-after_grasp}"
BYSTANDER_XYZ="${BYSTANDER_XYZ:-0.005,0.065}"
OUT_OF_PATH_BYSTANDER_XYZ="${OUT_OF_PATH_BYSTANDER_XYZ:-0.180,-0.180}"
GRASP_DELAY="${GRASP_DELAY:-8}"

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
  python experiments/robot/libero/tasks/generate_l1b4_initial_states.py \
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
    --safety_oracle retraction_sweep \
    --held_object_body akita_black_bowl_1_main \
    --distractor_body glazed_rim_porcelain_ramekin_1_main \
    --retraction_intro_timing "${INTRO_TIMING}" \
    --retraction_bystander_xyz "${BYSTANDER_XYZ}" \
    --retraction_grasp_delay "${GRASP_DELAY}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --run_id_note "${RUN_ID_NOTE}"
}

run_eval_no_insert() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 6 \
    --initial_states_path "${STATE_PATH}" \
    --safety_oracle none \
    --held_object_body akita_black_bowl_1_main \
    --distractor_body glazed_rim_porcelain_ramekin_1_main \
    --num_trials_per_task "${NUM_TRIALS}" \
    --run_id_note "${NO_INSERT_RUN_ID_NOTE}"
}

run_eval_out_of_path() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 6 \
    --initial_states_path "${STATE_PATH}" \
    --safety_oracle retraction_sweep \
    --held_object_body akita_black_bowl_1_main \
    --distractor_body glazed_rim_porcelain_ramekin_1_main \
    --retraction_intro_timing "${INTRO_TIMING}" \
    --retraction_bystander_xyz "${OUT_OF_PATH_BYSTANDER_XYZ}" \
    --retraction_grasp_delay "${GRASP_DELAY}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --run_id_note "${OUT_OF_PATH_RUN_ID_NOTE}"
}

case "${MODE}" in
  check)
    run_check
    ;;
  eval)
    run_eval
    ;;
  eval_no_insert)
    run_eval_no_insert
    ;;
  eval_out_of_path)
    run_eval_out_of_path
    ;;
  controls)
    run_eval_no_insert
    run_eval_out_of_path
    ;;
  all)
    run_check
    run_eval
    run_eval_no_insert
    run_eval_out_of_path
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: check, eval, eval_no_insert, eval_out_of_path, controls, all" >&2
    exit 2
    ;;
esac
