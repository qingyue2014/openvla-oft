#!/usr/bin/env bash
set -euo pipefail

# Generate and evaluate episode-matched L2-B2 Eb/Er/Ec states.
#
# Usage:
#   NUM_TRIALS=20 RENDER_GPU=1 \
#     bash experiments/robot/libero/tasks/run_l2b2_paired_pilot.sh all
#   bash experiments/robot/libero/tasks/run_l2b2_paired_pilot.sh report

shopt -s nullglob

MODE="${1:-all}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:-1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-violation}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
PAIR_TOLERANCE="${PAIR_TOLERANCE:-0.005}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
[[ "${RENDER_GPU}" != "-1" ]] && export EGL_DEVICE_ID="${RENDER_GPU}"

EB_STATE="experiments/robot/libero/tasks/l2b2_paired_eb_stove_off.hdf5"
ER_STATE="experiments/robot/libero/tasks/l2b2_paired_er_stove_on.hdf5"
EC_STATE="experiments/robot/libero/tasks/l2b2_paired_ec_far_stove.hdf5"

generate() {
  python experiments/robot/libero/tasks/generate_l2b2_paired_initial_states.py \
    --num_states "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --pair_tolerance "${PAIR_TOLERANCE}" \
    --eb_output "${EB_STATE}" \
    --er_output "${ER_STATE}" \
    --ec_output "${EC_STATE}"
}

run_condition() {
  local variant="$1"
  local state_path="$2"
  local note="$3"
  NUM_TRIALS="${NUM_TRIALS}" \
  SEED="${SEED}" \
  RENDER_GPU="${RENDER_GPU}" \
  SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE}" \
  CHECKPOINT="${CHECKPOINT}" \
  STATE_PATH="${state_path}" \
  RUN_ID_NOTE="${note}" \
    bash experiments/robot/libero/tasks/run_l2b2_basket_stove.sh "${variant}" eval
}

evaluate() {
  run_condition basket_off "${EB_STATE}" L2-B2-paired-Eb-stove-off
  run_condition basket "${ER_STATE}" L2-B2-paired-Er-stove-on
  run_condition basket_far "${EC_STATE}" L2-B2-paired-Ec-far-stove
}

latest_log() {
  local note="$1"
  local candidates=(experiments/logs/EVAL-*--"${note}".txt)
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    return 0
  fi
  ls -t "${candidates[@]}" | head -1
}

report() {
  for note in \
    L2-B2-paired-Eb-stove-off \
    L2-B2-paired-Er-stove-on \
    L2-B2-paired-Ec-far-stove
  do
    local log
    log="$(latest_log "${note}")"
    echo
    echo "### ${note}"
    if [[ -z "${log}" ]]; then
      echo "No log found"
      continue
    fi
    echo "${log}"
    grep -E \
      "SemanticHazardProximityOracle metrics|Safety violation|Success:|Overall success rate|Overall SVR|Overall valid-execution violation rate|Overall model collapse rate|Overall safe success rate" \
      "${log}" || true
  done
}

case "${MODE}" in
  generate)
    generate
    ;;
  eval)
    evaluate
    ;;
  report)
    report
    ;;
  all)
    generate
    evaluate
    report
    ;;
  *)
    echo "Usage: $0 [generate|eval|report|all]" >&2
    exit 2
    ;;
esac
