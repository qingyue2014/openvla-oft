#!/usr/bin/env bash
set -euo pipefail

# Run the frozen L2-B2 pilot conditions with consistent evaluation settings.
# This script does not select successful layouts or modify any BDDL file.
#
# Conditions:
#   e0       native libero_10 task 1, no added stove
#   core     Eb stove-off, Er active near basket, Ec active far away
#   path     matched path-layout stove-off/on pair
#   all      e0 + core + path
#   summary  print key metrics from existing L2-B2 logs
#
# Usage:
#   NUM_TRIALS=5 RENDER_GPU=1 SAVE_VIDEO_MODE=all \
#     bash experiments/robot/libero/tasks/run_l2b2_full_pilot.sh all

MODE="${1:-all}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
NUM_TRIALS="${NUM_TRIALS:-5}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:-1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"

run_e0() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_10 \
    --task_ids 1 \
    --safety_oracle none \
    --num_trials_per_task "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --render_gpu_device_id "${RENDER_GPU}" \
    --run_id_note L2-B2-E0-native-cream-cheese-butter
}

run_custom() {
  local variant="$1"
  NUM_TRIALS="${NUM_TRIALS}" \
  SEED="${SEED}" \
  RENDER_GPU="${RENDER_GPU}" \
  SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE}" \
  CHECKPOINT="${CHECKPOINT}" \
    bash experiments/robot/libero/tasks/run_l2b2_basket_stove.sh "${variant}" all
}

run_core() {
  run_custom basket_off
  run_custom basket
  run_custom basket_far
}

run_path() {
  run_custom basket_path_off
  run_custom basket_path
}

print_summary() {
  local found=0
  while IFS= read -r log; do
    found=1
    echo
    echo "### ${log}"
    grep -E \
      "Overall success rate|Overall SVR|Overall valid-execution violation rate|Overall model collapse rate|Overall safe success rate" \
      "${log}" || true
  done < <(find experiments/logs -maxdepth 1 -type f -name 'EVAL-*--L2-B2-*.txt' | sort)
  if [[ "${found}" == "0" ]]; then
    echo "No L2-B2 eval logs found under experiments/logs/." >&2
    return 1
  fi
}

case "${MODE}" in
  e0)
    run_e0
    ;;
  core)
    run_core
    ;;
  path)
    run_path
    ;;
  all)
    run_e0
    run_core
    run_path
    ;;
  summary)
    print_summary
    ;;
  *)
    echo "Usage: $0 [e0|core|path|all|summary]" >&2
    exit 2
    ;;
esac
