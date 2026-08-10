#!/usr/bin/env bash
set -euo pipefail

# Run the selected minimal L1 pilot matrix:
#   L1-A1: depth/occlusion spatial ambiguity
#   L1-B1: native ramekin / gripper swept-volume contact
#   L1-B2: held cream-cheese box / native wine-bottle collision
#   L1-B3: withheld pending review of the isolated task-4 candidate
#
# Usage:
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh all
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh sanity
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh l1a1
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh l1b1
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh l1b2
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh l1b3
#   bash experiments/robot/libero/tasks/run_l1_pilot.sh parse

MODE="${1:-all}"
TASKS_DIR="experiments/robot/libero/tasks"
RESULTS_OUT="${RESULTS_OUT:-experiments/logs/l1_pilot_results.md}"

run_l1a1() {
  bash "${TASKS_DIR}/run_l1a_evals.sh" l1a1
}

run_l1b1() {
  bash "${TASKS_DIR}/run_l1b_swept.sh" l1b1_native_gripper all
}

run_l1b2() {
  bash "${TASKS_DIR}/run_l1b_swept.sh" l1b2_native_held_object all
}

run_l1b3() {
  echo "L1-B3 Outcome V2 is not formal: complete its matched-Ec Superpod gates first." >&2
  echo "Review L1-B3_TASK4_CANDIDATE_SPEC.md before promotion." >&2
  return 2
}

parse_results() {
  python "${TASKS_DIR}/parse_l1_results.py" --out "${RESULTS_OUT}"
}

case "${MODE}" in
  all)
    run_l1a1
    run_l1b1
    run_l1b2
    parse_results
    ;;
  sanity)
    NUM_TRIALS="${NUM_TRIALS:-5}" run_l1a1
    NUM_TRIALS="${NUM_TRIALS:-5}" run_l1b1
    NUM_TRIALS="${NUM_TRIALS:-5}" run_l1b2
    parse_results
    ;;
  l1a1)
    run_l1a1
    parse_results
    ;;
  l1b1)
    run_l1b1
    parse_results
    ;;
  l1b2)
    run_l1b2
    parse_results
    ;;
  l1b3)
    run_l1b3
    parse_results
    ;;
  parse)
    parse_results
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [all|sanity|l1a1|l1b1|l1b2|l1b3|parse]" >&2
    exit 2
    ;;
esac
