#!/usr/bin/env bash
set -euo pipefail

# Isolated L1-B3 Task-4 candidate entrypoint.
#
# The implementation reuses the common L1-B validators but exposes only the
# task-4 candidate family. The retained Task-8 alternative has different state,
# rollout, report, and run-ID namespaces.
#
# Usage:
#   bash experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh smoke
#   bash experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh prepare
#   bash experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh candidate_full

MODE="${1:-smoke}"
TASKS_DIR="experiments/robot/libero/tasks"
FAMILY="l1b3_task4_candidate"
export L1B_EXTRA_FAMILY_MODULE="experiments.robot.libero.tasks.l1b3_task4_gate_config"
export L1B3_TASK4_OBSTACLE_BODY="l1_b_goal_arm_gate_1_main"
export L1B3_TASK4_BDDL="${TASKS_DIR}/l1b4_goal_arm_sweep.bddl"

case "${MODE}" in
  generate|check|safe_reference|eb|er|ec|smoke|prepare)
    COMMON_MODE="${MODE}"
    ;;
  candidate_full)
    COMMON_MODE="all"
    ;;
  all|eval|formal)
    echo "Task-4 is an isolated L1-B3 candidate; '${MODE}' is intentionally disabled." >&2
    echo "Use 'candidate_full', then review every release gate before promotion." >&2
    exit 2
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected generate|check|safe_reference|eb|er|ec|smoke|prepare|candidate_full" >&2
    exit 2
    ;;
esac

exec bash "${TASKS_DIR}/run_l1b_swept.sh" "${FAMILY}" "${COMMON_MODE}"
