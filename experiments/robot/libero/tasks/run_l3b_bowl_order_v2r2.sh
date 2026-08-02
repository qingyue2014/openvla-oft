#!/usr/bin/env bash
set -euo pipefail

# Cross-platform repaired native-only 50-state L3-B evaluation.  v2r2 keeps
# the Er drawer on the preregistered collision-free qpos=0.001 plateau and
# evaluates pi0.5 only, per the current experiment policy.

MODE="${1:-prepare}"
TASKS_DIR="experiments/robot/libero/tasks"
STATIC_ROOT="review/L3-B_bowl_order_v2r2_task"

export NUM_STATES=50
export FORMAL_EXPECTED_COUNT=50
export DESIGN_PREREGISTRATION="${TASKS_DIR}/l3b_bowl_v2r2_design_prereg.json"
export EB_STATES="${TASKS_DIR}/l3b_bowl_v2r2_eb_states.hdf5"
export ER_STATES="${TASKS_DIR}/l3b_bowl_v2r2_er_states.hdf5"
export EC_STATES="${TASKS_DIR}/l3b_bowl_v2r2_ec_states.hdf5"
export INITIAL_GATE="${STATIC_ROOT}/L3-B_bowl_v2r2_initial_gate_manifest.json"
export PAIRING_GATE="${STATIC_ROOT}/L3-B_bowl_v2r2_pairing_gate.json"
export RUNTIME_REPLAY_GATE="${STATIC_ROOT}/L3-B_bowl_v2r2_runtime_replay_gate.json"
export EB_PREFLIGHT="${STATIC_ROOT}/L3-B_bowl_v2r2_Eb_native_preflight.json"
export ER_PREFLIGHT="${STATIC_ROOT}/L3-B_bowl_v2r2_Er_native_preflight.json"
export EC_PREFLIGHT="${STATIC_ROOT}/L3-B_bowl_v2r2_Ec_native_preflight.json"
export SAFE_WITNESS_EPISODES="0,1,2,3,4"
export MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-5}"

case "${MODE}" in
  prepare|check|safe-witness)
    export REVIEW_ROOT="${STATIC_ROOT}"
    exec bash "${TASKS_DIR}/run_l3b_bowl_order.sh" "${MODE}"
    ;;
  pi05_smoke|pi05_formal)
    export REVIEW_ROOT="review/L3-B_bowl_order_v2r2_pi05_task"
    export RUN_TAG="v2r2_pi05"
    export MODEL_LABEL=pi05
    export SERVER_LOG="experiments/logs/l3b_bowl_v2r2_pi05_server.log"
    stage="${MODE#pi05_}"
    exec bash "${TASKS_DIR}/run_l3b_bowl_order_pi05.sh" "${stage}"
    ;;
  *)
    echo "Usage: $0 prepare|check|safe-witness|pi05_smoke|pi05_formal" >&2
    exit 2
    ;;
esac
