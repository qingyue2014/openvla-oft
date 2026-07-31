#!/usr/bin/env bash
set -euo pipefail

# Locked capability-only extension over the first 20 official native states.
# The preregistration is committed before execution and its SHA-256 is bound
# into the result. This runner intentionally exposes no near/far smoke mode.

MODE="${1:-prepare}"
TASKS_DIR="experiments/robot/libero/tasks"
PREREGISTRATION="${TASKS_DIR}/l3b_moka_native20_prereg.json"
VALIDATOR="${TASKS_DIR}/validate_l3b_moka_native20_prereg.py"

echo "The native20_v1 screen is frozen under source commit 6883655." >&2
echo "This checkout implements the v2 same-pot Er/Ec pairing and refuses to" >&2
echo "regenerate or overwrite the completed v1 artifacts." >&2
exit 2

git ls-files --error-unmatch "${PREREGISTRATION}" >/dev/null
git diff --quiet -- "${PREREGISTRATION}"
git diff --cached --quiet -- "${PREREGISTRATION}"

export NUM_STATES=20
export SMOKE_TRIALS=20
export MIN_NATIVE_SUCCESSES=12
export SCENE_SEED=42
export EVAL_SEED=42
export PI05_REPLAN_STEPS=5
export POST_SUCCESS_SETTLE_STEPS=100
export MAX_VIDEOS_PER_OUTCOME=5
export RUN_TAG=pi05_native20_v1
export REVIEW_ROOT=review/L3-B_moka_order_task/native20_pi05_v1
export NATIVE_STATES="${TASKS_DIR}/l3b_moka_native20_native_states.hdf5"
export NEAR_STATES="${TASKS_DIR}/l3b_moka_native20_near_first_states.hdf5"
export FAR_STATES="${TASKS_DIR}/l3b_moka_native20_far_first_states.hdf5"
export INITIAL_GATE="${REVIEW_ROOT}/L3-B_moka_initial_gate_manifest.json"
export PAIRING_GATE="${REVIEW_ROOT}/L3-B_moka_pairing_gate.json"
export RUNTIME_REPLAY_GATE="${REVIEW_ROOT}/L3-B_moka_runtime_replay_gate.json"
export NATIVE_PREFLIGHT="${REVIEW_ROOT}/L3-B_moka_native_native_preflight.json"
export NEAR_PREFLIGHT="${REVIEW_ROOT}/L3-B_moka_near_first_native_preflight.json"
export FAR_PREFLIGHT="${REVIEW_ROOT}/L3-B_moka_far_first_native_preflight.json"
export NATIVE_CAPABILITY_REPORT="${REVIEW_ROOT}/L3-B_moka_native20_capability.json"
export TRAJECTORY_ROOT="${REVIEW_ROOT}/${RUN_TAG}_trajectories"
export CAPABILITY_PREREGISTRATION="${PREREGISTRATION}"

python_bin="${PYTHON_BIN:-python}"
"${python_bin}" "${VALIDATOR}" spec \
  --preregistration "${PREREGISTRATION}"

case "${MODE}" in
  prepare)
    bash "${TASKS_DIR}/run_l3b_moka_order.sh" prepare
    "${python_bin}" "${VALIDATOR}" prepared \
      --preregistration "${PREREGISTRATION}"
    ;;
  check)
    bash "${TASKS_DIR}/run_l3b_moka_order.sh" check
    "${python_bin}" "${VALIDATOR}" prepared \
      --preregistration "${PREREGISTRATION}"
    ;;
  native_capability)
    bash "${TASKS_DIR}/run_l3b_moka_order.sh" check
    "${python_bin}" "${VALIDATOR}" prepared \
      --preregistration "${PREREGISTRATION}"
    set +e
    bash "${TASKS_DIR}/run_l3b_moka_order_pi05.sh" native_capability
    capability_status=$?
    set -e
    "${python_bin}" "${VALIDATOR}" result \
      --preregistration "${PREREGISTRATION}" \
      --report "${NATIVE_CAPABILITY_REPORT}" \
      --out-binding "${REVIEW_ROOT}/L3-B_moka_native20_result_binding.json"
    exit "${capability_status}"
    ;;
  *)
    echo "Usage: $0 prepare|check|native_capability" >&2
    exit 2
    ;;
esac
