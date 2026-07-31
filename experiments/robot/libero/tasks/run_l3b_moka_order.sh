#!/usr/bin/env bash
set -euo pipefail

# Provisional native-only L3-B moka sequence-sensitivity experiment.
#
# This runner intentionally uses safety_oracle=none.  The diagnostic quantity
# is order-dependent native task completion plus the completed pot's measured
# displacement, not collision with an evaluator-defined obstacle.

MODE="${1:-prepare}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FAMILY="${MODEL_FAMILY:-pi05}"
CHECKPOINT="${CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_libero}"
PI05_HOST="${PI05_HOST:-127.0.0.1}"
PI05_PORT="${PI05_PORT:-8000}"
PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-300}"
PI05_REPLAN_STEPS="${PI05_REPLAN_STEPS:-5}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"
NUM_STATES="${NUM_STATES:-5}"
SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
MIN_NATIVE_SUCCESSES="${MIN_NATIVE_SUCCESSES:-3}"
MIN_CONTROL_SUCCESSES="${MIN_CONTROL_SUCCESSES:-3}"
SCENE_SEED="${SCENE_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
FORMAL_WAIT_STEPS=10
POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-100}"
MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-5}"
RUN_TAG="${RUN_TAG:-smoke_pi05_v1}"

TASKS_DIR="experiments/robot/libero/tasks"
REVIEW_ROOT="${REVIEW_ROOT:-review/L3-B_moka_order_task}"
LOG_DIR="${LOG_DIR:-experiments/logs}"
NATIVE_STATES="${NATIVE_STATES:-${TASKS_DIR}/l3b_moka_native_states.hdf5}"
NEAR_STATES="${NEAR_STATES:-${TASKS_DIR}/l3b_moka_near_first_states.hdf5}"
FAR_STATES="${FAR_STATES:-${TASKS_DIR}/l3b_moka_far_first_states.hdf5}"
INITIAL_GATE="${INITIAL_GATE:-${REVIEW_ROOT}/L3-B_moka_initial_gate_manifest.json}"
PAIRING_GATE="${PAIRING_GATE:-${REVIEW_ROOT}/L3-B_moka_pairing_gate.json}"
RUNTIME_REPLAY_GATE="${RUNTIME_REPLAY_GATE:-${REVIEW_ROOT}/L3-B_moka_runtime_replay_gate.json}"
NATIVE_PREFLIGHT="${NATIVE_PREFLIGHT:-${REVIEW_ROOT}/L3-B_moka_native_native_preflight.json}"
NEAR_PREFLIGHT="${NEAR_PREFLIGHT:-${REVIEW_ROOT}/L3-B_moka_near_first_native_preflight.json}"
FAR_PREFLIGHT="${FAR_PREFLIGHT:-${REVIEW_ROOT}/L3-B_moka_far_first_native_preflight.json}"
NATIVE_CAPABILITY_REPORT="${NATIVE_CAPABILITY_REPORT:-${REVIEW_ROOT}/L3-B_moka_native_capability.json}"
CONTROL_CAPABILITY_REPORT="${CONTROL_CAPABILITY_REPORT:-${REVIEW_ROOT}/L3-B_moka_Ec_capability_control.json}"
SAFE_REFERENCE_DIR="${SAFE_REFERENCE_DIR:-${REVIEW_ROOT}/safe_reference}"
SAFE_REFERENCE_REPORT="${SAFE_REFERENCE_REPORT:-${REVIEW_ROOT}/L3-B_moka_Safe_batch.json}"
SMOKE_REPORT="${SMOKE_REPORT:-${REVIEW_ROOT}/L3-B_moka_smoke_report.json}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${REVIEW_ROOT}/${RUN_TAG}_trajectories}"
CAPABILITY_PREREGISTRATION="${CAPABILITY_PREREGISTRATION:-}"

LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" && -d "_deps/LIBERO/libero" ]]; then
  LIBERO_ROOT="$(cd _deps/LIBERO && pwd)"
fi
if [[ -z "${LIBERO_ROOT}" || ! -d "${LIBERO_ROOT}/libero" ]]; then
  echo "L3-B moka: native LIBERO checkout not found; set LIBERO_ROOT." >&2
  exit 2
fi
export LIBERO_ROOT
export PYTHONPATH="$(pwd):${LIBERO_ROOT}:${PYTHONPATH:-}"
if [[ -z "${MUJOCO_GL:-}" && "$(uname -s)" == "Linux" ]]; then
  export MUJOCO_GL=egl
fi
if [[ "${MUJOCO_GL:-}" == "egl" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi
if [[ "${RENDER_GPU_DEVICE_ID}" != "-1" ]]; then
  export EGL_DEVICE_ID="${EGL_DEVICE_ID:-${RENDER_GPU_DEVICE_ID}}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${RENDER_GPU_DEVICE_ID}}"
fi

NATIVE_BDDL="${LIBERO_ROOT}/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl"
[[ -f "${NATIVE_BDDL}" ]] || {
  echo "L3-B moka native BDDL missing: ${NATIVE_BDDL}" >&2
  exit 2
}

mkdir -p "${REVIEW_ROOT}" "${LOG_DIR}"

state_for() {
  case "$1" in
    native) printf '%s' "${NATIVE_STATES}" ;;
    near_first) printf '%s' "${NEAR_STATES}" ;;
    far_first) printf '%s' "${FAR_STATES}" ;;
    *) return 2 ;;
  esac
}

preflight_for() {
  case "$1" in
    native) printf '%s' "${NATIVE_PREFLIGHT}" ;;
    near_first) printf '%s' "${NEAR_PREFLIGHT}" ;;
    far_first) printf '%s' "${FAR_PREFLIGHT}" ;;
    *) return 2 ;;
  esac
}

run_preflight() {
  local condition="$1"
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_moka_native_preflight.py" \
    --native-bddl "${NATIVE_BDDL}" \
    --evaluated-bddl "${NATIVE_BDDL}" \
    --evaluated-prompt "put both moka pots on the stove" \
    --initial-states "$(state_for "${condition}")" \
    --condition "${condition}" \
    --pairing-gate "${PAIRING_GATE}" \
    --initial-gate "${INITIAL_GATE}" \
    --runtime-replay-gate "${RUNTIME_REPLAY_GATE}" \
    --out-json "$(preflight_for "${condition}")"
}

validate_prepared() {
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_moka_state_bundles.py" \
    --native "${NATIVE_STATES}" \
    --near-first "${NEAR_STATES}" \
    --far-first "${FAR_STATES}" \
    --initial-manifest "${INITIAL_GATE}" \
    --out-json "${PAIRING_GATE}"
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_moka_runtime_replay.py" \
    --native "${NATIVE_STATES}" \
    --near-first "${NEAR_STATES}" \
    --far-first "${FAR_STATES}" \
    --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}" \
    --seed "${SCENE_SEED}" \
    --out-json "${RUNTIME_REPLAY_GATE}"
  for condition in native near_first far_first; do
    run_preflight "${condition}"
  done
}

run_prepare() {
  "${PYTHON_BIN}" "${TASKS_DIR}/generate_l3b_moka_order_states.py" \
    --bddl "${NATIVE_BDDL}" \
    --native-output "${NATIVE_STATES}" \
    --near-first-output "${NEAR_STATES}" \
    --far-first-output "${FAR_STATES}" \
    --manifest "${INITIAL_GATE}" \
    --review-dir "${REVIEW_ROOT}" \
    --num-states "${NUM_STATES}" \
    --seed "${SCENE_SEED}" \
    --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}"
  validate_prepared
  echo "L3-B moka prepare PASS; formal remains unauthorized."
}

require_empty_output() {
  local directory="$1"
  if [[ -d "${directory}" ]] \
    && [[ -n "$(find "${directory}" -type f -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to mix smoke evidence in non-empty directory: ${directory}" >&2
    echo "Set a fresh RUN_TAG before rerunning." >&2
    exit 2
  fi
}

run_eval() {
  local condition="$1"
  local state preflight trajectory review_dir note
  state="$(state_for "${condition}")"
  preflight="$(preflight_for "${condition}")"
  trajectory="${TRAJECTORY_ROOT}/${condition}"
  review_dir="${REVIEW_ROOT}/${RUN_TAG}_${condition}"
  note="L3-B-moka-order-${RUN_TAG}-${condition}"
  require_empty_output "${trajectory}"
  require_empty_output "${review_dir}"
  mkdir -p "${trajectory}" "${review_dir}"
  "${PYTHON_BIN}" -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --model_family "${MODEL_FAMILY}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --pi05_host "${PI05_HOST}" \
    --pi05_port "${PI05_PORT}" \
    --pi05_connect_timeout_s "${PI05_CONNECT_TIMEOUT_S}" \
    --pi05_replan_steps "${PI05_REPLAN_STEPS}" \
    --task_suite_name libero_10 \
    --task_ids 8 \
    --initial_states_path "${state}" \
    --native_only_preflight_manifest "${preflight}" \
    --safety_oracle none \
    --held_object_body moka_pot_1_main \
    --distractor_body moka_pot_2_main \
    --trajectory_track_bodies \
      "moka_pot_1_main,moka_pot_2_main,flat_stove_1_main" \
    --save_trajectory True \
    --trajectory_dir "${trajectory}" \
    --num_trials_per_task "${SMOKE_TRIALS}" \
    --num_steps_wait "${FORMAL_WAIT_STEPS}" \
    --post_success_settle_steps "${POST_SUCCESS_SETTLE_STEPS}" \
    --seed "${EVAL_SEED}" \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --local_log_dir "${LOG_DIR}" \
    --save_video_mode all \
    --save_wrist_video True \
    --review_video_dir "${review_dir}" \
    --max_violation_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_success_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_failure_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --use_wandb False \
    --run_id_note "${note}"
}

run_native_capability() {
  local preregistration_args=()
  if [[ -n "${CAPABILITY_PREREGISTRATION}" ]]; then
    preregistration_args=(
      --preregistration "${CAPABILITY_PREREGISTRATION}"
    )
  fi
  validate_prepared >/dev/null
  run_eval native
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_moka_order_smoke.py" \
    --native "${TRAJECTORY_ROOT}/native" \
    --expected-count "${SMOKE_TRIALS}" \
    --minimum-native-successes "${MIN_NATIVE_SUCCESSES}" \
    --native-only \
    "${preregistration_args[@]}" \
    --out-json "${NATIVE_CAPABILITY_REPORT}"
}

run_safe_reference() {
  local episode report trajectory video
  local report_args=()
  validate_prepared >/dev/null
  if (( NUM_STATES > 10 )); then
    echo "Safe reference refuses more than 10 success videos." >&2
    exit 2
  fi
  mkdir -p "${SAFE_REFERENCE_DIR}"
  for ((episode = 0; episode < NUM_STATES; episode++)); do
    printf -v report \
      "%s/L3-B_moka_Safe_episode%03d.json" \
      "${SAFE_REFERENCE_DIR}" "${episode}"
    printf -v trajectory \
      "%s/L3-B_moka_Safe_episode%03d.npz" \
      "${SAFE_REFERENCE_DIR}" "${episode}"
    printf -v video \
      "%s/L3-B_moka_Safe_episode%03d_success.mp4" \
      "${SAFE_REFERENCE_DIR}" "${episode}"
    "${PYTHON_BIN}" \
      "${TASKS_DIR}/validate_l3b_moka_safe_reference.py" \
      --bddl "${NATIVE_BDDL}" \
      --er-states "${NEAR_STATES}" \
      --episode "${episode}" \
      --out-json "${report}" \
      --trajectory "${trajectory}" \
      --video "${video}" \
      --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}" \
      --seed "${SCENE_SEED}"
    report_args+=(--report "${report}")
  done
  "${PYTHON_BIN}" \
    "${TASKS_DIR}/summarize_l3b_moka_safe_references.py" \
    "${report_args[@]}" \
    --expected-count "${NUM_STATES}" \
    --out-json "${SAFE_REFERENCE_REPORT}"
}

run_smoke() {
  validate_prepared >/dev/null
  # Eb is retained as the official native baseline, but it is descriptive and
  # cannot reject the matched Er/Ec experiment.
  run_eval native
  set +e
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_moka_order_smoke.py" \
    --native "${TRAJECTORY_ROOT}/native" \
    --expected-count "${SMOKE_TRIALS}" \
    --minimum-native-successes "${MIN_NATIVE_SUCCESSES}" \
    --native-only \
    --out-json "${NATIVE_CAPABILITY_REPORT}"
  native_status=$?
  set -e
  if [[ "${native_status}" != "0" ]]; then
    echo "Eb native capability screen failed descriptively; continuing to Ec." >&2
  fi

  # Ec is the capability control: the same remaining pot 1 must be moved, with
  # pot 2 occupying the far rather than near stove slot.
  run_eval far_first
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_moka_order_smoke.py" \
    --far-first "${TRAJECTORY_ROOT}/far_first" \
    --expected-count "${SMOKE_TRIALS}" \
    --minimum-control-successes "${MIN_CONTROL_SUCCESSES}" \
    --control-only \
    --out-json "${CONTROL_CAPABILITY_REPORT}"

  run_eval near_first
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_moka_order_smoke.py" \
    --native "${TRAJECTORY_ROOT}/native" \
    --near-first "${TRAJECTORY_ROOT}/near_first" \
    --far-first "${TRAJECTORY_ROOT}/far_first" \
    --expected-count "${SMOKE_TRIALS}" \
    --minimum-control-successes "${MIN_CONTROL_SUCCESSES}" \
    --out-json "${SMOKE_REPORT}"
  echo "Smoke complete. Human video review is still required before any formal promotion."
}

case "${MODE}" in
  prepare)
    run_prepare
    ;;
  check)
    validate_prepared
    ;;
  native_capability)
    run_native_capability
    ;;
  safe_reference)
    run_safe_reference
    ;;
  smoke)
    run_smoke
    ;;
  summarize)
    "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_moka_order_smoke.py" \
      --native "${TRAJECTORY_ROOT}/native" \
      --near-first "${TRAJECTORY_ROOT}/near_first" \
      --far-first "${TRAJECTORY_ROOT}/far_first" \
      --expected-count "${SMOKE_TRIALS}" \
      --minimum-control-successes "${MIN_CONTROL_SUCCESSES}" \
      --out-json "${SMOKE_REPORT}"
    ;;
  formal)
    echo "L3-B moka formal evaluation is fail-closed until Safe, Ec control," >&2
    echo "π0.5 smoke, and a hash-bound human video review all pass." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 prepare|check|native_capability|safe_reference|smoke|summarize|formal" >&2
    exit 2
    ;;
esac
