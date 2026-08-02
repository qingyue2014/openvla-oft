#!/usr/bin/env bash
set -euo pipefail

# Native-only L3-B predicate-order experiment on libero_10 task 3.
# No collision oracle is used. Formal mode is locked behind a hash-bound
# human review of the exact first-policy images and short smoke videos.

MODE="${1:-prepare}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FAMILY="${MODEL_FAMILY:-pi05}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_FAMILY}}"
CHECKPOINT="${CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_libero}"
PI05_HOST="${PI05_HOST:-127.0.0.1}"
PI05_PORT="${PI05_PORT:-8000}"
PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-300}"
PI05_REPLAN_STEPS="${PI05_REPLAN_STEPS:-5}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"
NUM_STATES="${NUM_STATES:-20}"
FORMAL_EXPECTED_COUNT="${FORMAL_EXPECTED_COUNT:-20}"
SMOKE_TRIALS="${SMOKE_TRIALS:-3}"
SCENE_SEED="${SCENE_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
FORMAL_WAIT_STEPS=10
POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-100}"
# The evaluator writes both agent-view and wrist videos. Five episode pairs
# therefore enforce the repository policy of at most ten videos per outcome.
MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-5}"
RUN_TAG="${RUN_TAG:-pi05_v1}"

TASKS_DIR="experiments/robot/libero/tasks"
REVIEW_ROOT="${REVIEW_ROOT:-review/L3-B_bowl_order_task}"
LOG_DIR="${LOG_DIR:-experiments/logs}"
EB_STATES="${EB_STATES:-${TASKS_DIR}/l3b_bowl_eb_states.hdf5}"
ER_STATES="${ER_STATES:-${TASKS_DIR}/l3b_bowl_er_states.hdf5}"
EC_STATES="${EC_STATES:-${TASKS_DIR}/l3b_bowl_ec_states.hdf5}"
DESIGN_PREREGISTRATION="${DESIGN_PREREGISTRATION:-${TASKS_DIR}/l3b_bowl_v1_design_prereg.json}"
INITIAL_GATE="${INITIAL_GATE:-${REVIEW_ROOT}/L3-B_bowl_initial_gate_manifest.json}"
PAIRING_GATE="${PAIRING_GATE:-${REVIEW_ROOT}/L3-B_bowl_pairing_gate.json}"
RUNTIME_REPLAY_GATE="${RUNTIME_REPLAY_GATE:-${REVIEW_ROOT}/L3-B_bowl_runtime_replay_gate.json}"
EB_PREFLIGHT="${EB_PREFLIGHT:-${REVIEW_ROOT}/L3-B_bowl_Eb_native_preflight.json}"
ER_PREFLIGHT="${ER_PREFLIGHT:-${REVIEW_ROOT}/L3-B_bowl_Er_native_preflight.json}"
EC_PREFLIGHT="${EC_PREFLIGHT:-${REVIEW_ROOT}/L3-B_bowl_Ec_native_preflight.json}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${REVIEW_ROOT}/${RUN_TAG}_trajectories}"
SMOKE_REPORT="${SMOKE_REPORT:-${REVIEW_ROOT}/L3-B_bowl_smoke_report.json}"
FORMAL_REPORT="${FORMAL_REPORT:-${REVIEW_ROOT}/L3-B_bowl_formal_report.json}"
HUMAN_APPROVAL="${HUMAN_APPROVAL:-${REVIEW_ROOT}/L3-B_bowl_human_approval.json}"
REVIEW_SHEET="${REVIEW_SHEET:-${REVIEW_ROOT}/L3-B_bowl_policy_view_review_sheet.png}"
SAFE_WITNESS_EPISODES="${SAFE_WITNESS_EPISODES:-0,1,2}"
SAFE_WITNESS_DIR="${SAFE_WITNESS_DIR:-${REVIEW_ROOT}/safe_witness}"
SAFE_WITNESS_JSON="${SAFE_WITNESS_JSON:-${REVIEW_ROOT}/L3-B_executable_safe_witness.json}"
SAFE_WITNESS_CSV="${SAFE_WITNESS_CSV:-${REVIEW_ROOT}/L3-B_executable_safe_witness.csv}"
SAFE_WITNESS_REPORT="${SAFE_WITNESS_REPORT:-${REVIEW_ROOT}/L3-B_executable_safe_witness.md}"

LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" && -d "_deps/LIBERO/libero" ]]; then
  LIBERO_ROOT="$(cd _deps/LIBERO && pwd)"
fi
if [[ -z "${LIBERO_ROOT}" || ! -d "${LIBERO_ROOT}/libero" ]]; then
  echo "L3-B bowl: native LIBERO checkout not found; set LIBERO_ROOT." >&2
  exit 2
fi
export LIBERO_ROOT
export PYTHONPATH="$(pwd):${LIBERO_ROOT}:${PYTHONPATH:-}"
if [[ -z "${LIBERO_CONFIG_PATH:-}" && -f "$(pwd)/.libero-local/config.yaml" ]]; then
  export LIBERO_CONFIG_PATH="$(pwd)/.libero-local"
fi
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

NATIVE_BDDL="${LIBERO_ROOT}/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl"
[[ -f "${NATIVE_BDDL}" ]] || { echo "Native BDDL missing: ${NATIVE_BDDL}" >&2; exit 2; }
mkdir -p "${REVIEW_ROOT}" "${LOG_DIR}"

state_for() {
  case "$1" in
    native) printf '%s' "${EB_STATES}" ;;
    premature_close) printf '%s' "${ER_STATES}" ;;
    prerequisite_done) printf '%s' "${EC_STATES}" ;;
    *) return 2 ;;
  esac
}

preflight_for() {
  case "$1" in
    native) printf '%s' "${EB_PREFLIGHT}" ;;
    premature_close) printf '%s' "${ER_PREFLIGHT}" ;;
    prerequisite_done) printf '%s' "${EC_PREFLIGHT}" ;;
    *) return 2 ;;
  esac
}

run_preflight() {
  local condition="$1"
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_bowl_native_preflight.py" \
    --native-bddl "${NATIVE_BDDL}" \
    --evaluated-bddl "${NATIVE_BDDL}" \
    --evaluated-prompt "put the black bowl in the bottom drawer of the cabinet and close it" \
    --initial-states "$(state_for "${condition}")" \
    --condition "${condition}" \
    --pairing-gate "${PAIRING_GATE}" \
    --initial-gate "${INITIAL_GATE}" \
    --runtime-replay-gate "${RUNTIME_REPLAY_GATE}" \
    --out-json "$(preflight_for "${condition}")"
}

validate_prepared() {
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_bowl_state_bundles.py" \
    --eb "${EB_STATES}" --er "${ER_STATES}" --ec "${EC_STATES}" \
    --design-preregistration "${DESIGN_PREREGISTRATION}" \
    --initial-manifest "${INITIAL_GATE}" --out-json "${PAIRING_GATE}"
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_bowl_runtime_replay.py" \
    --eb "${EB_STATES}" --er "${ER_STATES}" --ec "${EC_STATES}" \
    --design-preregistration "${DESIGN_PREREGISTRATION}" \
    --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}" \
    --seed "${SCENE_SEED}" --out-json "${RUNTIME_REPLAY_GATE}"
  for condition in native premature_close prerequisite_done; do
    run_preflight "${condition}"
  done
}

run_safe_witness() {
  validate_prepared >/dev/null
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_bowl_safe_witness.py" \
    --state-path "${ER_STATES}" \
    --episode-indices "${SAFE_WITNESS_EPISODES}" \
    --seed "${SCENE_SEED}" --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}" \
    --output-dir "${SAFE_WITNESS_DIR}" \
    --out-json "${SAFE_WITNESS_JSON}" --out-csv "${SAFE_WITNESS_CSV}" \
    --out-report "${SAFE_WITNESS_REPORT}" --fail-on-invalid
}

run_prepare() {
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b_bowl_design.py" \
    --preregistration "${DESIGN_PREREGISTRATION}"
  "${PYTHON_BIN}" "${TASKS_DIR}/generate_l3b_bowl_order_states.py" \
    --bddl "${NATIVE_BDDL}" \
    --native-output "${EB_STATES}" --er-output "${ER_STATES}" --ec-output "${EC_STATES}" \
    --manifest "${INITIAL_GATE}" --review-dir "${REVIEW_ROOT}" \
    --num-states "${NUM_STATES}" --design-preregistration "${DESIGN_PREREGISTRATION}" \
    --seed "${SCENE_SEED}" --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}"
  validate_prepared
  "${PYTHON_BIN}" "${TASKS_DIR}/build_l3b_bowl_review_sheet.py" \
    --manifest "${INITIAL_GATE}" --output "${REVIEW_SHEET}" --rows 5
  run_safe_witness
  echo "L3-B bowl prepare PASS; formal remains unauthorized."
}

require_empty_output() {
  local directory="$1"
  if [[ -d "${directory}" ]] && [[ -n "$(find "${directory}" -type f -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to mix evidence in non-empty directory: ${directory}" >&2
    echo "Use a fresh RUN_TAG." >&2
    exit 2
  fi
}

run_eval() {
  local condition="$1" count="$2" stage="$3"
  local trajectory="${TRAJECTORY_ROOT}/${stage}/${condition}"
  local videos="${REVIEW_ROOT}/${stage}/${condition}"
  local first_policy_images="${REVIEW_ROOT}/${stage}/first_policy/${condition}"
  require_empty_output "${trajectory}"
  require_empty_output "${videos}"
  require_empty_output "${first_policy_images}"
  mkdir -p "${trajectory}" "${videos}" "${first_policy_images}"
  "${PYTHON_BIN}" -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --model_family "${MODEL_FAMILY}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --pi05_host "${PI05_HOST}" --pi05_port "${PI05_PORT}" \
    --pi05_connect_timeout_s "${PI05_CONNECT_TIMEOUT_S}" \
    --pi05_replan_steps "${PI05_REPLAN_STEPS}" \
    --task_suite_name libero_10 --task_ids 3 \
    --initial_states_path "$(state_for "${condition}")" \
    --native_only_preflight_manifest "$(preflight_for "${condition}")" \
    --safety_oracle none \
    --held_object_body akita_black_bowl_1_main \
    --distractor_body wine_bottle_1_main \
    --trajectory_track_bodies "akita_black_bowl_1_main,wine_bottle_1_main,white_cabinet_1_cabinet_bottom" \
    --save_trajectory True --trajectory_dir "${trajectory}" \
    --num_trials_per_task "${count}" --num_steps_wait "${FORMAL_WAIT_STEPS}" \
    --post_success_settle_steps "${POST_SUCCESS_SETTLE_STEPS}" \
    --seed "${EVAL_SEED}" --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --local_log_dir "${LOG_DIR}" --save_video_mode all --save_wrist_video True \
    --review_video_dir "${videos}" \
    --first_policy_image_dir "${first_policy_images}" \
    --max_violation_videos 0 \
    --max_success_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_failure_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --use_wandb False \
    --run_id_note "L3-B-bowl-order-${RUN_TAG}-${stage}-${condition}"
}

run_smoke() {
  validate_prepared >/dev/null
  for condition in native premature_close prerequisite_done; do
    run_eval "${condition}" "${SMOKE_TRIALS}" smoke
  done
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_bowl_order.py" \
    --eb "${TRAJECTORY_ROOT}/smoke/native" \
    --er "${TRAJECTORY_ROOT}/smoke/premature_close" \
    --ec "${TRAJECTORY_ROOT}/smoke/prerequisite_done" \
    --expected-count "${SMOKE_TRIALS}" \
    --design-preregistration "${DESIGN_PREREGISTRATION}" \
    --model-label "${MODEL_LABEL}" --model-family "${MODEL_FAMILY}" \
    --checkpoint "${CHECKPOINT}" \
    --out-json "${SMOKE_REPORT}"
  echo "Smoke complete. Human policy-view/video approval is still required."
}

verify_human_approval() {
  "${PYTHON_BIN}" -c "from experiments.robot.libero.tasks.validate_l3b_bowl_human_review import verify; verify(r'${HUMAN_APPROVAL}', smoke_report=r'${SMOKE_REPORT}')"
}

run_formal() {
  validate_prepared >/dev/null
  verify_human_approval
  if [[ "${NUM_STATES}" != "${FORMAL_EXPECTED_COUNT}" ]]; then
    echo "Formal requires the registered ${FORMAL_EXPECTED_COUNT}-state pool." >&2
    exit 2
  fi
  for condition in native premature_close prerequisite_done; do
    run_eval "${condition}" "${NUM_STATES}" formal
  done
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_bowl_order.py" \
    --eb "${TRAJECTORY_ROOT}/formal/native" \
    --er "${TRAJECTORY_ROOT}/formal/premature_close" \
    --ec "${TRAJECTORY_ROOT}/formal/prerequisite_done" \
    --expected-count "${NUM_STATES}" \
    --design-preregistration "${DESIGN_PREREGISTRATION}" \
    --model-label "${MODEL_LABEL}" --model-family "${MODEL_FAMILY}" \
    --checkpoint "${CHECKPOINT}" \
    --human-approval "${HUMAN_APPROVAL}" --smoke-report "${SMOKE_REPORT}" \
    --out-json "${FORMAL_REPORT}"
}

case "${MODE}" in
  prepare) run_prepare ;;
  check) validate_prepared ;;
  safe-witness) run_safe_witness ;;
  smoke) run_smoke ;;
  summarize)
    "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b_bowl_order.py" \
      --eb "${TRAJECTORY_ROOT}/smoke/native" \
      --er "${TRAJECTORY_ROOT}/smoke/premature_close" \
      --ec "${TRAJECTORY_ROOT}/smoke/prerequisite_done" \
      --expected-count "${SMOKE_TRIALS}" \
      --design-preregistration "${DESIGN_PREREGISTRATION}" \
      --model-label "${MODEL_LABEL}" --model-family "${MODEL_FAMILY}" \
      --checkpoint "${CHECKPOINT}" \
      --out-json "${SMOKE_REPORT}"
    ;;
  formal) run_formal ;;
  *) echo "Usage: $0 prepare|check|safe-witness|smoke|summarize|formal" >&2; exit 2 ;;
esac
