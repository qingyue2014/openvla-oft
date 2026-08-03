#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-design}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"
NUM_STATES="${NUM_STATES:-20}"
SMOKE_TRIALS="${SMOKE_TRIALS:-3}"
MIN_NATIVE_SUCCESSES="${MIN_NATIVE_SUCCESSES:-2}"
SCENE_SEED="${SCENE_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-10}"
RUN_TAG="${RUN_TAG:-openvla_oft_smoke_v7_native_like_candidate}"

TASKS_DIR="experiments/robot/libero/tasks"
REVIEW_ROOT="${REVIEW_ROOT:-review/L3-B3_task}"
LOG_DIR="${LOG_DIR:-experiments/logs}"
NATIVE_STATES="${NATIVE_STATES:-${TASKS_DIR}/l3b3_microwave_native_states.hdf5}"
ER_STATES="${ER_STATES:-${TASKS_DIR}/l3b3_microwave_closed_states.hdf5}"
EC_STATES="${EC_STATES:-${TASKS_DIR}/l3b3_microwave_open_control_states.hdf5}"
DESIGN_PREREG="${DESIGN_PREREG:-${TASKS_DIR}/l3b3_microwave_v7_design_prereg.json}"
CONTROLLER_PREREG="${CONTROLLER_PREREG:-${TASKS_DIR}/l3b3_microwave_v5_handle_grasp_controller_v3_prereg.json}"
CONTROLLER_INVALIDATION="${CONTROLLER_INVALIDATION:-${TASKS_DIR}/l3b3_microwave_v5_handle_grasp_controller_v3_invalidation.json}"
CONTROLLER_INVALIDATION_SHA256="8ed88f25359da1156315968e93d6dcc8d4511883be8255761e4c57ff9e3ea8f2"
DESIGN_PREFLIGHT="${DESIGN_PREFLIGHT:-${REVIEW_ROOT}/L3-B3_design_preflight.json}"
INITIAL_GATE="${INITIAL_GATE:-${REVIEW_ROOT}/L3-B3_initial_gate_manifest.json}"
PAIRING_GATE="${PAIRING_GATE:-${REVIEW_ROOT}/L3-B3_pairing_gate.json}"
RUNTIME_GATE="${RUNTIME_GATE:-${REVIEW_ROOT}/L3-B3_runtime_replay_gate.json}"
NATIVE_PREFLIGHT="${NATIVE_PREFLIGHT:-${REVIEW_ROOT}/L3-B3_native_only_preflight.json}"
OPENVLA_VIEW_GATE="${OPENVLA_VIEW_GATE:-${REVIEW_ROOT}/L3-B3_openvla_policy_view_gate.json}"
OPENVLA_VIEW_DIR="${OPENVLA_VIEW_DIR:-${REVIEW_ROOT}/openvla_policy_views}"
SAFE_REPORT="${SAFE_REPORT:-${REVIEW_ROOT}/L3-B3_safe_reference.json}"
SAFE_REVIEW_DIR="${SAFE_REVIEW_DIR:-${REVIEW_ROOT}/safe_reference_v5_handle_grasp_v2}"
NATIVE_SMOKE_REPORT="${NATIVE_SMOKE_REPORT:-${REVIEW_ROOT}/L3-B3_openvla_native_smoke.json}"
PAIRED_SMOKE_REPORT="${PAIRED_SMOKE_REPORT:-${REVIEW_ROOT}/L3-B3_openvla_paired_smoke.json}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${REVIEW_ROOT}/${RUN_TAG}_trajectories}"
PI05_EB_DIAGNOSTIC_ROOT="${PI05_EB_DIAGNOSTIC_ROOT:-review/L3-B2_task/moved_cup_pi05_eb_diagnostic}"
PI05_EB_DIAGNOSTIC_TRAJECTORIES="${PI05_EB_DIAGNOSTIC_TRAJECTORIES:-${PI05_EB_DIAGNOSTIC_ROOT}/trajectories}"
PI05_EB_DIAGNOSTIC_VIDEOS="${PI05_EB_DIAGNOSTIC_VIDEOS:-${PI05_EB_DIAGNOSTIC_ROOT}/videos}"
PI05_EB_DIAGNOSTIC_REPORT="${PI05_EB_DIAGNOSTIC_REPORT:-${PI05_EB_DIAGNOSTIC_ROOT}/L3-B2_moved_cup_pi05_eb_diagnostic.json}"
PI05_ER_DIAGNOSTIC_ROOT="${PI05_ER_DIAGNOSTIC_ROOT:-${REVIEW_ROOT}/pi05_er_checkpoint_diagnostic}"
PI05_ER_DIAGNOSTIC_TRAJECTORIES="${PI05_ER_DIAGNOSTIC_TRAJECTORIES:-${PI05_ER_DIAGNOSTIC_ROOT}/trajectories}"
PI05_ER_DIAGNOSTIC_VIDEOS="${PI05_ER_DIAGNOSTIC_VIDEOS:-${PI05_ER_DIAGNOSTIC_ROOT}/videos}"
PI05_ER_DIAGNOSTIC_REPORT="${PI05_ER_DIAGNOSTIC_REPORT:-${PI05_ER_DIAGNOSTIC_ROOT}/L3-B3_pi05_er_checkpoint_diagnostic.json}"

LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" && -d "_deps/LIBERO/libero" ]]; then
  LIBERO_ROOT="$(cd _deps/LIBERO && pwd)"
fi
if [[ -z "${LIBERO_ROOT}" || ! -d "${LIBERO_ROOT}/libero" ]]; then
  echo "L3-B3: native LIBERO checkout not found; set LIBERO_ROOT." >&2
  exit 2
fi
export LIBERO_ROOT
export PYTHONPATH="$(pwd):${LIBERO_ROOT}:${PYTHONPATH:-}"
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  LIBERO_CONFIG_PATH="$("${PYTHON_BIN}" -c \
    'from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import ensure_libero_config; print(ensure_libero_config())')"
fi
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  echo "L3-B3: could not create a checkout-local LIBERO config." >&2
  exit 2
fi
export LIBERO_CONFIG_PATH
if [[ -z "${MUJOCO_GL:-}" && "$(uname -s)" == "Linux" ]]; then
  export MUJOCO_GL=egl
fi
if [[ "${MUJOCO_GL:-}" == "egl" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

NATIVE_BDDL="${LIBERO_ROOT}/libero/libero/bddl_files/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl"
[[ -f "${NATIVE_BDDL}" ]] || {
  echo "L3-B3 native BDDL missing: ${NATIVE_BDDL}" >&2
  exit 2
}
mkdir -p "${REVIEW_ROOT}" "${LOG_DIR}"

run_design() {
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b3_microwave_design.py" \
    --spec "${DESIGN_PREREG}" \
    --native-bddl "${NATIVE_BDDL}" \
    --output "${DESIGN_PREFLIGHT}"
}

run_prepare() {
  run_design
  "${PYTHON_BIN}" "${TASKS_DIR}/generate_l3b3_microwave_precondition_states.py" \
    --bddl "${NATIVE_BDDL}" \
    --native-output "${NATIVE_STATES}" \
    --er-output "${ER_STATES}" \
    --ec-output "${EC_STATES}" \
    --manifest "${INITIAL_GATE}" \
    --review-dir "${REVIEW_ROOT}" \
    --num-states "${NUM_STATES}" \
    --design-preregistration "${DESIGN_PREREG}" \
    --seed "${SCENE_SEED}" \
    --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}"
  INCLUDE_OPENVLA_VIEW_GATE=0 run_check
}

run_check() {
  local -a command=(
    "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b3_microwave_artifacts.py"
    --manifest "${INITIAL_GATE}"
    --native "${NATIVE_STATES}"
    --er "${ER_STATES}"
    --ec "${EC_STATES}"
    --pairing-output "${PAIRING_GATE}"
    --runtime-output "${RUNTIME_GATE}"
    --preflight-output "${NATIVE_PREFLIGHT}"
  )
  if [[ "${INCLUDE_OPENVLA_VIEW_GATE:-1}" == "1" && -f "${OPENVLA_VIEW_GATE}" ]]; then
    command+=(
      --openvla-view-gate "${OPENVLA_VIEW_GATE}"
      --openvla-checkpoint "${CHECKPOINT}"
    )
  fi
  command+=(
    --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}"
    --seed "${SCENE_SEED}"
  )
  "${command[@]}"
}

run_policy_views() {
  "${PYTHON_BIN}" "${TASKS_DIR}/certify_l3b3_openvla_policy_views.py" \
    --manifest "${INITIAL_GATE}" \
    --output-dir "${OPENVLA_VIEW_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OPENVLA_VIEW_GATE}"
  run_check
}

run_safe_reference() {
  if [[ -f "${CONTROLLER_INVALIDATION}" ]]; then
    "${PYTHON_BIN}" -c \
      'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; actual=hashlib.sha256(p.read_bytes()).hexdigest(); record=json.loads(p.read_text()); assert actual == expected; assert record["controller_revision"] == 3; assert record["formal_authorized"] is False; print("PASS_L3B3_INVALIDATED_CONTROLLER_BOUND")' \
      "${CONTROLLER_INVALIDATION}" "${CONTROLLER_INVALIDATION_SHA256}"
    echo "L3-B3 Safe stopped: controller revision 3 is invalidated; v6 requires a fresh controller preregistration." >&2
    exit 2
  fi
  run_check >/dev/null
  if [[ -e "${SAFE_REPORT}" ]]; then
    echo "Refusing to overwrite L3-B3 Safe report: ${SAFE_REPORT}" >&2
    exit 2
  fi
  require_empty "${SAFE_REVIEW_DIR}"
  "${PYTHON_BIN}" "${TASKS_DIR}/validate_l3b3_microwave_safe_reference.py" \
    --er-states "${ER_STATES}" \
    --num-states "${NUM_STATES}" \
    --seed "${SCENE_SEED}" \
    --render-gpu-device-id "${RENDER_GPU_DEVICE_ID}" \
    --review-dir "${SAFE_REVIEW_DIR}" \
    --controller-prereg "${CONTROLLER_PREREG}" \
    --output "${SAFE_REPORT}"
}

state_for() {
  case "$1" in
    native) printf '%s' "${NATIVE_STATES}" ;;
    closed_microwave) printf '%s' "${ER_STATES}" ;;
    open_control) printf '%s' "${EC_STATES}" ;;
    *) return 2 ;;
  esac
}

require_empty_output() {
  local directory="$1"
  if [[ -d "${directory}" ]] \
    && [[ -n "$(find "${directory}" -type f -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to mix L3-B3 evidence in non-empty directory: ${directory}" >&2
    echo "Set a fresh RUN_TAG before rerunning." >&2
    exit 2
  fi
}

run_eval() {
  local condition="$1"
  local state trajectory videos
  state="$(state_for "${condition}")"
  trajectory="${TRAJECTORY_ROOT}/${condition}"
  videos="${REVIEW_ROOT}/smoke/${RUN_TAG}/${condition}"
  require_empty_output "${trajectory}"
  require_empty_output "${videos}"
  mkdir -p "${trajectory}" "${videos}"
  "${PYTHON_BIN}" -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --model_family openvla \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_10 \
    --task_ids 9 \
    --initial_states_path "${state}" \
    --native_only_preflight_manifest "${NATIVE_PREFLIGHT}" \
    --safety_oracle none \
    --held_object_body white_yellow_mug_1_main \
    --distractor_body porcelain_mug_1_main \
    --trajectory_track_bodies \
      "white_yellow_mug_1_main,porcelain_mug_1_main,microwave_1_main,microwave_1_microdoorroot" \
    --center_crop True \
    --num_images_in_input 2 \
    --save_trajectory True \
    --trajectory_dir "${trajectory}" \
    --num_trials_per_task "${SMOKE_TRIALS}" \
    --num_steps_wait 10 \
    --post_success_settle_steps 100 \
    --seed "${EVAL_SEED}" \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --local_log_dir "${LOG_DIR}" \
    --save_video_mode all \
    --save_wrist_video True \
    --review_video_dir "${videos}" \
    --max_violation_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_success_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_failure_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --use_wandb False \
    --run_id_note "L3-B3-${RUN_TAG}-${condition}"
}

run_smoke() {
  "${PYTHON_BIN}" -c \
    'import sys; from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import verify_safe_reference_artifact; verify_safe_reference_artifact(sys.argv[1], sys.argv[2], expected_count=20); print("PASS_L3B3_SAFE_ARTIFACT_BOUND")' \
    "${SAFE_REPORT}" "${ER_STATES}"
  run_policy_views >/dev/null
  run_eval native
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b3_microwave_smoke.py" \
    --native "${TRAJECTORY_ROOT}/native" \
    --native-only \
    --expected-count "${SMOKE_TRIALS}" \
    --minimum-native-successes "${MIN_NATIVE_SUCCESSES}" \
    --output "${NATIVE_SMOKE_REPORT}"
  run_eval closed_microwave
  run_eval open_control
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b3_microwave_smoke.py" \
    --native "${TRAJECTORY_ROOT}/native" \
    --er "${TRAJECTORY_ROOT}/closed_microwave" \
    --ec "${TRAJECTORY_ROOT}/open_control" \
    --expected-count "${SMOKE_TRIALS}" \
    --minimum-native-successes "${MIN_NATIVE_SUCCESSES}" \
    --output "${PAIRED_SMOKE_REPORT}"
  echo "L3-B3 smoke artifacts complete; human video approval is still required."
}

run_pi05_eb_diagnostic() {
  if [[ "${SMOKE_TRIALS}" -gt 5 ]]; then
    echo "L3-B2 moved-cup pi0.5 diagnostic is limited to five episodes." >&2
    exit 2
  fi
  # Regenerate the exact 20-state native-only geometry in this worktree.  The
  # diagnostic consumes only Eb/native and remains unauthorized for formal use.
  run_prepare
  require_empty_output "${PI05_EB_DIAGNOSTIC_ROOT}"
  mkdir -p "${PI05_EB_DIAGNOSTIC_TRAJECTORIES}" "${PI05_EB_DIAGNOSTIC_VIDEOS}"
  "${PYTHON_BIN}" -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --model_family pi05 \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --pi05_host "${PI05_HOST:-127.0.0.1}" \
    --pi05_port "${PI05_PORT:-8000}" \
    --pi05_connect_timeout_s "${PI05_CONNECT_TIMEOUT_S:-300}" \
    --pi05_replan_steps "${PI05_REPLAN_STEPS:-5}" \
    --task_suite_name libero_10 \
    --task_ids 9 \
    --initial_states_path "${NATIVE_STATES}" \
    --native_only_preflight_manifest "${NATIVE_PREFLIGHT}" \
    --safety_oracle none \
    --held_object_body white_yellow_mug_1_main \
    --distractor_body porcelain_mug_1_main \
    --trajectory_track_bodies \
      "white_yellow_mug_1_main,porcelain_mug_1_main,microwave_1_main,microwave_1_microdoorroot" \
    --center_crop True \
    --num_images_in_input 2 \
    --save_trajectory True \
    --trajectory_dir "${PI05_EB_DIAGNOSTIC_TRAJECTORIES}" \
    --num_trials_per_task "${SMOKE_TRIALS}" \
    --num_steps_wait 10 \
    --post_success_settle_steps 100 \
    --seed "${EVAL_SEED}" \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --local_log_dir "${LOG_DIR}" \
    --save_video_mode all \
    --save_wrist_video True \
    --review_video_dir "${PI05_EB_DIAGNOSTIC_VIDEOS}" \
    --max_violation_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_success_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_failure_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --use_wandb False \
    --run_id_note "L3-B2-moved-cup-pi05-Eb-diagnostic"
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b2_moved_cup_pi05_eb.py" \
    --trajectory-dir "${PI05_EB_DIAGNOSTIC_TRAJECTORIES}" \
    --expected-count "${SMOKE_TRIALS}" \
    --output "${PI05_EB_DIAGNOSTIC_REPORT}"
  echo "L3-B2 moved-cup pi0.5 Eb diagnostic complete; formal remains unauthorized."
}

run_pi05_er_checkpoint_diagnostic() {
  if [[ "${SMOKE_TRIALS}" -ne 1 ]]; then
    echo "L3-B3 pi0.5 exact-Er checkpoint diagnostic requires exactly one episode." >&2
    exit 2
  fi
  # User-authorized checkpoint comparison only. Regenerate and bind all three
  # paired bundles, then consume exactly one closed-door Er state. No formal or
  # post-OpenVLA cascade authorization is granted by this path.
  run_prepare
  require_empty_output "${PI05_ER_DIAGNOSTIC_ROOT}"
  mkdir -p "${PI05_ER_DIAGNOSTIC_TRAJECTORIES}" "${PI05_ER_DIAGNOSTIC_VIDEOS}"
  "${PYTHON_BIN}" -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --model_family pi05 \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --pi05_host "${PI05_HOST:-127.0.0.1}" \
    --pi05_port "${PI05_PORT:-8000}" \
    --pi05_connect_timeout_s "${PI05_CONNECT_TIMEOUT_S:-300}" \
    --pi05_replan_steps "${PI05_REPLAN_STEPS:-5}" \
    --task_suite_name libero_10 \
    --task_ids 9 \
    --initial_states_path "${ER_STATES}" \
    --native_only_preflight_manifest "${NATIVE_PREFLIGHT}" \
    --safety_oracle none \
    --held_object_body white_yellow_mug_1_main \
    --distractor_body porcelain_mug_1_main \
    --trajectory_track_bodies \
      "white_yellow_mug_1_main,porcelain_mug_1_main,microwave_1_main,microwave_1_microdoorroot" \
    --center_crop True \
    --num_images_in_input 2 \
    --save_trajectory True \
    --trajectory_dir "${PI05_ER_DIAGNOSTIC_TRAJECTORIES}" \
    --num_trials_per_task "${SMOKE_TRIALS}" \
    --num_steps_wait 10 \
    --post_success_settle_steps 100 \
    --seed "${EVAL_SEED}" \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --local_log_dir "${LOG_DIR}" \
    --save_video_mode all \
    --save_wrist_video True \
    --review_video_dir "${PI05_ER_DIAGNOSTIC_VIDEOS}" \
    --max_violation_videos 1 \
    --max_success_videos 1 \
    --max_failure_videos 1 \
    --use_wandb False \
    --run_id_note "L3-B3-pi05-Er-checkpoint-diagnostic"
  "${PYTHON_BIN}" "${TASKS_DIR}/summarize_l3b3_pi05_er_checkpoint_diagnostic.py" \
    --trajectory-dir "${PI05_ER_DIAGNOSTIC_TRAJECTORIES}" \
    --expected-count 1 \
    --output "${PI05_ER_DIAGNOSTIC_REPORT}"
  echo "L3-B3 pi0.5 exact-Er checkpoint diagnostic complete; formal remains unauthorized."
}

case "${MODE}" in
  design) run_design ;;
  prepare) run_prepare ;;
  check) run_check ;;
  policy_views) run_policy_views ;;
  safe_reference) run_safe_reference ;;
  smoke) run_smoke ;;
  pi05_eb_diagnostic) run_pi05_eb_diagnostic ;;
  pi05_er_checkpoint_diagnostic) run_pi05_er_checkpoint_diagnostic ;;
  formal)
    echo "L3-B3 formal is fail-closed until Safe, OpenVLA-OFT smoke, and hash-bound human review pass." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 design|prepare|check|policy_views|safe_reference|smoke|pi05_eb_diagnostic|pi05_er_checkpoint_diagnostic|formal" >&2
    exit 2
    ;;
esac
