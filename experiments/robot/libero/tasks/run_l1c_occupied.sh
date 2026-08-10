#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
MODE="${2:-}"
if [[ ! "${SCENARIO}" =~ ^l1c[234]$ ]] || [[ -z "${MODE}" ]]; then
  echo "Usage: $0 l1c2|l1c3|l1c4 bodies|check|native_preflight|preview|verify|screen_occupants|calibrate|safe_reference|eb|er|ec|replay|smoke|analyze|record|eval" >&2
  exit 2
fi

case "${SCENARIO}" in
  l1c2) SLUG="occupied-tray" ;;
  l1c3) SLUG="occupied-drawer" ;;
  l1c4) SLUG="occupied-basket" ;;
esac
UPPER_SCENARIO="$(printf '%s' "${SCENARIO}" | tr '[:lower:]' '[:upper:]' | sed 's/C/-C/')"

PIPELINE="experiments/robot/libero/tasks/l1c_occupied_pipeline.py"
STATE_DIR="${STATE_DIR:-experiments/robot/libero/tasks}"
LOG_DIR="${LOG_DIR:-experiments/logs}"
EB_STATES="${EB_STATES:-${STATE_DIR}/${SCENARIO}_eb_states.hdf5}"
ER_STATES="${ER_STATES:-${STATE_DIR}/${SCENARIO}_er_states.hdf5}"
EC_STATES="${EC_STATES:-${STATE_DIR}/${SCENARIO}_ec_states.hdf5}"
SOURCE_INDICES="${SOURCE_INDICES:-${STATE_DIR}/${SCENARIO}_source_indices.json}"
STATE_BUNDLE_MANIFEST="${STATE_BUNDLE_MANIFEST:-${STATE_DIR}/${SCENARIO}_state_bundle.json}"
PREVIEW_DIR="${PREVIEW_DIR:-${STATE_DIR}/${SCENARIO}_preview}"
PREVIEW_MANIFEST="${PREVIEW_MANIFEST:-${PREVIEW_DIR}/manifest.json}"
REVIEW_DIR="${REVIEW_DIR:-review/${UPPER_SCENARIO}_task}"

NUM_TRIALS="${NUM_TRIALS:-50}"
SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
CALIBRATION_NUM_STATES="${CALIBRATION_NUM_STATES:-8}"
if [[ "${SCENARIO}" == "l1c4" ]]; then
  DEFAULT_CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-object"
  DEFAULT_SAFE_REFERENCE_GRASP_DEPTH="0.040"
else
  DEFAULT_CHECKPOINT="RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"
  DEFAULT_SAFE_REFERENCE_GRASP_DEPTH="0.025"
fi
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
MODEL_FAMILY="${MODEL_FAMILY:-openvla}"
case "${MODEL_FAMILY}" in
  openvla|pi05|cosmos) ;;
  *)
    echo "Unsupported L1-C model family ${MODEL_FAMILY}; expected openvla, pi05, or cosmos." >&2
    exit 2
    ;;
esac
PI05_HOST="${PI05_HOST:-127.0.0.1}"
PI05_PORT="${PI05_PORT:-8000}"
PI05_REPLAN_STEPS="${PI05_REPLAN_STEPS:-5}"
PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-900}"
MODEL_OPEN_LOOP_STEPS="${MODEL_OPEN_LOOP_STEPS:-$([[ "${MODEL_FAMILY}" == "cosmos" ]] && printf 16 || printf 8)}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-10}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"

# Match the established PhysCog runners: most server checkouts keep LIBERO as
# a sibling of openvla-oft (~/04-mycode/LIBERO).  LIBERO_ROOT may also point
# directly at that repository when a different layout is used.
LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" ]]; then
  if [[ -d "../LIBERO/libero" ]]; then
    LIBERO_ROOT="$(cd ../LIBERO && pwd)"
  elif [[ -d "../libero/libero" ]]; then
    LIBERO_ROOT="$(cd ../libero && pwd)"
  elif [[ -d "_deps/LIBERO/libero" ]]; then
    LIBERO_ROOT="$(cd _deps/LIBERO && pwd)"
  fi
fi
if [[ -z "${LIBERO_ROOT}" || ! -d "${LIBERO_ROOT}/libero" ]]; then
  echo "Could not locate LIBERO. Set LIBERO_ROOT to the LIBERO repository root." >&2
  exit 2
fi
export LIBERO_ROOT
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  LIBERO_CONFIG_PATH="$(python -c "from experiments.robot.libero.tasks.l1c_occupied_pipeline import _ensure_libero_config; print(_ensure_libero_config())")"
fi
if [[ -z "${LIBERO_CONFIG_PATH}" || ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  echo "Could not create a task-local LIBERO config for ${LIBERO_ROOT}." >&2
  exit 2
fi
export LIBERO_CONFIG_PATH

NATIVE_SUITE="$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').native_suite)")"
NATIVE_TASK_ID="$(python -c "from experiments.robot.libero.tasks.l1c_occupied_pipeline import _native_task_match; from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(_native_task_match(get_spec('${SCENARIO}'))[1])" | tail -n 1)"
if [[ "${SCENARIO}" == "l1c4" && ! "${NATIVE_SUITE}" =~ ^libero_(spatial|object|goal|10)$ ]]; then
  echo "L1-C4 must use one of libero_spatial/libero_object/libero_goal/libero_10; got ${NATIVE_SUITE}." >&2
  exit 2
fi

RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-}"
NOTE_SUFFIX="${RUN_ID_SUFFIX:+-${RUN_ID_SUFFIX}}"
EB_NOTE="${UPPER_SCENARIO}-${SLUG}-eb${NOTE_SUFFIX}"
ER_NOTE="${UPPER_SCENARIO}-${SLUG}-risk${NOTE_SUFFIX}"
EC_NOTE="${UPPER_SCENARIO}-${SLUG}-ec${NOTE_SUFFIX}"
EB_TRAJ="rollouts/${NATIVE_SUITE}/${EB_NOTE}/trajectories"
ER_TRAJ="rollouts/${NATIVE_SUITE}/${ER_NOTE}/trajectories"
EC_TRAJ="rollouts/${NATIVE_SUITE}/${EC_NOTE}/trajectories"

CALIBRATION_CSV="${LOG_DIR}/${SCENARIO}_calibration.csv"
CALIBRATION_REPORT="${LOG_DIR}/${SCENARIO}_calibration.md"
SAFE_REFERENCE_CSV="${LOG_DIR}/${SCENARIO}_safe_reference.csv"
SAFE_REFERENCE_REPORT="${LOG_DIR}/${SCENARIO}_safe_reference.md"
SAFE_REFERENCE_TRAJ="${LOG_DIR}/${SCENARIO}_safe_reference_trajectories"
SAFE_REFERENCE_VIDEOS="${REVIEW_DIR}/safe_reference"
ER_REPLAY_CSV="${LOG_DIR}/${SCENARIO}_eb_to_er_replay.csv"
ER_REPLAY_REPORT="${LOG_DIR}/${SCENARIO}_eb_to_er_replay.md"
EC_REPLAY_CSV="${LOG_DIR}/${SCENARIO}_eb_to_ec_replay.csv"
EC_REPLAY_REPORT="${LOG_DIR}/${SCENARIO}_eb_to_ec_replay.md"
ATTRIBUTION_CSV="${LOG_DIR}/${SCENARIO}_attribution.csv"
ATTRIBUTION_REPORT="${LOG_DIR}/${SCENARIO}_attribution.md"
PREVIEW_CSV="${LOG_DIR}/${SCENARIO}_exact_state_preview.csv"
PREVIEW_REPORT="${LOG_DIR}/${SCENARIO}_exact_state_preview.md"
NATIVE_PREFLIGHT_JSON="${LOG_DIR}/${SCENARIO}_native_preflight.json"
NATIVE_PREFLIGHT_REPORT="${LOG_DIR}/${SCENARIO}_native_preflight.md"
HUMAN_VISIBILITY_REVIEW="${HUMAN_VISIBILITY_REVIEW:-${REVIEW_DIR}/visibility_review.md}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ "${RENDER_GPU_DEVICE_ID}" != "-1" ]]; then
  # robosuite versions differ in which EGL selector they honor. Keep both in
  # sync with the explicit render_gpu_device_id passed to OffScreenRenderEnv.
  export EGL_DEVICE_ID="${EGL_DEVICE_ID:-${RENDER_GPU_DEVICE_ID}}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${RENDER_GPU_DEVICE_ID}}"
fi

common_state_args=(
  --scenario "${SCENARIO}"
  --eb_states "${EB_STATES}"
  --er_states "${ER_STATES}"
  --ec_states "${EC_STATES}"
)

resolve_bddl() {
  python "${PIPELINE}" resolve-bddl --scenario "${SCENARIO}"
}

run_check() {
  local n="${1:-${NUM_TRIALS}}"
  python "${PIPELINE}" generate "${common_state_args[@]}" \
    --source_indices "${SOURCE_INDICES}" \
    --bundle_manifest "${STATE_BUNDLE_MANIFEST}" --num_states "${n}"
  run_native_preflight
}

run_native_preflight() {
  python "${PIPELINE}" native-preflight "${common_state_args[@]}" \
    --out_json "${NATIVE_PREFLIGHT_JSON}" \
    --out_report "${NATIVE_PREFLIGHT_REPORT}"
  grep -q 'PASS_NATIVE_ONLY_PREFLIGHT' "${NATIVE_PREFLIGHT_REPORT}"
}

run_bodies() {
  python "${PIPELINE}" list-bodies --scenario "${SCENARIO}"
}

run_preview() {
  python "${PIPELINE}" preview "${common_state_args[@]}" \
    --source_indices "${SOURCE_INDICES}" \
    --bundle_manifest "${STATE_BUNDLE_MANIFEST}" \
    --preview_manifest "${PREVIEW_MANIFEST}" \
    --out_dir "${PREVIEW_DIR}" --num_states "${PREVIEW_NUM_STATES:-3}" \
    --policy_model_family "${POLICY_MODEL_FAMILY:-openvla}" \
    --out_csv "${PREVIEW_CSV}" --out_report "${PREVIEW_REPORT}"
}

run_verify() {
  python "${PIPELINE}" verify "${common_state_args[@]}" \
    --source_indices "${SOURCE_INDICES}" \
    --bundle_manifest "${STATE_BUNDLE_MANIFEST}" \
    --preview_manifest "${PREVIEW_MANIFEST}" \
    --min_states "${1:-${NUM_TRIALS}}"
}

require_human_visibility_review() {
  if [[ ! -f "${HUMAN_VISIBILITY_REVIEW}" ]] || \
     ! grep -q 'PASS_HUMAN_VISIBILITY' "${HUMAN_VISIBILITY_REVIEW}"; then
    echo "Formal evaluation requires a manual policy-view verdict at ${HUMAN_VISIBILITY_REVIEW} containing PASS_HUMAN_VISIBILITY." >&2
    exit 1
  fi
}

run_screen_occupants() {
  python "${PIPELINE}" screen-occupants "${common_state_args[@]}"
}

run_calibrate() {
  python "${PIPELINE}" calibrate "${common_state_args[@]}" \
    --num_states "${CALIBRATION_NUM_STATES}" \
    --out_csv "${CALIBRATION_CSV}" --out_report "${CALIBRATION_REPORT}"
}

run_safe_reference() {
  # Do not leave a stale report that can be mistaken for the current scene if
  # the prerequisite Eb-trajectory check exits before writing new results.
  rm -f "${SAFE_REFERENCE_CSV}" "${SAFE_REFERENCE_REPORT}"
  mkdir -p "${SAFE_REFERENCE_TRAJ}" "${SAFE_REFERENCE_VIDEOS}"
  find "${SAFE_REFERENCE_TRAJ}" -maxdepth 1 -type f -name '*.npz' -delete
  find "${SAFE_REFERENCE_VIDEOS}" -maxdepth 1 -type f -name '*.mp4' -delete
  python "${PIPELINE}" safe-reference "${common_state_args[@]}" \
    --num_states "${CALIBRATION_NUM_STATES}" \
    --max_attempts_per_state "${SAFE_REFERENCE_MAX_ATTEMPTS:-0}" \
    --grasp_depth "${SAFE_REFERENCE_GRASP_DEPTH:-${DEFAULT_SAFE_REFERENCE_GRASP_DEPTH}}" \
    --eb_trajectories "${EB_TRAJ}" --trajectory_dir "${SAFE_REFERENCE_TRAJ}" \
    --video_dir "${SAFE_REFERENCE_VIDEOS}" \
    --out_csv "${SAFE_REFERENCE_CSV}" --out_report "${SAFE_REFERENCE_REPORT}"
}

run_condition() {
  local condition="$1"
  local trials="$2"
  local state_path note oracle trajectory_dir
  case "${condition}" in
    eb) state_path="${EB_STATES}"; note="${EB_NOTE}"; oracle="none"; trajectory_dir="${EB_TRAJ}" ;;
    er) state_path="${ER_STATES}"; note="${ER_NOTE}"; oracle="occupied_goal"; trajectory_dir="${ER_TRAJ}" ;;
    ec) state_path="${EC_STATES}"; note="${EC_NOTE}"; oracle="none"; trajectory_dir="${EC_TRAJ}" ;;
  esac
  local condition_review_dir="${REVIEW_DIR}/${condition}"
  local model_args=(
    --model_family "${MODEL_FAMILY}"
    --pretrained_checkpoint "${CHECKPOINT}"
    --num_open_loop_steps "${MODEL_OPEN_LOOP_STEPS}"
  )
  if [[ "${MODEL_FAMILY}" == "pi05" ]]; then
    model_args+=(
      --pi05_host "${PI05_HOST}"
      --pi05_port "${PI05_PORT}"
      --pi05_replan_steps "${PI05_REPLAN_STEPS}"
      --pi05_connect_timeout_s "${PI05_CONNECT_TIMEOUT_S}"
    )
  fi
  mkdir -p "${trajectory_dir}" "${condition_review_dir}"
  find "${trajectory_dir}" -maxdepth 1 -type f \( -name '*.npz' -o -name 'index.jsonl' \) -delete
  find "${condition_review_dir}" -maxdepth 1 -type f -name '*.mp4' -delete
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    "${model_args[@]}" \
    --task_suite_name "${NATIVE_SUITE}" \
    --task_ids "${NATIVE_TASK_ID}" \
    --native_only_preflight_manifest "${NATIVE_PREFLIGHT_JSON}" \
    --initial_states_path "${state_path}" \
    --num_trials_per_task "${trials}" \
    --safety_oracle "${oracle}" \
    --held_object_body "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').target_body)")" \
    --distractor_body "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').occupant_body)")" \
    --occupancy_support_body "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').anchor_body)")" \
    --occupancy_max_displacement "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_occupant_displacement)")" \
    --occupancy_max_tilt_change_deg "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_occupant_tilt_change_deg)")" \
    --occupancy_min_target_clearance "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').min_target_clearance)")" \
    --occupancy_min_target_tilt_deg "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').min_target_tilt_deg)")" \
    --occupancy_max_target_tilt_deg "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_target_tilt_deg)")" \
    --occupancy_max_target_post_release_xy_displacement "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_target_post_release_xy_displacement)")" \
    --trajectory_track_bodies "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').anchor_body)")" \
    --trajectory_dir "${trajectory_dir}" \
    --post_success_settle_steps 60 \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --max_violation_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_success_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --max_failure_videos "${MAX_VIDEOS_PER_OUTCOME}" \
    --review_video_dir "${condition_review_dir}" \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --num_steps_wait 10 \
    --seed "${SEED:-7}" \
    --run_id_note "${note}"
  local video
  for video in "${condition_review_dir}"/*.mp4; do
    [[ -e "${video}" ]] || continue
    if [[ "$(basename "${video}")" != "${UPPER_SCENARIO}-${condition}-"* ]]; then
      mv "${video}" "${condition_review_dir}/${UPPER_SCENARIO}-${condition}-$(basename "${video}")"
    fi
  done
}

run_replay() {
  python "${PIPELINE}" replay "${common_state_args[@]}" \
    --condition er --eb_trajectories "${EB_TRAJ}" \
    --out_csv "${ER_REPLAY_CSV}" --out_report "${ER_REPLAY_REPORT}"
  python "${PIPELINE}" replay "${common_state_args[@]}" \
    --condition ec --eb_trajectories "${EB_TRAJ}" \
    --out_csv "${EC_REPLAY_CSV}" --out_report "${EC_REPLAY_REPORT}"
}

run_analyze() {
  python "${PIPELINE}" analyze --scenario "${SCENARIO}" \
    --eb "${EB_TRAJ}" --er "${ER_TRAJ}" --ec "${EC_TRAJ}" \
    --er_replay_csv "${ER_REPLAY_CSV}" \
    --ec_replay_csv "${EC_REPLAY_CSV}" \
    --safe_reference_csv "${SAFE_REFERENCE_CSV}" \
    --out_csv "${ATTRIBUTION_CSV}" --out_report "${ATTRIBUTION_REPORT}"
  python experiments/robot/libero/tasks/record_experiment_results.py --log_dir "${LOG_DIR}"
  python experiments/robot/libero/tasks/generate_result_tables.py --log_dir "${LOG_DIR}"
}

run_record() {
  python experiments/robot/libero/tasks/record_experiment_results.py --log_dir "${LOG_DIR}"
  python experiments/robot/libero/tasks/generate_result_tables.py --log_dir "${LOG_DIR}"
}

case "${MODE}" in
  bodies) run_bodies ;;
  check) run_check ;;
  native_preflight) run_native_preflight ;;
  preview) run_native_preflight; run_preview ;;
  verify) run_native_preflight; run_verify ;;
  screen_occupants) run_native_preflight; run_screen_occupants ;;
  calibrate) run_native_preflight; run_calibrate ;;
  safe_reference) run_native_preflight; run_safe_reference ;;
  eb|er|ec) run_native_preflight; run_verify; run_condition "${MODE}" "${NUM_TRIALS}" ;;
  replay) run_native_preflight; run_replay ;;
  smoke)
    run_check "${SMOKE_TRIALS}"
    PREVIEW_NUM_STATES="${SMOKE_TRIALS}" run_preview
    grep -q 'PASS_EXACT_STATE_PREVIEW' "${PREVIEW_REPORT}"
    run_verify "${SMOKE_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_condition eb "${SMOKE_TRIALS}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    run_condition er "${SMOKE_TRIALS}"
    run_condition ec "${SMOKE_TRIALS}"
    run_replay
    run_analyze
    ;;
  analyze) run_analyze ;;
  record) run_record ;;
  eval)
    run_check "${NUM_TRIALS}"
    PREVIEW_NUM_STATES="${PREVIEW_NUM_STATES:-3}" run_preview
    grep -q 'PASS_EXACT_STATE_PREVIEW' "${PREVIEW_REPORT}"
    run_verify "${NUM_TRIALS}"
    require_human_visibility_review
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_condition eb "${NUM_TRIALS}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    run_condition er "${NUM_TRIALS}"
    run_condition ec "${NUM_TRIALS}"
    run_replay
    run_analyze
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
