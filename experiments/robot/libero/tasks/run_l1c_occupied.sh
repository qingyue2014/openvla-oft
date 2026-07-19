#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
MODE="${2:-}"
if [[ ! "${SCENARIO}" =~ ^l1c[234]$ ]] || [[ -z "${MODE}" ]]; then
  echo "Usage: $0 l1c2|l1c3|l1c4 bodies|check|preview|verify|validate_layout|screen_occupants|calibrate|competence|policy_probe|safe_reference|eb|er|ec|replay|smoke|analyze|record|eval" >&2
  exit 2
fi

case "${SCENARIO}" in
  l1c2) SLUG="occupied-tray" ;;
  l1c3) SLUG="occupied-drawer" ;;
  l1c4) SLUG="occupied-cabinet-top" ;;
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

NUM_TRIALS="${NUM_TRIALS:-50}"
SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
CALIBRATION_NUM_STATES="${CALIBRATION_NUM_STATES:-8}"
if [[ "${SCENARIO}" == "l1c2" ]]; then
  # The deterministic LIBERO-90 SFT policy reached only 7/50 on this native
  # task, while the GRPO LIBERO-90 checkpoint reproducibly reached 39/50 and
  # missed the fixed 80% competence gate. Use RLinf's official RL-trained
  # all-task LIBERO-130 checkpoint with the same published sampling settings;
  # keep the seed identical across Eb/Er/Ec for paired attribution.
  DEFAULT_CHECKPOINT="RLinf/RLinf-OpenVLAOFT-LIBERO-130"
  DEFAULT_DO_SAMPLE="true"
  DEFAULT_TEMPERATURE="1.6"
else
  DEFAULT_CHECKPOINT="RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"
  DEFAULT_DO_SAMPLE="false"
  DEFAULT_TEMPERATURE="1.0"
fi
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
DO_SAMPLE="${DO_SAMPLE:-${DEFAULT_DO_SAMPLE}}"
TEMPERATURE="${TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
TOP_P="${TOP_P:-1.0}"
MODEL_SEED="${MODEL_SEED:-7}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-violation}"
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

EB_NOTE="${UPPER_SCENARIO}-${SLUG}-eb"
ER_NOTE="${UPPER_SCENARIO}-${SLUG}-risk"
EC_NOTE="${UPPER_SCENARIO}-${SLUG}-ec"
EB_TRAJ="rollouts/libero_90/${EB_NOTE}/trajectories"
ER_TRAJ="rollouts/libero_90/${ER_NOTE}/trajectories"
EC_TRAJ="rollouts/libero_90/${EC_NOTE}/trajectories"

CALIBRATION_CSV="${LOG_DIR}/${SCENARIO}_calibration.csv"
CALIBRATION_REPORT="${LOG_DIR}/${SCENARIO}_calibration.md"
SAFE_REFERENCE_CSV="${LOG_DIR}/${SCENARIO}_safe_reference.csv"
SAFE_REFERENCE_ATTEMPTS_CSV="${LOG_DIR}/${SCENARIO}_safe_reference_attempts.csv"
SAFE_REFERENCE_REPORT="${LOG_DIR}/${SCENARIO}_safe_reference.md"
SAFE_REFERENCE_TRAJ="${LOG_DIR}/${SCENARIO}_safe_reference_trajectories"
EB_COMPETENCE_CSV="${LOG_DIR}/${SCENARIO}_eb_competence.csv"
EB_COMPETENCE_REPORT="${LOG_DIR}/${SCENARIO}_eb_competence.md"
ER_REPLAY_CSV="${LOG_DIR}/${SCENARIO}_eb_to_er_replay.csv"
ER_REPLAY_REPORT="${LOG_DIR}/${SCENARIO}_eb_to_er_replay.md"
EC_REPLAY_CSV="${LOG_DIR}/${SCENARIO}_eb_to_ec_replay.csv"
EC_REPLAY_REPORT="${LOG_DIR}/${SCENARIO}_eb_to_ec_replay.md"
ATTRIBUTION_CSV="${LOG_DIR}/${SCENARIO}_attribution.csv"
ATTRIBUTION_REPORT="${LOG_DIR}/${SCENARIO}_attribution.md"
PREVIEW_CSV="${LOG_DIR}/${SCENARIO}_exact_state_preview.csv"
PREVIEW_REPORT="${LOG_DIR}/${SCENARIO}_exact_state_preview.md"

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
    --out_csv "${PREVIEW_CSV}" --out_report "${PREVIEW_REPORT}"
}

run_verify() {
  python "${PIPELINE}" verify "${common_state_args[@]}" \
    --source_indices "${SOURCE_INDICES}" \
    --bundle_manifest "${STATE_BUNDLE_MANIFEST}" \
    --preview_manifest "${PREVIEW_MANIFEST}" --min_states "${1:-${NUM_TRIALS}}"
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
  rm -f "${SAFE_REFERENCE_CSV}" "${SAFE_REFERENCE_ATTEMPTS_CSV}" \
    "${SAFE_REFERENCE_REPORT}"
  mkdir -p "${SAFE_REFERENCE_TRAJ}"
  find "${SAFE_REFERENCE_TRAJ}" -maxdepth 1 -type f -name '*.npz' -delete
  python "${PIPELINE}" safe-reference "${common_state_args[@]}" \
    --num_states "${CALIBRATION_NUM_STATES}" \
    --max_attempts_per_state "${SAFE_REFERENCE_MAX_ATTEMPTS:-0}" \
    --eb_trajectories "${EB_TRAJ}" --trajectory_dir "${SAFE_REFERENCE_TRAJ}" \
    --out_csv "${SAFE_REFERENCE_CSV}" --out_report "${SAFE_REFERENCE_REPORT}"
}

run_competence() {
  python "${PIPELINE}" competence --scenario "${SCENARIO}" \
    --trajectories "${EB_TRAJ}" --min_episodes "${1:-${NUM_TRIALS}}" \
    --out_csv "${EB_COMPETENCE_CSV}" --out_report "${EB_COMPETENCE_REPORT}"
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
  mkdir -p "${trajectory_dir}"
  find "${trajectory_dir}" -maxdepth 1 -type f \( -name '*.npz' -o -name 'index.jsonl' \) -delete
  local bddl
  bddl="$(resolve_bddl)"
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --do_sample "${DO_SAMPLE}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${MODEL_SEED}" \
    --task_suite_name libero_90 \
    --bddl_file "${bddl}" \
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
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --run_id_note "${note}"
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
  python experiments/robot/libero/tasks/generate_result_tables.py --records "${LOG_DIR}/experiment_records.csv"
}

run_record() {
  python experiments/robot/libero/tasks/record_experiment_results.py --log_dir "${LOG_DIR}"
  python experiments/robot/libero/tasks/generate_result_tables.py --records "${LOG_DIR}/experiment_records.csv"
}

case "${MODE}" in
  bodies) run_bodies ;;
  check) run_check ;;
  preview) run_preview ;;
  verify) run_verify ;;
  validate_layout)
    run_verify "${NUM_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    ;;
  screen_occupants) run_screen_occupants ;;
  calibrate) run_calibrate ;;
  competence) run_competence ;;
  policy_probe)
    run_verify "${NUM_TRIALS}"
    run_condition eb "${NUM_TRIALS}"
    run_competence "${NUM_TRIALS}"
    grep -q 'PASS_EB_COMPETENCE' "${EB_COMPETENCE_REPORT}"
    ;;
  safe_reference) run_safe_reference ;;
  eb|er|ec) run_condition "${MODE}" "${NUM_TRIALS}" ;;
  replay) run_replay ;;
  smoke)
    run_check "${SMOKE_TRIALS}"
    PREVIEW_NUM_STATES="${SMOKE_TRIALS}" run_preview
    grep -q 'PASS_EXACT_STATE_PREVIEW' "${PREVIEW_REPORT}"
    run_verify "${SMOKE_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_condition eb "${SMOKE_TRIALS}"
    run_competence "${SMOKE_TRIALS}"
    grep -q 'PASS_EB_COMPETENCE' "${EB_COMPETENCE_REPORT}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    run_replay
    grep -q 'PASS_ACTION_SEPARATION' "${ER_REPLAY_REPORT}"
    grep -q 'PASS_EC_UNCHANGED_EB_REPLAY_SAFE' "${EC_REPLAY_REPORT}"
    run_condition er "${SMOKE_TRIALS}"
    run_condition ec "${SMOKE_TRIALS}"
    run_analyze
    ;;
  analyze) run_analyze ;;
  record) run_record ;;
  eval)
    run_verify "${NUM_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_condition eb "${NUM_TRIALS}"
    run_competence "${NUM_TRIALS}"
    grep -q 'PASS_EB_COMPETENCE' "${EB_COMPETENCE_REPORT}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    run_replay
    grep -q 'PASS_ACTION_SEPARATION' "${ER_REPLAY_REPORT}"
    grep -q 'PASS_EC_UNCHANGED_EB_REPLAY_SAFE' "${EC_REPLAY_REPORT}"
    run_condition er "${NUM_TRIALS}"
    run_condition ec "${NUM_TRIALS}"
    run_analyze
    grep -q 'BENCHMARK_READY_FOR_ATTRIBUTION' "${ATTRIBUTION_REPORT}"
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
