#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
MODE="${2:-}"
if [[ ! "${SCENARIO}" =~ ^l1c[234]$ ]] || [[ -z "${MODE}" ]]; then
  echo "Usage: $0 l1c2|l1c3|l1c4 check|preview|calibrate|safe_reference|eb|er|ec|replay|smoke|analyze|eval" >&2
  exit 2
fi

case "${SCENARIO}" in
  l1c2) SLUG="occupied-basket" ;;
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
PREVIEW_DIR="${PREVIEW_DIR:-${STATE_DIR}/${SCENARIO}_preview}"

NUM_TRIALS="${NUM_TRIALS:-50}"
SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
CALIBRATION_NUM_STATES="${CALIBRATION_NUM_STATES:-8}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-90}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-violation}"
MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-10}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"

EB_NOTE="${UPPER_SCENARIO}-${SLUG}-eb"
ER_NOTE="${UPPER_SCENARIO}-${SLUG}-risk"
EC_NOTE="${UPPER_SCENARIO}-${SLUG}-ec"
EB_TRAJ="rollouts/libero_90/${EB_NOTE}/trajectories"
ER_TRAJ="rollouts/libero_90/${ER_NOTE}/trajectories"
EC_TRAJ="rollouts/libero_90/${EC_NOTE}/trajectories"

CALIBRATION_CSV="${LOG_DIR}/${SCENARIO}_calibration.csv"
CALIBRATION_REPORT="${LOG_DIR}/${SCENARIO}_calibration.md"
SAFE_REFERENCE_CSV="${LOG_DIR}/${SCENARIO}_safe_reference.csv"
SAFE_REFERENCE_REPORT="${LOG_DIR}/${SCENARIO}_safe_reference.md"
ER_REPLAY_CSV="${LOG_DIR}/${SCENARIO}_eb_to_er_replay.csv"
ER_REPLAY_REPORT="${LOG_DIR}/${SCENARIO}_eb_to_er_replay.md"
EC_REPLAY_CSV="${LOG_DIR}/${SCENARIO}_eb_to_ec_replay.csv"
EC_REPLAY_REPORT="${LOG_DIR}/${SCENARIO}_eb_to_ec_replay.md"
ATTRIBUTION_CSV="${LOG_DIR}/${SCENARIO}_attribution.csv"
ATTRIBUTION_REPORT="${LOG_DIR}/${SCENARIO}_attribution.md"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

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
    --source_indices "${SOURCE_INDICES}" --num_states "${n}"
}

run_preview() {
  python "${PIPELINE}" preview "${common_state_args[@]}" \
    --out_dir "${PREVIEW_DIR}" --num_states "${PREVIEW_NUM_STATES:-3}"
}

run_calibrate() {
  python "${PIPELINE}" calibrate "${common_state_args[@]}" \
    --num_states "${CALIBRATION_NUM_STATES}" \
    --out_csv "${CALIBRATION_CSV}" --out_report "${CALIBRATION_REPORT}"
}

run_safe_reference() {
  python "${PIPELINE}" safe-reference "${common_state_args[@]}" \
    --num_states "${CALIBRATION_NUM_STATES}" \
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
  mkdir -p "${trajectory_dir}"
  find "${trajectory_dir}" -maxdepth 1 -type f \( -name '*.npz' -o -name 'index.jsonl' \) -delete
  local bddl
  bddl="$(resolve_bddl)"
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
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

case "${MODE}" in
  check) run_check ;;
  preview) run_preview ;;
  calibrate) run_calibrate ;;
  safe_reference) run_safe_reference ;;
  eb|er|ec) run_condition "${MODE}" "${NUM_TRIALS}" ;;
  replay) run_replay ;;
  smoke)
    run_check "${SMOKE_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    run_condition eb "${SMOKE_TRIALS}"
    run_condition er "${SMOKE_TRIALS}"
    run_condition ec "${SMOKE_TRIALS}"
    run_replay
    run_analyze
    ;;
  analyze) run_analyze ;;
  eval)
    run_check "${NUM_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    run_condition eb "${NUM_TRIALS}"
    run_condition er "${NUM_TRIALS}"
    run_condition ec "${NUM_TRIALS}"
    run_replay
    run_analyze
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
