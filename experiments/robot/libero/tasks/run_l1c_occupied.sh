#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
MODE="${2:-}"
if [[ ! "${SCENARIO}" =~ ^l1c[2345]$ ]] || [[ -z "${MODE}" ]]; then
  echo "Usage: $0 l1c2|l1c3|l1c4|l1c5 bodies|check|native_preflight|preview|verify|screen_occupants|calibrate|safe_reference|construct|eb|er|ec|replay|smoke|analyze|record|eval" >&2
  exit 2
fi

case "${SCENARIO}" in
  l1c2) SLUG="occupied-tray" ;;
  l1c3) SLUG="occupied-drawer" ;;
  l1c4) SLUG="occupied-basket" ;;
  l1c5) SLUG="orange-juice-occupied-basket" ;;
esac
UPPER_SCENARIO="$(printf '%s' "${SCENARIO}" | tr '[:lower:]' '[:upper:]' | sed 's/C/-C/')"

PIPELINE="experiments/robot/libero/tasks/l1c_occupied_pipeline.py"
L1C5_FREEZE_VERIFIER="experiments/robot/libero/tasks/verify_l1c5_frozen_gate.py"
L1C5_FROZEN_GATE_SHA256="14eeb148208f536eca7920ddde28b285502007d6b719739f6742712264cd5937"
L1C5_FROZEN_GATE_VARIANT="${L1C5_FROZEN_GATE_VARIANT:-original}"
case "${L1C5_FROZEN_GATE_VARIANT}" in
  original) ;;
  model-informed-v1)
    L1C5_FREEZE_VERIFIER="experiments/robot/libero/tasks/verify_l1c5_mi_v1_frozen_gate.py"
    L1C5_FROZEN_GATE_SHA256="188618a62db589895b8e3f6c07e9128073a14066421f10321c844127962e946c"
    ;;
  *)
    echo "Unsupported L1-C5 frozen-gate variant: ${L1C5_FROZEN_GATE_VARIANT}" >&2
    exit 2
    ;;
esac
L1C5_EC_AMENDMENT="experiments/robot/libero/tasks/l1c5_ec_replay_amendment_20260811.json"
L1C5_EC_AMENDMENT_SHA256="bb8beab2c573635742e2bdc2357962596e9873cf64f7c2fed1f968e9f698ded0"
L1C5_UPRIGHT_AMENDMENT="experiments/robot/libero/tasks/l1c5_posthoc_upright_oracle_amendment_20260811.json"
L1C5_UPRIGHT_AMENDMENT_SHA256="5f8afdf49032ff4f2aff65c9a469b5445b9ace2be148b2ab0c0421e017a30ad9"
L1C5_EC_MATCH_MIN_RATE="${L1C5_EC_MATCH_MIN_RATE:-1.0}"
L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT="${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT:-}"
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
if [[ "${SCENARIO}" == "l1c4" || "${SCENARIO}" == "l1c5" ]]; then
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
COSMOS_HOST="${COSMOS_HOST:-127.0.0.1}"
COSMOS_PORT="${COSMOS_PORT:-0}"
COSMOS_CONNECT_TIMEOUT_S="${COSMOS_CONNECT_TIMEOUT_S:-900}"
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
if [[ ( "${SCENARIO}" == "l1c4" || "${SCENARIO}" == "l1c5" ) && ! "${NATIVE_SUITE}" =~ ^libero_(spatial|object|goal|10)$ ]]; then
  echo "${UPPER_SCENARIO} must use one of libero_spatial/libero_object/libero_goal/libero_10; got ${NATIVE_SUITE}." >&2
  exit 2
fi
if [[ "${SCENARIO}" == "l1c5" && "${L1C5_EC_MATCH_MIN_RATE}" != "1.0" ]]; then
  if [[ "${L1C5_EC_MATCH_MIN_RATE}" != "0.98" ]]; then
    echo "L1-C5 EC replay relaxation requires the explicit post-hoc 98% amendment unlock." >&2
    exit 2
  fi
  if [[ "${L1C5_FROZEN_GATE_VARIANT}" == "model-informed-v1" ]]; then
    python "${L1C5_FREEZE_VERIFIER}"
  elif [[ "${L1C5_EC_AMENDMENT_UNLOCK:-}" == "I_ACKNOWLEDGE_POSTHOC_98_PERCENT" ]]; then
    test -s "${L1C5_EC_AMENDMENT}"
    observed_amendment_sha="$(sha256sum "${L1C5_EC_AMENDMENT}" | awk '{print $1}')"
    if [[ "${observed_amendment_sha}" != "${L1C5_EC_AMENDMENT_SHA256}" ]]; then
      echo "L1-C5 EC replay amendment hash mismatch." >&2
      exit 2
    fi
    grep -Fq 'POSTHOC_AMENDED_98_PERCENT_NOT_ORIGINAL_PREREGISTRATION' "${L1C5_EC_AMENDMENT}"
  elif [[ "${L1C5_POSTHOC_ORACLE_UNLOCK:-}" != "I_ACKNOWLEDGE_POSTHOC_NO_POST_RELEASE_XY_LIMIT" ]]; then
    echo "L1-C5 EC replay relaxation requires a hash-bound post-hoc amendment." >&2
    exit 2
  fi
fi
if [[ -n "${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}" || \
      -n "${L1C5_POSTHOC_ORACLE_UNLOCK:-}" ]]; then
  if [[ "${SCENARIO}" == "l1c5" && \
        "${L1C5_FROZEN_GATE_VARIANT}" == "model-informed-v1" && \
        "${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}" == "inf" && \
        -z "${L1C5_POSTHOC_ORACLE_UNLOCK:-}" ]]; then
    python "${L1C5_FREEZE_VERIFIER}"
  elif [[ "${SCENARIO}" != "l1c5" || \
        "${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}" != "inf" || \
        "${L1C5_POSTHOC_ORACLE_UNLOCK:-}" != "I_ACKNOWLEDGE_POSTHOC_NO_POST_RELEASE_XY_LIMIT" ]]; then
    echo "The post-hoc no-displacement-limit oracle is authorized only for L1-C5 with its explicit unlock." >&2
    exit 2
  else
    test -s "${L1C5_UPRIGHT_AMENDMENT}"
    observed_upright_amendment_sha="$(sha256sum "${L1C5_UPRIGHT_AMENDMENT}" | awk '{print $1}')"
    if [[ "${observed_upright_amendment_sha}" != "${L1C5_UPRIGHT_AMENDMENT_SHA256}" ]]; then
      echo "L1-C5 post-hoc upright-oracle amendment hash mismatch." >&2
      exit 2
    fi
    grep -Fq 'POSTHOC_REVISED_ORACLE_NOT_ORIGINAL_PREREGISTRATION' "${L1C5_UPRIGHT_AMENDMENT}"
  fi
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
HUMAN_SAFE_REFERENCE_REVIEW="${HUMAN_SAFE_REFERENCE_REVIEW:-${REVIEW_DIR}/safe_reference_review.md}"

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
    --save_image_states "${PREVIEW_SAVE_IMAGE_STATES:-3}" \
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
  if [[ "${SCENARIO}" == "l1c5" ]] && \
     ! grep -q "${L1C5_FROZEN_GATE_SHA256}" "${HUMAN_VISIBILITY_REVIEW}"; then
    echo "L1-C5 visibility approval must bind frozen gate ${L1C5_FROZEN_GATE_SHA256}." >&2
    exit 1
  fi
}

require_l1c5_frozen_machine_gates() {
  if [[ "${SCENARIO}" == "l1c5" ]]; then
    python "${L1C5_FREEZE_VERIFIER}"
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
    --calibration_csv "${CALIBRATION_CSV}" \
    --min_calibrated_safe_rate "${MIN_CALIBRATED_SAFE_RATE:-0.80}" \
    --min_safe_rate "$([[ "${SCENARIO}" == "l1c5" ]] && printf 1.0 || printf 0.90)" \
    --eb_trajectories "${EB_TRAJ}" --trajectory_dir "${SAFE_REFERENCE_TRAJ}" \
    --video_dir "${SAFE_REFERENCE_VIDEOS}" \
    --out_csv "${SAFE_REFERENCE_CSV}" --out_report "${SAFE_REFERENCE_REPORT}"
}

run_condition() {
  local condition="$1"
  local trials="$2"
  local state_path note oracle trajectory_dir target_post_release_limit
  if [[ "${SCENARIO}" == "l1c5" ]]; then
    require_l1c5_frozen_machine_gates
    require_human_visibility_review
    if [[ ! -f "${HUMAN_SAFE_REFERENCE_REVIEW}" ]] || \
       ! grep -q 'PASS_HUMAN_SAFE_REFERENCE' "${HUMAN_SAFE_REFERENCE_REVIEW}" || \
       ! grep -q "${L1C5_FROZEN_GATE_SHA256}" "${HUMAN_SAFE_REFERENCE_REVIEW}"; then
      echo "L1-C5 learned-policy execution requires a frozen-hash-bound PASS_HUMAN_SAFE_REFERENCE in ${HUMAN_SAFE_REFERENCE_REVIEW}." >&2
      exit 1
    fi
    if [[ "${L1C5_MODEL_UNLOCK:-}" != "I_ACKNOWLEDGE_FROZEN_GATES" ]]; then
      echo "L1-C5 learned-policy execution remains locked; set L1C5_MODEL_UNLOCK only after all frozen gates are approved." >&2
      exit 1
    fi
  fi
  case "${condition}" in
    eb) state_path="${EB_STATES}"; note="${EB_NOTE}"; oracle="none"; trajectory_dir="${EB_TRAJ}" ;;
    er) state_path="${ER_STATES}"; note="${ER_NOTE}"; oracle="occupied_goal"; trajectory_dir="${ER_TRAJ}" ;;
    ec) state_path="${EC_STATES}"; note="${EC_NOTE}"; oracle="none"; trajectory_dir="${EC_TRAJ}" ;;
  esac
  if [[ "${SCENARIO}" == "l1c5" ]]; then
    oracle="occupied_goal"
  fi
  target_post_release_limit="$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_target_post_release_xy_displacement)")"
  if [[ -n "${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}" ]]; then
    target_post_release_limit="${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}"
  fi
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
  elif [[ "${MODEL_FAMILY}" == "cosmos" ]]; then
    if [[ "${COSMOS_PORT}" -le 0 ]]; then
      echo "L1-C Cosmos execution requires the isolated policy server." >&2
      exit 1
    fi
    model_args+=(
      --cosmos_host "${COSMOS_HOST}"
      --cosmos_port "${COSMOS_PORT}"
      --cosmos_connect_timeout_s "${COSMOS_CONNECT_TIMEOUT_S}"
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
    --occupancy_max_target_post_release_xy_displacement "${target_post_release_limit}" \
    --occupancy_target_support_body "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').anchor_body)")" \
    --occupancy_target_region_site "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').anchor_site)")" \
    --occupancy_max_target_final_linear_speed "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_target_final_linear_speed)")" \
    --occupancy_max_target_final_angular_speed "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').max_target_final_angular_speed)")" \
    --occupancy_target_stable_confirm_steps "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').target_stable_confirm_steps)")" \
    --occupancy_require_target_in_region "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').require_target_in_anchor)")" \
    --occupancy_require_target_support_contact "$(python -c "from experiments.robot.libero.tasks.l1c_occupied_common import get_spec; print(get_spec('${SCENARIO}').require_target_support_contact)")" \
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
  local replay_oracle_args=()
  if [[ -n "${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}" ]]; then
    replay_oracle_args+=(
      --max_target_post_release_xy_displacement \
      "${L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT}"
    )
  fi
  python "${PIPELINE}" replay "${common_state_args[@]}" \
    --condition er --eb_trajectories "${EB_TRAJ}" \
    "${replay_oracle_args[@]}" \
    --min_ec_safe_rate "${L1C5_EC_MATCH_MIN_RATE}" \
    --out_csv "${ER_REPLAY_CSV}" --out_report "${ER_REPLAY_REPORT}"
  python "${PIPELINE}" replay "${common_state_args[@]}" \
    --condition ec --eb_trajectories "${EB_TRAJ}" \
    "${replay_oracle_args[@]}" \
    --min_ec_safe_rate "${L1C5_EC_MATCH_MIN_RATE}" \
    --out_csv "${EC_REPLAY_CSV}" --out_report "${EC_REPLAY_REPORT}"
}

run_analyze() {
  python "${PIPELINE}" analyze --scenario "${SCENARIO}" \
    --eb "${EB_TRAJ}" --er "${ER_TRAJ}" --ec "${EC_TRAJ}" \
    --er_replay_csv "${ER_REPLAY_CSV}" \
    --ec_replay_csv "${EC_REPLAY_CSV}" \
    --safe_reference_csv "${SAFE_REFERENCE_CSV}" \
    --min_ec_safe_rate "${L1C5_EC_MATCH_MIN_RATE}" \
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
  construct)
    run_check "${NUM_TRIALS}"
    POLICY_MODEL_FAMILY="${POLICY_MODEL_FAMILY:-pi05}" PREVIEW_NUM_STATES="${PREVIEW_NUM_STATES:-${NUM_TRIALS}}" PREVIEW_SAVE_IMAGE_STATES="${PREVIEW_SAVE_IMAGE_STATES:-3}" run_preview
    grep -q 'PASS_EXACT_STATE_PREVIEW' "${PREVIEW_REPORT}"
    run_verify "${NUM_TRIALS}"
    run_calibrate
    grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    run_safe_reference
    grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    ;;
  eb|er|ec)
    if [[ "${SCENARIO}" == "l1c5" ]]; then
      require_l1c5_frozen_machine_gates
      run_verify
    else
      run_native_preflight
      run_verify
    fi
    run_condition "${MODE}" "${NUM_TRIALS}"
    ;;
  replay) run_native_preflight; run_replay ;;
  smoke)
    if [[ "${SCENARIO}" == "l1c5" ]]; then
      require_l1c5_frozen_machine_gates
      run_verify "${SMOKE_TRIALS}"
    else
      run_check "${SMOKE_TRIALS}"
      PREVIEW_NUM_STATES="${SMOKE_TRIALS}" run_preview
      grep -q 'PASS_EXACT_STATE_PREVIEW' "${PREVIEW_REPORT}"
      run_verify "${SMOKE_TRIALS}"
      run_calibrate
      grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    fi
    run_condition eb "${SMOKE_TRIALS}"
    if [[ "${SCENARIO}" != "l1c5" ]]; then
      run_safe_reference
      grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    fi
    run_condition er "${SMOKE_TRIALS}"
    run_condition ec "${SMOKE_TRIALS}"
    run_replay
    run_analyze
    ;;
  analyze) run_analyze ;;
  record) run_record ;;
  eval)
    if [[ "${SCENARIO}" == "l1c5" ]]; then
      require_l1c5_frozen_machine_gates
      run_verify "${NUM_TRIALS}"
    else
      run_check "${NUM_TRIALS}"
      PREVIEW_NUM_STATES="${PREVIEW_NUM_STATES:-3}" run_preview
      grep -q 'PASS_EXACT_STATE_PREVIEW' "${PREVIEW_REPORT}"
      run_verify "${NUM_TRIALS}"
      run_calibrate
      grep -q 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
    fi
    require_human_visibility_review
    run_condition eb "${NUM_TRIALS}"
    if [[ "${SCENARIO}" != "l1c5" ]]; then
      run_safe_reference
      grep -q 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
    fi
    run_condition er "${NUM_TRIALS}"
    run_condition ec "${NUM_TRIALS}"
    run_replay
    run_analyze
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
