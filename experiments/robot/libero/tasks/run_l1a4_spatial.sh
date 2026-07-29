#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
TASKS_DIR="experiments/robot/libero/tasks"
LOG_DIR="${LOG_DIR:-experiments/logs}"
PIPELINE="${TASKS_DIR}/l1a4_spatial_pipeline.py"

NUM_STATES="${NUM_STATES:-50}"
EB_CAPABILITY_TRIALS="${EB_CAPABILITY_TRIALS:-10}"
SEED="${SEED:-42}"
EVAL_SEED="${EVAL_SEED:-7}"
MODEL_FAMILY="${MODEL_FAMILY:-openvla}"
CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial}"
PI05_HOST="${PI05_HOST:-127.0.0.1}"
PI05_PORT="${PI05_PORT:-8000}"
PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-300}"
PI05_REPLAN_STEPS="${PI05_REPLAN_STEPS:-5}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-none}"
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-True}"

EB_STATES="${TASKS_DIR}/l1a4_spatial_eb_states.hdf5"
ER_STATES="${TASKS_DIR}/l1a4_spatial_er_states.hdf5"
EC_STATES="${TASKS_DIR}/l1a4_spatial_ec_states.hdf5"
PAIRING="${TASKS_DIR}/l1a4_spatial_pairing.json"
PREFLIGHT_MANIFEST="${TASKS_DIR}/l1a4_spatial_native_preflight.json"
PREFLIGHT_REPORT="${LOG_DIR}/l1a4_spatial_native_preflight.md"
PREVIEW_DIR="${TASKS_DIR}/l1a4_spatial_preview"
VISIBILITY_REVIEW="${TASKS_DIR}/L1-A4-SPATIAL_VISIBILITY_REVIEW.md"

TARGET="akita_black_bowl_1_main"
LURE="akita_black_bowl_2_main"
TRACKED="${TARGET},${LURE},plate_1_main,glazed_rim_porcelain_ramekin_1_main,cookies_1_main,wooden_cabinet_1_main,flat_stove_1_main"
EB_NOTE="${EB_NOTE:-L1-A4-between-eb-native-pi05}"

if [[ -d "_deps/LIBERO/libero" ]]; then
  export LIBERO_ROOT="${LIBERO_ROOT:-$(cd _deps/LIBERO && pwd)}"
elif [[ -d "../LIBERO/libero" ]]; then
  export LIBERO_ROOT="${LIBERO_ROOT:-$(cd ../LIBERO && pwd)}"
elif [[ -d "../libero/libero" ]]; then
  export LIBERO_ROOT="${LIBERO_ROOT:-$(cd ../libero && pwd)}"
fi
if [[ -z "${LIBERO_ROOT:-}" || ! -d "${LIBERO_ROOT}/libero" ]]; then
  echo "L1-A4 spatial cannot locate native LIBERO; set LIBERO_ROOT." >&2
  exit 2
fi
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ "${RENDER_GPU_DEVICE_ID}" != "-1" ]]; then
  export EGL_DEVICE_ID="${EGL_DEVICE_ID:-${RENDER_GPU_DEVICE_ID}}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${RENDER_GPU_DEVICE_ID}}"
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

preflight() {
  log "L1-A4 spatial native-only preflight"
  python "${TASKS_DIR}/validate_l1a4_spatial_native_preflight.py" \
    --manifest "${PREFLIGHT_MANIFEST}" \
    --report "${PREFLIGHT_REPORT}"
}

generate() {
  log "L1-A4 spatial paired EB/ER/EC generation"
  python "${PIPELINE}" generate \
    --eb_states "${EB_STATES}" \
    --er_states "${ER_STATES}" \
    --ec_states "${EC_STATES}" \
    --pairing_manifest "${PAIRING}" \
    --preflight_manifest "${PREFLIGHT_MANIFEST}" \
    --preflight_report "${PREFLIGHT_REPORT}" \
    --preview_dir "${PREVIEW_DIR}" \
    --preview_count 3 \
    --num_states "${NUM_STATES}" \
    --seed "${SEED}"
}

states_ready() {
  [[ -f "${EB_STATES}" && -f "${ER_STATES}" && -f "${EC_STATES}" ]] \
    && [[ -f "${PAIRING}" && -f "${PREFLIGHT_MANIFEST}" ]] \
    && grep -q "PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE" "${PAIRING}" \
    && grep -q "PASS_L1A4_SPATIAL_NATIVE_ONLY_PREFLIGHT" "${PREFLIGHT_MANIFEST}"
}

ensure_states() {
  if ! states_ready; then
    generate
  fi
}

preview() {
  ensure_states
  log "L1-A4 spatial exact serialized-state policy-view preview"
  python "${PIPELINE}" preview \
    --eb_states "${EB_STATES}" \
    --er_states "${ER_STATES}" \
    --ec_states "${EC_STATES}" \
    --out_dir "${PREVIEW_DIR}" \
    --num_states 3
}

require_visibility_review() {
  ensure_states
  if [[ ! -f "${VISIBILITY_REVIEW}" ]] \
    || ! grep -q "PASS_HUMAN_POLICY_VIEW_VISIBILITY" "${VISIBILITY_REVIEW}"; then
    {
      echo "L1-A4 spatial HUMAN_VISIBILITY_REVIEW_REQUIRED."
      echo "Inspect EB/ER/EC agentview and eye-in-hand PNGs under ${PREVIEW_DIR},"
      echo "then record PASS_HUMAN_POLICY_VIEW_VISIBILITY in ${VISIBILITY_REVIEW}."
    } >&2
    exit 2
  fi
}

eb_capability() {
  require_visibility_review
  log "L1-A4 spatial EB evaluation: ${EB_NOTE}"
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --model_family "${MODEL_FAMILY}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --pi05_host "${PI05_HOST}" \
    --pi05_port "${PI05_PORT}" \
    --pi05_connect_timeout_s "${PI05_CONNECT_TIMEOUT_S}" \
    --pi05_replan_steps "${PI05_REPLAN_STEPS}" \
    --task_suite_name libero_spatial \
    --task_ids 0 \
    --initial_states_path "${EB_STATES}" \
    --native_only_preflight_manifest "${PREFLIGHT_MANIFEST}" \
    --safety_oracle none \
    --held_object_body "${TARGET}" \
    --trajectory_track_bodies "${TRACKED}" \
    --save_trajectory "${SAVE_TRAJECTORY}" \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --num_trials_per_task "${EB_CAPABILITY_TRIALS}" \
    --seed "${EVAL_SEED}" \
    --save_video_mode "${SAVE_VIDEO_MODE}" \
    --run_id_note "${EB_NOTE}"
  echo "verdict=PASS_L1A4_SPATIAL_EB_CAPABILITY_RUN"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  check)
    generate
    ;;
  preview)
    preview
    ;;
  eb_capability)
    eb_capability
    ;;
  *)
    echo "Usage: $0 preflight|check|preview|eb_capability" >&2
    exit 2
    ;;
esac
