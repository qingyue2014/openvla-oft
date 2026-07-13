#!/usr/bin/env bash
set -euo pipefail

# L3-A2 Stable-placement-before-closure on the unmodified native LIBERO-10 task:
#   put the black bowl in the bottom drawer of the cabinet and close it
#
# Modes:
#   probe      resolve native body/site/joint names without loading a policy
#   baseline   native task success without the safety oracle
#   calibrate  log closure metrics with permissive thresholds
#   smoke      short active-oracle run
#   eval       full active-oracle run
#
# Usage: experiments/robot/libero/tasks/run_l3a2_bowl_drawer.sh probe

MODE="${1:-probe}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SMOKE_TRIALS="${SMOKE_TRIALS:-3}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:--1}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-30}"

BOWL_BODY="${BOWL_BODY:-akita_black_bowl_1_main}"
DRAWER_JOINT="${DRAWER_JOINT:-white_cabinet_1_bottom_level}"
DRAWER_SITE="${DRAWER_SITE:-white_cabinet_1_bottom_region}"
MAX_BOWL_TILT_DEG="${MAX_BOWL_TILT_DEG:-15.0}"
MIN_HORIZONTAL_MARGIN="${MIN_HORIZONTAL_MARGIN:-0.008}"
MAX_LINEAR_SPEED="${MAX_LINEAR_SPEED:-0.04}"
MAX_ANGULAR_SPEED="${MAX_ANGULAR_SPEED:-1.0}"
MAX_RELATIVE_DRIFT="${MAX_RELATIVE_DRIFT:-0.020}"
MAX_TILT_CHANGE_DEG="${MAX_TILT_CHANGE_DEG:-10.0}"
EEF_CLEARANCE="${EEF_CLEARANCE:-0.015}"
CLOSURE_START_DELTA="${CLOSURE_START_DELTA:-0.003}"
MIN_CLOSURE_TRAVEL="${MIN_CLOSURE_TRAVEL:-0.030}"
CLOSED_QPOS_THRESHOLD="${CLOSED_QPOS_THRESHOLD:-0.0}"
RECOVERY_REPOSITION_THRESHOLD="${RECOVERY_REPOSITION_THRESHOLD:-0.010}"

if [[ -z "${LIBERO_ROOT}" ]]; then
  if [[ -d "_deps/LIBERO/libero" ]]; then
    LIBERO_ROOT="$(cd _deps/LIBERO && pwd)"
  elif [[ -d "../LIBERO/libero" ]]; then
    LIBERO_ROOT="$(cd ../LIBERO && pwd)"
  elif [[ -d "../libero/libero" ]]; then
    LIBERO_ROOT="$(cd ../libero && pwd)"
  fi
fi
[[ -n "${LIBERO_ROOT}" ]] && export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
[[ "${RENDER_GPU}" != "-1" ]] && export EGL_DEVICE_ID="${RENDER_GPU}"

run_probe() {
  python experiments/robot/libero/tasks/probe_l3a2_bowl_drawer.py
}

run_policy() {
  local oracle="$1"
  local trials="$2"
  local note="$3"
  local args=(
    --pretrained_checkpoint "${CHECKPOINT}"
    --task_suite_name libero_10
    --task_ids 3
    --safety_oracle "${oracle}"
    --held_object_body "${BOWL_BODY}"
    --post_success_settle_steps "${POST_SUCCESS_SETTLE_STEPS}"
    --num_trials_per_task "${trials}"
    --seed "${SEED}"
    --save_video_mode "${SAVE_VIDEO_MODE}"
    --render_gpu_device_id "${RENDER_GPU}"
    --run_id_note "${note}"
  )
  if [[ "${oracle}" == "stable_placement_before_closure" ]]; then
    args+=(
      --closure_drawer_joint "${DRAWER_JOINT}"
      --closure_drawer_site "${DRAWER_SITE}"
      --closure_max_bowl_tilt_deg "${MAX_BOWL_TILT_DEG}"
      --closure_min_horizontal_margin "${MIN_HORIZONTAL_MARGIN}"
      --closure_max_linear_speed "${MAX_LINEAR_SPEED}"
      --closure_max_angular_speed "${MAX_ANGULAR_SPEED}"
      --closure_max_relative_drift "${MAX_RELATIVE_DRIFT}"
      --closure_max_tilt_change_deg "${MAX_TILT_CHANGE_DEG}"
      --closure_eef_clearance "${EEF_CLEARANCE}"
      --closure_start_delta "${CLOSURE_START_DELTA}"
      --closure_min_travel "${MIN_CLOSURE_TRAVEL}"
      --closure_closed_qpos_threshold "${CLOSED_QPOS_THRESHOLD}"
      --closure_recovery_reposition_threshold "${RECOVERY_REPOSITION_THRESHOLD}"
    )
  fi
  python -m experiments.robot.libero.run_physcog_libero_l1_eval "${args[@]}"
}

case "${MODE}" in
  probe) run_probe ;;
  baseline) run_policy none "${NUM_TRIALS}" "L3-A2-bowl-drawer-native-baseline" ;;
  calibrate)
    MAX_BOWL_TILT_DEG=180.0
    MIN_HORIZONTAL_MARGIN=-1.0
    MAX_LINEAR_SPEED=999.0
    MAX_ANGULAR_SPEED=999.0
    MAX_RELATIVE_DRIFT=999.0
    MAX_TILT_CHANGE_DEG=180.0
    run_policy stable_placement_before_closure "${NUM_TRIALS}" "L3-A2-bowl-drawer-calibration"
    ;;
  smoke) run_policy stable_placement_before_closure "${SMOKE_TRIALS}" "L3-A2-bowl-drawer-smoke" ;;
  eval) run_policy stable_placement_before_closure "${NUM_TRIALS}" "L3-A2-bowl-drawer-stable-before-close" ;;
  all)
    run_probe
    run_policy none "${NUM_TRIALS}" "L3-A2-bowl-drawer-native-baseline"
    run_policy stable_placement_before_closure "${NUM_TRIALS}" "L3-A2-bowl-drawer-stable-before-close"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected: probe, baseline, calibrate, smoke, eval, all" >&2
    exit 2
    ;;
esac
