#!/usr/bin/env bash
set -euo pipefail

# L3 alignment-conditioned release on the native LIBERO-10 task:
#   "pick up the book and place it in the back compartment of the caddy"
#
# Usage:
#   experiments/robot/libero/tasks/run_l3_book_caddy.sh probe
#   experiments/robot/libero/tasks/run_l3_book_caddy.sh calibrate
#   experiments/robot/libero/tasks/run_l3_book_caddy.sh smoke
#   experiments/robot/libero/tasks/run_l3_book_caddy.sh eval

MODE="${1:-probe}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SMOKE_TRIALS="${SMOKE_TRIALS:-3}"
RENDER_GPU="${RENDER_GPU:--1}"
LIBERO_ROOT="${LIBERO_ROOT:-}"

BOOK_BODY="${BOOK_BODY:-black_book_1_main}"
BACK_SITE="${BACK_SITE:-desk_caddy_1_back_contain_region}"
MAX_ALIGNMENT_DEG="${MAX_ALIGNMENT_DEG:-15.0}"
MAX_TILT_DEG="${MAX_TILT_DEG:-20.0}"
MIN_REGION_MARGIN="${MIN_REGION_MARGIN:-0.002}"
MAX_RELEASE_SPEED="${MAX_RELEASE_SPEED:-0.08}"
MAX_POST_DRIFT="${MAX_POST_DRIFT:-0.025}"
RELEASE_CONFIRM_STEPS="${RELEASE_CONFIRM_STEPS:-2}"
SETTLE_STEPS="${SETTLE_STEPS:-50}"

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
  python experiments/robot/libero/tasks/probe_l3_book_caddy.py
}

run_eval() {
  local trials="$1"
  local note="$2"
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_10 \
    --task_ids 5 \
    --safety_oracle alignment_conditioned_release \
    --held_object_body "${BOOK_BODY}" \
    --release_container_site "${BACK_SITE}" \
    --release_max_alignment_error_deg "${MAX_ALIGNMENT_DEG}" \
    --release_max_tilt_deg "${MAX_TILT_DEG}" \
    --release_min_region_margin "${MIN_REGION_MARGIN}" \
    --release_max_speed "${MAX_RELEASE_SPEED}" \
    --release_max_post_drift "${MAX_POST_DRIFT}" \
    --release_confirm_steps "${RELEASE_CONFIRM_STEPS}" \
    --post_success_settle_steps "${SETTLE_STEPS}" \
    --num_trials_per_task "${trials}" \
    --save_video_mode all \
    --run_id_note "${note}" \
    --render_gpu_device_id "${RENDER_GPU}"
}

case "${MODE}" in
  probe) run_probe ;;
  calibrate)
    MAX_ALIGNMENT_DEG=180.0
    MAX_TILT_DEG=180.0
    MIN_REGION_MARGIN=-1.0
    MAX_RELEASE_SPEED=999.0
    MAX_POST_DRIFT=999.0
    run_eval "${NUM_TRIALS}" "L3-book-caddy-release-calibration"
    ;;
  smoke) run_eval "${SMOKE_TRIALS}" "L3-book-caddy-release-smoke" ;;
  eval)  run_eval "${NUM_TRIALS}" "L3-book-caddy-alignment-conditioned-release" ;;
  all)   run_probe; run_eval "${NUM_TRIALS}" "L3-book-caddy-alignment-conditioned-release" ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: probe, calibrate, smoke, eval, all" >&2
    exit 2
    ;;
esac
