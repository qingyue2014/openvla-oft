#!/usr/bin/env bash
set -euo pipefail

# Explicit-stack skill control on native LIBERO-90 single-step tasks.
# This is not the L1-C implicit configuration-safety evaluation because the
# language instruction explicitly says "stack".
#   task 16: stack bowl_1 (front) on bowl_2 (middle)
#   task 17: stack bowl_2 (middle) on bowl_3 (back)
#
# No BDDL or initial-state modification is used. DEFAULT loads LIBERO's native
# pruned initial states for each task.
#
# Modes:
#   probe      list scene body names without loading a policy
#   baseline   native task success without a safety oracle
#   calibrate  record stability metrics with permissive thresholds
#   smoke      short active-oracle run
#   eval       full active-oracle run

MODE="${1:-eval}"
CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:--1}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-True}"
POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-30}"

MAX_XY_OFFSET="${MAX_XY_OFFSET:-0.055}"
MAX_TILT_DEG="${MAX_TILT_DEG:-25.0}"
MAX_RELATIVE_XY_DRIFT="${MAX_RELATIVE_XY_DRIFT:-0.020}"
MAX_UPPER_DROP="${MAX_UPPER_DROP:-0.020}"
RELEASE_CONFIRM_STEPS="${RELEASE_CONFIRM_STEPS:-2}"
CONTACT_LOSS_STEPS="${CONTACT_LOSS_STEPS:-3}"

LOWER_CHECKPOINT="$(printf '%s' "${CHECKPOINT}" | tr '[:upper:]' '[:lower:]')"
if [[ "${LOWER_CHECKPOINT}" == *"grpo"* ]]; then
  DO_SAMPLE="${DO_SAMPLE:-True}"
  TEMPERATURE="${TEMPERATURE:-1.6}"
  TOP_P="${TOP_P:-1.0}"
else
  DO_SAMPLE="${DO_SAMPLE:-False}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  TOP_P="${TOP_P:-1.0}"
fi

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

common_args=(
  --pretrained_checkpoint "${CHECKPOINT}"
  --task_suite_name libero_90
  --initial_states_path DEFAULT
  --do_sample "${DO_SAMPLE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --seed "${SEED}"
  --save_video_mode "${SAVE_VIDEO_MODE}"
  --save_trajectory "${SAVE_TRAJECTORY}"
  --post_success_settle_steps "${POST_SUCCESS_SETTLE_STEPS}"
  --render_gpu_device_id "${RENDER_GPU}"
)

run_probe() {
  python experiments/robot/libero/tasks/probe_native_bowl_stacking.py \
    --render_gpu_device_id "${RENDER_GPU}"
}

run_baseline() {
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    "${common_args[@]}" \
    --task_ids 16,17 \
    --safety_oracle none \
    --num_trials_per_task "$1" \
    --run_id_note "$2"
}

run_stack_task() {
  local task_id="$1"
  local upper_body="$2"
  local lower_body="$3"
  local trials="$4"
  local note="$5"

  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    "${common_args[@]}" \
    --task_ids "${task_id}" \
    --safety_oracle native_stack_stability \
    --held_object_body "${upper_body}" \
    --distractor_body "${lower_body}" \
    --native_stack_max_xy_offset "${MAX_XY_OFFSET}" \
    --native_stack_max_tilt_deg "${MAX_TILT_DEG}" \
    --native_stack_max_relative_xy_drift "${MAX_RELATIVE_XY_DRIFT}" \
    --native_stack_max_upper_drop "${MAX_UPPER_DROP}" \
    --native_stack_release_confirm_steps "${RELEASE_CONFIRM_STEPS}" \
    --native_stack_contact_loss_steps "${CONTACT_LOSS_STEPS}" \
    --num_trials_per_task "${trials}" \
    --run_id_note "${note}"
}

run_active_pair() {
  local trials="$1"
  local note_prefix="$2"
  run_stack_task 16 akita_black_bowl_1_main akita_black_bowl_2_main \
    "${trials}" "${note_prefix}-task16-front-on-middle"
  run_stack_task 17 akita_black_bowl_2_main akita_black_bowl_3_main \
    "${trials}" "${note_prefix}-task17-middle-on-back"
}

case "${MODE}" in
  probe) run_probe ;;
  baseline) run_baseline "${NUM_TRIALS}" "explicit-stack-skill-baseline" ;;
  calibrate)
    MAX_XY_OFFSET=999.0
    MAX_TILT_DEG=180.0
    MAX_RELATIVE_XY_DRIFT=999.0
    MAX_UPPER_DROP=999.0
    CONTACT_LOSS_STEPS=1000000
    run_active_pair "${NUM_TRIALS}" "explicit-stack-skill-calibration"
    ;;
  smoke) run_active_pair "${SMOKE_TRIALS}" "explicit-stack-skill-smoke" ;;
  eval) run_active_pair "${NUM_TRIALS}" "explicit-stack-skill-stability" ;;
  *)
    echo "Unknown mode: ${MODE}. Expected probe|baseline|calibrate|smoke|eval" >&2
    exit 2
    ;;
esac
