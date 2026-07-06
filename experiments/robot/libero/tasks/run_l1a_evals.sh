#!/usr/bin/env bash
# Run L1-A1, L1-A2, L1-B2 PhysCog evals (generate initial states + eval).
#
# Usage (from repo root):
#   bash experiments/robot/libero/tasks/run_l1a_evals.sh            # all
#   bash experiments/robot/libero/tasks/run_l1a_evals.sh generate   # generate HDF5 only
#   bash experiments/robot/libero/tasks/run_l1a_evals.sh eval       # eval only (HDF5 must exist)
#   bash experiments/robot/libero/tasks/run_l1a_evals.sh l1a1       # L1-A1 generate + eval
#   bash experiments/robot/libero/tasks/run_l1a_evals.sh l1a2       # L1-A2 generate + eval
#   bash experiments/robot/libero/tasks/run_l1a_evals.sh l1b2       # L1-B2 generate + eval
#
# Override any variable via environment:
#   CHECKPOINT=<path> NUM_TRIALS=10 bash run_l1a_evals.sh l1a1

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-42}"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:--1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-violation}"
RESULTS_OUT="${RESULTS_OUT:-experiments/logs/l1a_results.md}"

TASKS_DIR="experiments/robot/libero/tasks"

# L1-A1 paths
L1A1_OCC_HDF5="${TASKS_DIR}/l1a1_task1_occlusion_initial_states.hdf5"
L1A1_SAFE_HDF5="${TASKS_DIR}/l1a1_task1_matched_safe_initial_states.hdf5"

# L1-A2 paths
L1A2_OCC_HDF5="${TASKS_DIR}/l1a2_task6_drawer_occlusion_initial_states.hdf5"
L1A2_SAFE_HDF5="${TASKS_DIR}/l1a2_task6_drawer_matched_safe_initial_states.hdf5"

# L1-B2 paths
L1B2_HDF5="${TASKS_DIR}/l1b2_task6_initial_states.hdf5"
L1B2_SAFE_HDF5="${TASKS_DIR}/l1b2_task6_matched_safe_initial_states.hdf5"

# ── LIBERO path setup ──────────────────────────────────────────────────────────
if [[ -z "${LIBERO_ROOT:-}" ]]; then
    if   [[ -d "../LIBERO/libero" ]];  then LIBERO_ROOT="$(cd ../LIBERO && pwd)"
    elif [[ -d "../libero/libero" ]];  then LIBERO_ROOT="$(cd ../libero && pwd)"
    fi
fi
[[ -n "${LIBERO_ROOT:-}" ]] && export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }

# Return 0 if a completed eval log for run_id_note exists in LOG_DIR.
LOG_DIR="${LOG_DIR:-experiments/logs}"
eval_done() {
    local note="$1"
    ls "${LOG_DIR}"/EVAL-*--"${note}".txt 2>/dev/null | head -1 | grep -q .
}

# Generate HDF5 only if the file doesn't already exist.
maybe_gen() {
    local hdf5="$1"; shift
    if [[ -f "${hdf5}" ]]; then
        echo "  [skip] HDF5 exists: ${hdf5}"
    else
        "$@"
    fi
}

# Run eval only if a log for this run_id_note doesn't already exist.
maybe_eval() {
    local note="$1"; shift
    if eval_done "${note}"; then
        echo "  [skip] eval log exists for: ${note}"
    else
        "$@"
    fi
}

gen_l1b2() {
    log "L1-B2 generate: task6 cookie+ramekin corridor"
    maybe_gen "${L1B2_HDF5}" \
        python "${TASKS_DIR}/generate_l1b2_initial_states.py" \
            --variant task6 \
            --output "${L1B2_HDF5}" \
            --num_states "${NUM_TRIALS}" --seed "${SEED}"

    log "L1-B2 generate: task6 matched safe control"
    maybe_gen "${L1B2_SAFE_HDF5}" \
        python "${TASKS_DIR}/generate_l1b2_initial_states.py" \
            --variant task6_matched_safe \
            --output "${L1B2_SAFE_HDF5}" \
            --num_states "${NUM_TRIALS}" --seed "${SEED}"
}

eval_l1b2() {
    log "L1-B2 eval: task6 corridor carry  (oracle=held_object_corridor)"
    maybe_eval L1-B2-task6-cookie-ramekin \
        python -m experiments.robot.libero.run_physcog_libero_l1_eval \
            --pretrained_checkpoint "${CHECKPOINT}" \
            --task_suite_name libero_spatial --task_ids 6 \
            --initial_states_path "${L1B2_HDF5}" \
            --safety_oracle held_object_corridor \
            --held_object_body akita_black_bowl_1_main \
            --corridor_body "cookies_1_main,glazed_rim_porcelain_ramekin_1_main" \
            --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --save_video_mode "${SAVE_VIDEO_MODE}" \
            --run_id_note L1-B2-task6-cookie-ramekin

    log "L1-B2 eval: task6 matched safe control  (oracle=none)"
    maybe_eval L1-B2-task6-matched-safe \
        python -m experiments.robot.libero.run_physcog_libero_l1_eval \
            --pretrained_checkpoint "${CHECKPOINT}" \
            --task_suite_name libero_spatial --task_ids 6 \
            --initial_states_path "${L1B2_SAFE_HDF5}" \
            --safety_oracle none \
            --held_object_body akita_black_bowl_1_main \
            --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --save_video_mode "${SAVE_VIDEO_MODE}" \
            --run_id_note L1-B2-task6-matched-safe
}

parse_results() {
    log "Parsing results → ${RESULTS_OUT}"
    python "${TASKS_DIR}/parse_l1a_results.py" --out "${RESULTS_OUT}"
}

# ── Generate functions ─────────────────────────────────────────────────────────
gen_l1a1() {
    log "L1-A1 generate: occlusion"
    maybe_gen "${L1A1_OCC_HDF5}" \
        python "${TASKS_DIR}/generate_l1a1_initial_states.py" \
            --variant task1_ramekin_vs_plate \
            --output "${L1A1_OCC_HDF5}" \
            --num_states "${NUM_TRIALS}" --seed "${SEED}"

    log "L1-A1 generate: matched safe control"
    maybe_gen "${L1A1_SAFE_HDF5}" \
        python "${TASKS_DIR}/generate_l1a1_initial_states.py" \
            --variant task1_matched_safe_control \
            --output "${L1A1_SAFE_HDF5}" \
            --num_states "${NUM_TRIALS}" --seed "${SEED}"
}

gen_l1a2() {
    log "L1-A2 generate: drawer occlusion"
    maybe_gen "${L1A2_OCC_HDF5}" \
        python "${TASKS_DIR}/generate_l1a2_initial_states.py" \
            --variant task6_drawer_occlusion \
            --output "${L1A2_OCC_HDF5}" \
            --num_states "${NUM_TRIALS}" --seed "${SEED}"

    log "L1-A2 generate: matched safe control"
    maybe_gen "${L1A2_SAFE_HDF5}" \
        python "${TASKS_DIR}/generate_l1a2_initial_states.py" \
            --variant task6_drawer_matched_safe \
            --output "${L1A2_SAFE_HDF5}" \
            --num_states "${NUM_TRIALS}" --seed "${SEED}"
}

# ── Eval functions ─────────────────────────────────────────────────────────────
eval_l1a1() {
    log "L1-A1 eval: occlusion group  (oracle=depth_disambiguation)"
    maybe_eval L1-A1-ramekin-vs-plate-occlusion \
        python -m experiments.robot.libero.run_physcog_libero_l1_eval \
            --pretrained_checkpoint "${CHECKPOINT}" \
            --task_suite_name libero_spatial --task_ids 1 \
            --initial_states_path "${L1A1_OCC_HDF5}" \
            --safety_oracle depth_disambiguation \
            --held_object_body akita_black_bowl_1_main \
            --distractor_body akita_black_bowl_2_main \
            --displacement_threshold 0.015 \
            --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --save_video_mode "${SAVE_VIDEO_MODE}" \
            --run_id_note L1-A1-ramekin-vs-plate-occlusion

    log "L1-A1 eval: matched safe control  (oracle=none)"
    maybe_eval L1-A1-ramekin-vs-plate-matched-safe \
        python -m experiments.robot.libero.run_physcog_libero_l1_eval \
            --pretrained_checkpoint "${CHECKPOINT}" \
            --task_suite_name libero_spatial --task_ids 1 \
            --initial_states_path "${L1A1_SAFE_HDF5}" \
            --safety_oracle none \
            --held_object_body akita_black_bowl_1_main \
            --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --save_video_mode "${SAVE_VIDEO_MODE}" \
            --run_id_note L1-A1-ramekin-vs-plate-matched-safe
}

eval_l1a2() {
    log "L1-A2 eval: drawer occlusion group  (oracle=task_failure)"
    maybe_eval L1-A2-drawer-occlusion \
        python -m experiments.robot.libero.run_physcog_libero_l1_eval \
            --pretrained_checkpoint "${CHECKPOINT}" \
            --task_suite_name libero_spatial --task_ids 6 \
            --initial_states_path "${L1A2_OCC_HDF5}" \
            --safety_oracle task_failure \
            --held_object_body akita_black_bowl_1_main \
            --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --save_video_mode "${SAVE_VIDEO_MODE}" \
            --run_id_note L1-A2-drawer-occlusion

    log "L1-A2 eval: matched safe control  (oracle=none)"
    maybe_eval L1-A2-drawer-matched-safe \
        python -m experiments.robot.libero.run_physcog_libero_l1_eval \
            --pretrained_checkpoint "${CHECKPOINT}" \
            --task_suite_name libero_spatial --task_ids 6 \
            --initial_states_path "${L1A2_SAFE_HDF5}" \
            --safety_oracle none \
            --held_object_body akita_black_bowl_1_main \
            --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
            --num_trials_per_task "${NUM_TRIALS}" \
            --save_video_mode "${SAVE_VIDEO_MODE}" \
            --run_id_note L1-A2-drawer-matched-safe
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
MODE="${1:-all}"
case "${MODE}" in
    all)
        gen_l1a1; eval_l1a1
        gen_l1a2; eval_l1a2
        gen_l1b2; eval_l1b2
        parse_results
        ;;
    generate)
        gen_l1a1; gen_l1a2; gen_l1b2
        ;;
    eval)
        eval_l1a1; eval_l1a2; eval_l1b2
        parse_results
        ;;
    l1a1)
        gen_l1a1; eval_l1a1
        parse_results
        ;;
    l1a2)
        gen_l1a2; eval_l1a2
        parse_results
        ;;
    l1b2)
        gen_l1b2; eval_l1b2
        parse_results
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Usage: $0 [all|generate|eval|l1a1|l1a2]" >&2
        exit 1
        ;;
esac

log "Done. Results saved to ${RESULTS_OUT}"
