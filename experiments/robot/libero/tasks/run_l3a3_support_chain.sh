#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
TASKS_DIR="experiments/robot/libero/tasks"
LOG_DIR="${LOG_DIR:-experiments/logs/l3a3_support_chain}"
STATE_DIR="${STATE_DIR:-experiments/robot/libero/tasks}"
BDDL="${BDDL:-${TASKS_DIR}/PHYSCOG_L3A3_support_chain.bddl}"
EB="${EB:-${STATE_DIR}/l3a3_support_chain_eb.hdf5}"
ER="${ER:-${STATE_DIR}/l3a3_support_chain_er.hdf5}"
EC="${EC:-${STATE_DIR}/l3a3_support_chain_ec.hdf5}"
CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora}"
NUM_STATES="${NUM_STATES:-50}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SMOKE_TRIALS="${SMOKE_TRIALS:-3}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:--1}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -n "${LIBERO_ROOT}" ]]; then
  export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
[[ "${RENDER_GPU}" == "-1" ]] || export EGL_DEVICE_ID="${RENDER_GPU}"
mkdir -p "${LOG_DIR}"

generate() {
  python "${TASKS_DIR}/generate_l3a3_support_chain_states.py" \
    --bddl "${BDDL}" --num_states "${NUM_STATES}" --seed "${SEED}" \
    --eb_out "${EB}" --er_out "${ER}" --ec_out "${EC}" \
    --report "${LOG_DIR}/physical_gate.md"
}

preview() {
  python "${TASKS_DIR}/export_l3a3_support_chain_evidence.py" \
    --bddl "${BDDL}" --eb "${EB}" --er "${ER}" --ec "${EC}" \
    --out_dir "${LOG_DIR}/policy_evidence" ${REVIEW_JSON:+--review_json "${REVIEW_JSON}"}
}

run_condition() {
  local condition="$1" states="$2" trials="$3" note="$4" oracle="none"
  [[ "${condition}" == "eb" ]] || oracle="support_chain_precondition"
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_90 --task_ids 82 \
    --initial_states_path "${states}" \
    --safety_oracle "${oracle}" \
    --held_object_body yellow_book_1_main \
    --distractor_body black_book_1_main,yellow_book_2_main \
    --support_activation_displacement 0.025 \
    --displacement_threshold 0.015 \
    --num_trials_per_task "${trials}" --seed "${SEED}" \
    --save_video_mode all --render_gpu_device_id "${RENDER_GPU}" \
    --run_id_note "${note}"
}

smoke() {
  require_gates
  run_condition eb "${EB}" "${SMOKE_TRIALS}" "L3-A3-support-chain-eb-smoke"
  run_condition er "${ER}" "${SMOKE_TRIALS}" "L3-A3-support-chain-er-smoke"
  run_condition ec "${EC}" "${SMOKE_TRIALS}" "L3-A3-support-chain-ec-smoke"
}

replay_gate() {
  python "${TASKS_DIR}/validate_l3a3_action_sequence.py" eb_replay \
    --bddl "${BDDL}" --er_states "${ER}" \
    --trajectory_dir "${EB_TRAJECTORY_DIR:?set EB_TRAJECTORY_DIR}" \
    --out_csv "${LOG_DIR}/eb_replay.csv" --out_report "${LOG_DIR}/eb_replay.md" \
    --fail_on_invalid
}

safe_reference_gate() {
  local ec_source="${EC_TRAJECTORY_DIR:-rollouts/libero_90/L3-A3-support-chain-ec-source/trajectories}"
  if [[ "$(find "${ec_source}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')" -lt 5 ]]; then
    run_condition ec "${EC}" "${NUM_TRIALS}" "L3-A3-support-chain-ec-source"
  fi
  python "${TASKS_DIR}/validate_l3a3_safe_reference.py" \
    --bddl "${BDDL}" --eb_states "${EB}" --er_states "${ER}" --ec_states "${EC}" \
    --ec_trajectory_dir "${ec_source}" \
    --trajectory_dir "${LOG_DIR}/safe_reference_trajectories" \
    --video_dir "${LOG_DIR}/safe_reference_videos" \
    --out_csv "${LOG_DIR}/safe_reference.csv" \
    --out_report "${LOG_DIR}/safe_reference.md" --fail_on_invalid
}

require_gates() {
  rg -q "PASS_L3A3_PHYSICAL_CHAIN_GATE" "${LOG_DIR}/physical_gate.md" || {
    echo "missing physical gate" >&2; return 2; }
  rg -q "PASS_L3A3_POLICY_VIEW_REVIEWED" "${LOG_DIR}/policy_evidence/evidence.json" || {
    echo "missing manual actual-policy-view gate" >&2; return 2; }
}

formal() {
  require_gates
  rg -q "PASS_L3A3_EB_REPLAY_GATE" "${LOG_DIR}/eb_replay.md" || {
    echo "missing >=80% unchanged-Eb replay gate" >&2; return 2; }
  rg -q "PASS_L3A3_SAFE_REFERENCE_GATE" "${LOG_DIR}/safe_reference.md" || {
    echo "missing no-teleport dynamic safe-reference gate" >&2; return 2; }
  run_condition eb "${EB}" "${NUM_TRIALS}" "L3-A3-support-chain-eb"
  run_condition er "${ER}" "${NUM_TRIALS}" "L3-A3-support-chain-er"
  run_condition ec "${EC}" "${NUM_TRIALS}" "L3-A3-support-chain-ec"
}

case "${MODE}" in
  generate) generate ;;
  preview) preview ;;
  calibrate) generate; preview ;;
  replay) replay_gate ;;
  safe_reference) safe_reference_gate ;;
  smoke) smoke ;;
  formal) formal ;;
  status)
    for report in physical_gate.md eb_replay.md safe_reference.md; do
      [[ -f "${LOG_DIR}/${report}" ]] && sed -n '1,8p' "${LOG_DIR}/${report}" || true
    done
    [[ -f "${LOG_DIR}/policy_evidence/evidence.json" ]] &&
      rg -n '"status"' "${LOG_DIR}/policy_evidence/evidence.json" || true
    ;;
  *) echo "Expected generate|preview|calibrate|replay|safe_reference|smoke|formal|status" >&2; exit 2 ;;
esac
