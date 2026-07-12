#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODE="${1:-all}"
NUM_TRIALS="${NUM_TRIALS:-5}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:-1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-10}"
PATH_BDDL="experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_stove_on_path.bddl"
EC_BDDL="experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_stove_out_of_path.bddl"
EB_STATE="experiments/robot/libero/tasks/l2b2b_paired_eb_path_off.hdf5"
ER_STATE="experiments/robot/libero/tasks/l2b2b_paired_er_path_on.hdf5"
EC_STATE="experiments/robot/libero/tasks/l2b2b_paired_ec_path_far.hdf5"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_DEVICE_ID="${RENDER_GPU}"

generate() {
  python experiments/robot/libero/tasks/generate_l2b2_paired_initial_states.py \
    --num_states "${NUM_TRIALS}" --seed "${SEED}" \
    --eb_bddl "${PATH_BDDL}" --er_bddl "${PATH_BDDL}" --ec_bddl "${EC_BDDL}" \
    --eb_output "${EB_STATE}" --er_output "${ER_STATE}" --ec_output "${EC_STATE}"
}
run_one() {
  STATE_PATH="$2" RUN_ID_NOTE="$3" NUM_TRIALS="${NUM_TRIALS}" SEED="${SEED}" \
  RENDER_GPU="${RENDER_GPU}" SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE}" CHECKPOINT="${CHECKPOINT}" \
    bash experiments/robot/libero/tasks/run_l2b2_basket_stove.sh "$1" eval
}
evaluate() {
  run_one basket_path_off "${EB_STATE}" L2-B2b-paired-Eb-path-off
  run_one basket_path "${ER_STATE}" L2-B2b-paired-Er-path-on
  run_one basket_path_far "${EC_STATE}" L2-B2b-paired-Ec-path-far
}
references() { python experiments/robot/libero/tasks/validate_l2b2b_reference_paths.py; }
report() {
  for note in L2-B2b-paired-Eb-path-off L2-B2b-paired-Er-path-on L2-B2b-paired-Ec-path-far; do
    files=(experiments/logs/EVAL-*--"${note}".txt); echo; echo "### ${note}"
    [[ ${#files[@]} -gt 0 ]] || { echo "No log found"; continue; }
    log="$(ls -t "${files[@]}" | head -1)"; echo "${log}"
    grep -E "SemanticHazardProximityOracle metrics|Safety violation|Success:|Overall success rate|Overall SVR|Overall safe success rate" "${log}" || true
  done
}
case "${MODE}" in
  generate) generate;; eval) evaluate;; references) references;; report) report;;
  all) references; generate; evaluate; report;;
  *) echo "Usage: $0 [references|generate|eval|report|all]" >&2; exit 2;;
esac
