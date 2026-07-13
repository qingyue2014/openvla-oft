#!/usr/bin/env bash
set -euo pipefail

# Native LIBERO-10 task 0; prompt/layout/goal are unchanged.
# Usage: CHECKPOINT=/path/to/checkpoint bash .../run_l3c_shared_space.sh [eb|er|ec|all|summarize]
MODE="${1:-all}"
TRIALS="${TRIALS:-20}"

run_condition() {
  local condition="$1"
  : "${CHECKPOINT:?Set CHECKPOINT to an OpenVLA-OFT checkpoint}"
  python -m experiments.robot.libero.run_physcog_libero_l1_eval \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name libero_10 --task_ids 0 --num_trials_per_task "$TRIALS" \
    --safety_oracle none --held_object_body moka_pot_1_main \
    --trajectory_track_bodies moka_pot_1_main,chefmate_8_frypan_1_main,flat_stove_1 \
    --num_open_loop_steps 1 --save_trajectory True --save_video_mode violation \
    --l3c_condition "$condition" --run_id_note "L3-C-${condition}"
}

if [[ "$MODE" == "eb" || "$MODE" == "er" || "$MODE" == "ec" ]]; then
  run_condition "$MODE"
elif [[ "$MODE" == "all" ]]; then
  run_condition eb
  run_condition er
  run_condition ec
elif [[ "$MODE" == "summarize" ]]; then
  python -m experiments.robot.libero.tasks.summarize_l3c_shared_space \
    --eb rollouts/libero_10/L3-C-eb/trajectories \
    --er rollouts/libero_10/L3-C-er/trajectories \
    --ec rollouts/libero_10/L3-C-ec/trajectories \
    --out experiments/logs/l3c_shared_space.json
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
