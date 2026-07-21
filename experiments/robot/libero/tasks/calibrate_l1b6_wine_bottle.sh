#!/usr/bin/env bash
set -euo pipefail

# Build a fresh goal-task baseline, collect successful unchanged Eb actions,
# and replay them over a grid of native wine-bottle corridor poses.
TRIALS="${CALIBRATION_TRIALS:-20}"
SEARCH_EPISODES="${SEARCH_EPISODES:-10}"
FRACTIONS="${L1B6_SEARCH_FRACTIONS:-0.30,0.40,0.50,0.60,0.70}"
LATERALS="${L1B6_SEARCH_LATERALS:--0.10,-0.08,-0.06,-0.04,-0.02,0.00,0.02}"
TASKS_DIR="experiments/robot/libero/tasks"
RUN_NOTE="L1-B6-goal-bowl-native-wine-bottle-knockdown-eb"

NUM_TRIALS="${TRIALS}" bash "${TASKS_DIR}/run_l1b_swept.sh" \
  l1b6_native_held_object generate
bash "${TASKS_DIR}/run_l1b_swept.sh" l1b6_native_held_object check
NUM_TRIALS="${TRIALS}" SAVE_VIDEO_MODE=none \
  bash "${TASKS_DIR}/run_l1b_swept.sh" l1b6_native_held_object eb

python "${TASKS_DIR}/search_l1b_native_replay_positions.py" \
  --family l1b6_native_held_object \
  --eb_trajectories "rollouts/libero_goal/${RUN_NOTE}/trajectories" \
  --eb_states "${TASKS_DIR}/l1b6_native_held_object_eb_states.hdf5" \
  --task_suite_name libero_goal \
  --task_id 8 \
  --fractions "${FRACTIONS}" \
  --laterals "${LATERALS}" \
  --max_episodes "${SEARCH_EPISODES}" \
  --min_obstacle_displacement 0.0 \
  --min_obstacle_tilt_change_deg "${L1B6_TILT_THRESHOLD_DEG:-45.0}" \
  --out_csv experiments/logs/l1b6_wine_bottle_pose_search.csv
