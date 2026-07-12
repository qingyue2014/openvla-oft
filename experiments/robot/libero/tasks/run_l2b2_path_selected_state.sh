#!/usr/bin/env bash
set -euo pipefail

# Repeat one selected L2-B2 basket_path initial state and evaluate it.
# Defaults reproduce the successful zero-based demo_5 from the 10-trial pilot.
#
# Usage:
#   bash experiments/robot/libero/tasks/run_l2b2_path_selected_state.sh
#   DEMO_INDEX=7 REPEATS=3 RENDER_GPU=1 \
#     bash experiments/robot/libero/tasks/run_l2b2_path_selected_state.sh

SOURCE_STATE_PATH="${SOURCE_STATE_PATH:-experiments/robot/libero/tasks/l2b2_basket_stove_on_path_initial_states.hdf5}"
DEMO_INDEX="${DEMO_INDEX:-5}"
REPEATS="${REPEATS:-5}"
RENDER_GPU="${RENDER_GPU:-1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
OUTPUT_STATE_PATH="${OUTPUT_STATE_PATH:-experiments/robot/libero/tasks/l2b2_path_demo${DEMO_INDEX}_repeat${REPEATS}.hdf5}"
RUN_ID_NOTE="${RUN_ID_NOTE:-L2-B2-Er-path-demo${DEMO_INDEX}-repeat${REPEATS}}"

python experiments/robot/libero/tasks/extract_repeated_initial_state.py \
  --input "${SOURCE_STATE_PATH}" \
  --output "${OUTPUT_STATE_PATH}" \
  --demo_index "${DEMO_INDEX}" \
  --repeats "${REPEATS}" \
  --overwrite

STATE_PATH="${OUTPUT_STATE_PATH}" \
NUM_TRIALS="${REPEATS}" \
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE}" \
RENDER_GPU="${RENDER_GPU}" \
RUN_ID_NOTE="${RUN_ID_NOTE}" \
  bash experiments/robot/libero/tasks/run_l2b2_basket_stove.sh basket_path eval

echo
echo "Evaluation finished."
echo "  Initial states: ${OUTPUT_STATE_PATH}"
echo "  Logs:           experiments/logs/EVAL-*--${RUN_ID_NOTE}.txt"
echo "  Rollouts:       rollouts/libero_10/${RUN_ID_NOTE}/"
