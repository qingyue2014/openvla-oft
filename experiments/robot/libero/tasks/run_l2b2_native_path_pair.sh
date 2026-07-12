#!/usr/bin/env bash
set -euo pipefail

# Validate and evaluate the matched L2-B2 native-layout path pair.
# Both conditions use the same BDDL and native movable-object regions; only
# the stove state and safety threshold differ.
#
# Usage:
#   bash experiments/robot/libero/tasks/run_l2b2_native_path_pair.sh probe
#   NUM_TRIALS=5 RENDER_GPU=1 \
#     bash experiments/robot/libero/tasks/run_l2b2_native_path_pair.sh pair

MODE="${1:-pair}"
NUM_TRIALS="${NUM_TRIALS:-5}"
RENDER_GPU="${RENDER_GPU:-1}"
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
FIXED_LAYOUT="${FIXED_LAYOUT:-0}"
BDDL_FILE="experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_stove_on_path.bddl"

run_probe() {
  python experiments/robot/libero/tasks/probe_l2b2_basket_stove.py \
    --bddl "${BDDL_FILE}" \
    --out_dir experiments/robot/libero/tasks/l2b2_native_path_probe \
    --num_resets 3 \
    --list_bodies
}

run_condition() {
  local variant="$1"
  NUM_TRIALS="${NUM_TRIALS}" \
  RENDER_GPU="${RENDER_GPU}" \
  SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE}" \
  FIXED_LAYOUT="${FIXED_LAYOUT}" \
    bash experiments/robot/libero/tasks/run_l2b2_basket_stove.sh "${variant}" all
}

case "${MODE}" in
  probe)
    run_probe
    ;;
  off)
    run_condition basket_path_off
    ;;
  on)
    run_condition basket_path
    ;;
  pair)
    run_probe
    run_condition basket_path_off
    run_condition basket_path
    ;;
  *)
    echo "Usage: $0 [probe|off|on|pair]" >&2
    exit 2
    ;;
esac

echo
echo "L2-B2 native-layout ${MODE} run finished."
echo "Inspect: experiments/logs/EVAL-*--L2-B2-cream-cheese-basket-stove-on-path*.txt"
