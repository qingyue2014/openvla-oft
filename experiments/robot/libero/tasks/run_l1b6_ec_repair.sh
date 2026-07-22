#!/usr/bin/env bash
set -euo pipefail

python experiments/robot/libero/tasks/repair_l1b6_ec_control_states.py
bash experiments/robot/libero/tasks/run_l1b_swept.sh l1b6_native_held_object check
SAVE_VIDEO_MODE=none bash experiments/robot/libero/tasks/run_l1b_swept.sh l1b6_native_held_object ec
