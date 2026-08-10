#!/usr/bin/env bash
set -euo pipefail
exec bash experiments/robot/libero/tasks/run_l1c_occupied.sh l1c5 "${1:-}"
