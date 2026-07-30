#!/usr/bin/env bash
set -euo pipefail
echo "run_l1c4_occupied_cabinet.sh is deprecated; L1-C4 now uses the native libero_object basket task." >&2
exec bash experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh "${1:-}"
